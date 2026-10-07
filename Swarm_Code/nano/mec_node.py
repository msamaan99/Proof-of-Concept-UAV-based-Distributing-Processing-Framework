"""
mec_node.py — DISTRIBUTED PROCESSING LAYER

The one program you run by hand on each Jetson. It reads which role the network
layer gave this node and runs the matching half of the application.

    ssh nano1  →  python3 mec_node.py
    ssh nano2  →  python3 mec_node.py

Same command on both boards. Nothing to edit, no flags to remember.

HOW IT KNOWS ITS ROLE
---------------------
swarm_net.py already decided. It publishes the answer, along with every node's
health, to swarm/neighbors.json:

    swarm_net.py  ──writes──►  swarm/neighbors.json   ──►  this file
                               role + every node's cpu/ram/gpu/ip/inference

    this file     ──writes──►  swarm/local_perf.json  ──►  swarm_net.py
                               measured inference time and queue depth

That handoff is why this layer has no registration protocol and no heartbeat
sockets. The network layer already discovered who is present and is already
exchanging their health twice a second, so repeating either here would be a
second, slower, less reliable copy of work already done.

  MASTER   ingest frames from the sensors, decide where each one runs, run
           inference locally or ship it to a worker, log, forward to the GCS.
  WORKER   pull frames from the master, run inference, send detections back.

If the network layer hands this node a different role — which is what happens
during a failover — this switches over in place. The model stays loaded, so the
changeover costs a second or two rather than the 10-30s a cold TensorRT load
would. You will see it happen in your terminal.

Ctrl+C to stop. It never touches the network.
"""

import argparse
import collections
import csv
import glob
import json
import logging
import math
import os
import queue
import re
import signal
import socket as pysocket
import subprocess
import sys
import threading
import time

# Before numpy, cv2 or torch. OpenBLAS misdetects the Jetson's Cortex-A57 and
# kills the process with SIGILL — no traceback, no message, nothing printed at
# all, which reads like the program silently refusing to start. The Nanos'
# .bashrc exports this but only for interactive shells, so running this by hand
# at the board's own prompt works while `ssh nano 'python3 mec_node.py'` dies
# before its first log line. The systemd unit sets it too; this covers the case
# where neither applies.
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import cv2
import numpy as np
import zmq

import frame_trace

import matplotlib
matplotlib.use("Agg")

# ── Select the scheduling algorithm — set MEC_SCHED, do not edit this ──
#
#   MEC_SCHED=lyapunov   (default)   MEC_SCHED=greedy
#   MEC_SCHED=rr                     MEC_SCHED=fixed   MEC_OFFLOAD_PCT=25
#
# This used to be three import lines with two of them commented out. A
# four-algorithm comparison means eight edits across two boards, and an edit
# that lands on the wrong board produces a run labelled as an algorithm it did
# not use — a mislabelling nothing downstream can detect, because the results
# file looks completely normal. An env var cannot be got wrong silently: it is
# in the command you ran, in the startup log line, and in the results filename.
_SCHED = (os.environ.get("MEC_SCHED") or "lyapunov").strip().lower()
if _SCHED in ("lyapunov", "lyap"):
    from scheduler_lyapunov import NodeProfiler, EWMA, SwarmLoadBalancer
elif _SCHED in ("greedy", "greedy_ect", "ect"):
    from completion_time_scheduler import NodeProfiler, EWMA, SwarmLoadBalancer
elif _SCHED in ("rr", "round_robin", "roundrobin"):
    from scheduler_rr import NodeProfiler, EWMA, SwarmLoadBalancer
elif _SCHED in ("fixed", "manual", "static"):
    from scheduler_fixed import NodeProfiler, EWMA, SwarmLoadBalancer
else:
    raise SystemExit(
        "MEC_SCHED=%r is not an algorithm. Use one of: "
        "lyapunov, greedy, rr, fixed" % _SCHED)

# Always from the shared module regardless of which scheduler is selected
# above: the routing diagnostics have to price the local option the same way
# every scheduler does, or the numbers in the log would not be the numbers the
# decision was made on.
from completion_time_scheduler import estimate_local_time

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

GCS_IP = "192.168.50.23"        # The operator laptop

PI_INGEST_PORT = 5000           # Pi     → master   JPEG frames
WORK_PORT = 5001                # master → worker   inference requests
RESULTS_PORT = 5003             # worker → master   detections
GCS_PORT = 6000                 # master → GCS      annotated frames

# A frame older than this is no longer useful.
#
# 0.6 s was calibrated when the master was a 51 ms laptop. With a Jetson master
# the budget is spent before anything can be scheduled: the master needs ~190 ms
# for its own inference and the worker ~190 + 65 ms of link, so at 600 ms the
# worker cannot hold even two frames and fails its feasibility check on every
# comparison. 1.2 s leaves the worker room for two or three frames in flight,
# which is what makes offloading possible at all on this hardware.
#
# Override with MEC_DEADLINE_MS to run the same schedulers under a tighter
# budget. This is causal, not cosmetic: enforce_deadline() compares each node's
# estimated completion time against what is left of the budget, so a smaller
# number vetoes more offloads and drops infeasible frames sooner. A timeout
# applied at the GCS instead would change only what is displayed.
#
# Set the same value on the ground station (MEC_DEADLINE_MS there too) or its
# latency colours and jitter buffer describe a budget the master is not
# enforcing — the drift this file's header warns about.
FRAME_DEADLINE_SEC = float(os.environ.get("MEC_DEADLINE_MS", "1200")) / 1000.0

# Frames the master may hold before the scheduler has looked at them. See the
# note on task_queue: this is a latency budget, not a buffer.
TASK_QUEUE_MAX = 6

# Frames the scheduler has assigned to this node and that are waiting to be
# decoded. Still JPEGs at this point, so depth here is cheap — ~38 kB a frame.
LOCAL_DECODE_QUEUE_MAX = 30

# Decoded frames waiting for the GPU. Deliberately shallow: every slot holds a
# ~921 kB BGR array, and two is already enough that the GPU never waits on the
# decoder. Backlog belongs in the queue ahead of this one, where it costs 38 kB
# a frame instead of 921 kB — which is the whole reason decoding moved to its
# own stage rather than staying on the ingest thread.
LOCAL_INFER_QUEUE_MAX = 3

# Frames queued for one worker before the dispatch path gives up and keeps them
# local. This is a second admission control, independent of the scheduler's
# deadline check: even with the deadline veto disabled, a full queue here
# silently converts an offload decision into a local one.
#
# 10 is right for operating the system — it bounds how far behind a worker can
# fall. Raise it with MEC_OUTGOING_MAX only to measure what happens with no
# admission control at all, and expect the worker's backlog to grow until frames
# die on WORKER_RESULT_TIMEOUT_SEC or FRAME_TTL_SEC instead.
OUTGOING_QUEUE_MAX = int(os.environ.get("MEC_OUTGOING_MAX", "10"))

# Honour the scheduler's choice even when the dispatch path would rather not.
#
# Two things downstream of pick() can turn an offload into a local run without
# the scheduler ever hearing about it: a full outgoing queue, and a send that
# fails because the worker's socket is blocked. Both are correct for operating a
# live system — a delivered frame beats a lost one — and both are wrong for a
# scheduler whose whole job is to hold a ratio, because the ratio it reports is
# then not the ratio that ran.
#
# On by default for the fixed-ratio scheduler and off for the others, which is
# the split that matches intent: lyapunov, greedy and rr are all trying to make
# a good decision per frame and should take the fallback; "fixed 50%" is a
# policy and should either execute or fail visibly.
# MEC_NO_DEADLINE=1 turns this on too, whatever the scheduler. Asking for the
# deadline veto to be removed means asking for the stated split to execute, and
# leaving this off would let the dispatch queue overrule it instead — the same
# override arriving by a quieter route.
STRICT_OFFLOAD = os.environ.get(
    "MEC_STRICT_OFFLOAD",
    "1" if (_SCHED in ("fixed", "manual", "static")
            or os.environ.get("MEC_NO_DEADLINE") == "1") else "0") == "1"
SWITCH_MARGIN_SEC = 0.005       # Hysteresis: minimum advantage to switch paths
FRAME_TTL_SEC = 5.0             # How long a frame waits in the store for a result

# A dispatched frame that has produced nothing after this long is written off.
# Generously past the deadline — by now the frame is dead either way and the only
# thing still at stake is the bookkeeping. Without this, a worker that dies
# mid-frame leaves its in-flight count permanently high and the scheduler keeps
# avoiding it for work it is not actually doing.
WORKER_RESULT_TIMEOUT_SEC = 2.0

# ── Failover reconnection ──────────────────────────────────────────────────
# Applied to every socket that connects out to a peer that might move or restart.
#
# ZMQ's defaults are wrong for this: heartbeats are off, so a dead peer is only
# noticed when TCP says so, and Linux retries an established connection 15 times
# with doubling backoff. A peer that disappears can go unnoticed for tens of
# seconds. ZMTP heartbeats detect it in HEARTBEAT_TIMEOUT regardless of TCP, then
# reconnection attempts start immediately.
# Retry at a fixed interval, no exponential backoff: an outage lasts as long as
# the other Nano needs to rebuild the network, and a backed-off socket would be
# waiting seconds between attempts by the end of it and add that to the real
# recovery time.
HEARTBEAT_IVL_MS = 1000
HEARTBEAT_TIMEOUT_MS = 3000
RECONNECT_IVL_MS = 200
RECONNECT_IVL_MAX_MS = 0        # 0 = constant interval, no backoff

# Everything this file writes or reads sits next to the file itself, resolved
# from __file__ rather than the working directory. That matters twice over:
# swarm_net.py is started by systemd from its own WorkingDirectory while this is
# started by hand from wherever your shell happens to be, and the two MUST agree
# on where the handoff files live. A relative path would have them writing to two
# different places, with this side waiting forever for a file that does exist —
# somewhere else.
HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(HERE)               # the yolov5 repo, one level up

STATE_DIR = os.path.join(HERE, "runtime")
NEIGHBORS_FILE = os.path.join(STATE_DIR, "neighbors.json")
LOCAL_PERF_FILE = os.path.join(STATE_DIR, "local_perf.json")
STATE_MAX_AGE_SEC = 10.0        # Older than this → the network layer is not running
STATE_POLL_SEC = 1.0
PERF_PUBLISH_SEC = 1.0
STATS_INTERVAL_SEC = 10.0

# How often the results CSV is flushed to disk. Not per row: see csv_loop.
CSV_FLUSH_INTERVAL_SEC = 1.0

# Set MEC_NO_CSV=1 to run without the results CSV at all. Intended as a
# diagnostic A/B, not a normal mode: the file it suppresses is the run's actual
# output. Telemetry is unaffected either way — it is written once a second from
# state_loop, on a thread the frame path never touches.
CSV_ENABLED = not os.environ.get("MEC_NO_CSV")

# Per-frame routing diagnostics. Off by default so a recorded run measures the
# scheduler and not the instrumentation; set MEC_VERBOSE=1 to turn it on when
# you need to know *why* a worker is not being used.
ROUTING_DIAG = bool(os.environ.get("MEC_VERBOSE"))

# Stop the run by itself once this many frames have been accounted for
# (delivered + dropped), then shut down the same way Ctrl+C does:
#
#     MEC_SCHED=rr MEC_RUN_FRAMES=5000 python3 mec_node.py
#
# Set this for a measurement run so every algorithm is compared over the same
# number of frames. Stopping by hand gives the four runs four different lengths,
# and length is not neutral: a longer run spends a larger share of itself in the
# warmed-up steady state, so the algorithm that happened to run longest looks
# better for a reason that has nothing to do with scheduling.
#
# 0 — the default — means run until Ctrl+C, which is the normal operating mode.
RUN_FRAME_LIMIT = int(os.environ.get("MEC_RUN_FRAMES", "0") or 0)

MASTER = "master"
WORKER = "worker"

RESULTS_CSV_COLUMNS = [
    "Wall Clock", "Node ID", "Frame ID", "Assigned Node",
    "Total Latency (ms)", "Network RTT (ms)", "GPU Inference (ms)",
    "Queue Length", "Decision Overhead (ms)", "Drop Reason",
]
TELEMETRY_CSV_COLUMNS = [
    "Timestamp", "Node", "Temperature_C", "Battery_Pct", "CPU_Pct", "RAM_Pct", "Infer_ms",
]

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("MEC_VERBOSE") else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mec")


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def tune_for_failover(sock, fail_fast=False):
    """Apply reconnection settings to a socket that connects out to a peer.

    Only for connecting sockets. A bound socket has nothing to reconnect to — it
    waits, and whoever comes back finds it.

    *fail_fast* sets ZMQ_IMMEDIATE, which refuses to queue for a peer that is not
    currently connected. Right for frames: anything buffered during an outage
    arrives seconds stale, gets dropped by the deadline check anyway, and
    displaces fresh frames on the way. Wrong for anything where a brief hiccup
    should be ridden out rather than discarded.
    """
    sock.setsockopt(zmq.HEARTBEAT_IVL, HEARTBEAT_IVL_MS)
    sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, HEARTBEAT_TIMEOUT_MS)
    sock.setsockopt(zmq.RECONNECT_IVL, RECONNECT_IVL_MS)
    sock.setsockopt(zmq.RECONNECT_IVL_MAX, RECONNECT_IVL_MAX_MS)
    sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
    sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 5)
    sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 2)
    if fail_fast:
        sock.setsockopt(zmq.IMMEDIATE, 1)
    return sock


def percentile(sorted_vals, pct):
    """Linear-interpolation percentile. *sorted_vals* must already be sorted."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class PeriodicReporter:
    """Timer for summary log lines.

    Per-frame logging at 10 FPS is twenty lines a second, which buries every
    message that matters. Counters plus one aggregated line on a timer replaced
    it; per-frame detail is still available at DEBUG with MEC_VERBOSE=1.
    """

    __slots__ = ("interval", "_last")

    def __init__(self, interval_sec):
        self.interval = interval_sec
        self._last = time.time()

    def due(self):
        now = time.time()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False


CLOCK_WARN_OFFSET_SEC = 0.05        # Above this, latency numbers carry real error


def warn_if_clock_unsynced():
    """Say something loud if this node's clock has not been disciplined yet.

    Worth checking at startup because the failure is silent and total. End-to-end
    latency is this node's clock minus the Pi's, and the deadline check does the
    same subtraction — so a run started before chrony has stepped the clock gives
    both garbage numbers and wrong routing, with nothing in the output to say so.

    The trap is one of timing rather than configuration. The network does not
    exist until a Nano has claimed the access point, the laptop cannot be reached
    until it has joined, and only then can chrony correct anything. Start a run
    inside that window and the first minutes of data are worthless.

    Best-effort: if chrony is not installed there is nothing to check and this
    stays quiet rather than nagging about a tool the operator chose not to use.
    """
    try:
        out = subprocess.run(["chronyc", "-n", "tracking"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return                          # chrony absent — nothing to say

    reference = re.search(r"Reference ID\s*:\s*(\S+)", out)
    offset = re.search(r"System time\s*:\s*([\d.]+) seconds", out)

    # An all-zero reference ID means chronyd is running but has never reached a
    # source — the usual state right after boot, before the swarm network exists.
    if reference and set(reference.group(1)) <= {"0"}:
        log.warning("CLOCK NOT SYNCHRONISED — chrony has no time source yet. "
                    "Latency figures and deadline drops will be wrong until it "
                    "settles. Check: chronyc tracking")
    elif offset and float(offset.group(1)) > CLOCK_WARN_OFFSET_SEC:
        log.warning("Clock is %.0f ms off the reference — every latency figure "
                    "will carry that error", float(offset.group(1)) * 1000)
    elif offset:
        log.info("Clock synchronised — %.1f ms offset", float(offset.group(1)) * 1000)


def load_model():
    """Load YOLOv5n and warm it up. Returns (model, device, names).

    Prefers the TensorRT engine if one has been built, falling back to PyTorch
    weights. Both live one directory up, alongside the yolov5 repo.

    The warmup pass is not optional: it forces CUDA context creation and
    TensorRT graph initialisation. Skip it and that cost lands on the first real
    frame instead, where it shows up as a one-off ~2s outlier in the latency data.
    """
    import torch
    # This file lives in <yolov5>/swarm/, so the repo is one level up. Both the
    # import path and the weights are resolved from there rather than from the
    # working directory — otherwise running it from anywhere but the swarm folder
    # looks for the model in the wrong place and fails with a confusing error.
    sys.path.append(YOLO_ROOT)
    from models.common import DetectMultiBackend

    device = torch.device("cuda:0")
    engine = os.path.join(YOLO_ROOT, "yolov5n.engine")
    weights = os.path.join(YOLO_ROOT, "yolov5n.pt")
    path = engine if os.path.exists(engine) else weights
    if not os.path.exists(path):
        raise SystemExit(
            f"No model found. Expected {engine} or {weights} — "
            f"put yolov5n.engine or yolov5n.pt in {YOLO_ROOT}")
    backend = "TensorRT" if path.endswith(".engine") else "PyTorch"

    t0 = time.time()
    model = DetectMultiBackend(path, device=device, fp16=True)
    model(torch.zeros((1, 3, 640, 640), device=device).half())
    log.info("Model ready — %s (%s) in %.1fs", path, backend, time.time() - t0)
    return model, device, model.names


# ═══════════════════════════════════════════════════════════════════════════
#  Reading the network layer
# ═══════════════════════════════════════════════════════════════════════════

def read_swarm_state():
    """Load swarm/neighbors.json, or None if it is missing or stale.

    Stale matters as much as missing: if swarm_net.py died, the file it left
    behind still names a role and a peer list that may be hours out of date.
    Treating that as authoritative would be worse than having nothing.
    """
    try:
        with open(NEIGHBORS_FILE, "r") as f:
            state = json.load(f)
    except (IOError, ValueError):
        return None
    if time.time() - state.get("updated", 0) > STATE_MAX_AGE_SEC:
        return None
    return state


def publish_perf(node_id, role, infer_ms, queue_depth):
    """Tell the network layer what this node measured.

    Inference time is application knowledge — the network layer has no way to
    observe it — so it goes out through here, gets folded into the telemetry
    broadcast, and reaches the other Nano's scheduler as a warm starting estimate
    rather than a cold guess.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = LOCAL_PERF_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"node_id": node_id, "role": role, "infer_ms": infer_ms,
                       "queue": queue_depth, "updated": time.time()}, f)
        os.replace(tmp, LOCAL_PERF_FILE)    # Atomic: the reader never sees a half-write
    except (IOError, OSError):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  FrameStore
# ═══════════════════════════════════════════════════════════════════════════

class FrameStore:
    """Holds frames between arrival and result, keyed "sensor_id:frame_id".

    Holds the JPEG as it arrived, never a decoded array. Decoding happens after
    the routing decision, on whichever machine is going to run inference — see
    local_decode_loop. Keeping pixels here would mean decoding frames that get
    offloaded or dropped, and it would cost about 24x the memory: a 640x480 BGR
    array is 921 kB against a ~38 kB JPEG, held for FRAME_TTL_SEC at the sensor
    frame rate, on a board that is also holding a TensorRT engine.

    Two timestamps per entry, and the difference matters:

      capture_ts    when the Pi grabbed the frame, on the Pi's clock. Reported
                    latency is measured against this. The deadline is NOT — see
                    arrival_mono.
      ts            when this node received it, on this node's wall clock. Kept
                    for diagnostics only — nothing decides anything on it, since
                    a wall clock here can step.
      arrival_mono  when this node received it, on this node's monotonic clock.
                    The deadline is measured against this, so that a frame's age
                    is a single subtraction on one clock that no other machine
                    and no time daemon can move. Wall clock is unusable here:
                    time.time() on this board has stepped by years mid-run when
                    the clock was corrected, and a deadline measured across such
                    a step drops every frame it sees.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._store = collections.OrderedDict()

    def put(self, key, raw_bytes, capture_ts=None):
        with self._lock:
            self._evict_stale()
            now = time.time()
            self._store[key] = {"ts": now, "capture_ts": capture_ts or now,
                                "arrival_mono": time.monotonic(),
                                "bytes": raw_bytes}

    def get(self, key):
        with self._lock:
            entry = self._store.get(key)
            return dict(entry) if entry is not None else None

    def pop(self, key):
        with self._lock:
            return self._store.pop(key, None)

    def _evict_stale(self):
        """Drop entries past their TTL. Caller holds the lock.

        Insertion order is arrival order, so the oldest is always first and the
        scan can stop at the first live entry.

        Monotonic, for the same reason the deadline is: a wall-clock step would
        otherwise make every held frame look older than the TTL at once and
        evict the entire store in a single pass.
        """
        now = time.monotonic()
        while self._store:
            _, entry = next(iter(self._store.items()))
            if now - entry["arrival_mono"] > FRAME_TTL_SEC:
                self._store.popitem(last=False)
            else:
                break


class SensorClockOffset:
    """Estimates each sensor's clock offset from the frames it is already sending.

    REPORTING ONLY. Nothing is dropped, scheduled, or routed on this estimate.

    It used to gate the deadline check, on the reasoning that a frame's age is
    `now - capture_ts` — this node's clock minus the Pi's — and that this class
    could take the skew back out. It could not do so reliably enough to decide
    whether a frame lives: a skew larger than the deadline drops every frame,
    and on a board with no RTC that promotes itself to master with a clock years
    off, that is not a corner case. The deadline is now measured from the frame's
    arrival here on a monotonic clock, so no estimate stands between an arriving
    frame and the scheduler.

    What remains is the reported latency figure, which genuinely does span the
    two clocks and has no other way to be computed. A bad estimate there costs
    a wrong number in a CSV, not a frame.

    So estimate the offset from traffic that is already flowing. Per frame:

        sample = recv_ts - capture_ts = network_delay - skew

    Network delay is strictly positive and varies with queueing; skew is constant
    across the window. Taking the minimum over recent frames drives the queueing
    term toward its floor and leaves

        min_sample ≈ min_network_delay - skew

    and subtracting that from the naive age cancels the skew exactly:

        corrected = (true_age - skew) - (min_network_delay - skew)
                  = true_age - min_network_delay

    What is left is an underestimate bounded by the minimum one-way delay — a few
    milliseconds on this LAN, against a 600 ms deadline. An unbounded, silent
    error is traded for a small, bounded, known one.

    Per sensor, because each Pi carries its own clock. Time-bounded rather than
    count-bounded so that a chrony step ages out of the window instead of pinning
    the minimum to a stale value for the rest of the run. A new master starts
    estimating from its first frame and is converged within a second, which is
    the property that makes this survive a failover.
    """

    WINDOW_SEC = 30.0

    # A sample this far from the current floor is not network jitter — jitter on
    # this link is milliseconds. It is one of the two clocks being stepped, and
    # the history either side of a step describes a different time base.
    STEP_THRESHOLD_SEC = 1.0

    def __init__(self):
        self._lock = threading.Lock()
        self._samples = collections.defaultdict(collections.deque)

    def observe(self, sensor, capture_ts, recv_ts):
        if not capture_ts:
            return
        sample = recv_ts - capture_ts
        with self._lock:
            window = self._samples[sensor]
            if window:
                floor = min(v for _, v in window)
                if abs(sample - floor) > self.STEP_THRESHOLD_SEC:
                    # Waiting for the stale samples to age out would leave the
                    # estimate mixing two time bases for a whole window — long
                    # enough to drop every frame on a deadline that never
                    # actually elapsed. Throw the history away and re-converge
                    # from this sample instead; it costs one frame, not thirty
                    # seconds.
                    log.warning("Clock step on '%s' — offset moved %.1fs, "
                                "discarding %d stale samples",
                                sensor, sample - floor, len(window))
                    window.clear()
            window.append((recv_ts, sample))
            cutoff = recv_ts - self.WINDOW_SEC
            while window and window[0][0] < cutoff:
                window.popleft()

    def offset(self, sensor):
        """Best estimate of (min network delay − skew) for this sensor.

        Zero until a sample exists, which degrades to the uncorrected behaviour
        rather than inventing a correction out of nothing.
        """
        with self._lock:
            window = self._samples.get(sensor)
            if not window:
                return 0.0
            return min(v for _, v in window)

    def snapshot(self):
        """Current estimate per sensor, for the summary line."""
        with self._lock:
            return {s: min(v for _, v in w)
                    for s, w in self._samples.items() if w}


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER
# ═══════════════════════════════════════════════════════════════════════════

class MasterRole:
    """Ingest, schedule, infer, log, forward."""

    def __init__(self, model, device, names, node_id):
        self.node_id = node_id
        self.model, self.device, self.names = model, device, names

        self._stop = threading.Event()
        self._threads = []
        self.ctx = zmq.Context()

        self.frames = FrameStore()
        self.clock_offsets = SensorClockOffset()
        # Per-frame timestamp trace. No-op unless MEC_TRACE_N is set, and
        # bounded even then — see frame_trace.py.
        self.trace = frame_trace.FrameTrace()
        if self.trace.enabled:
            log.info("Frame tracing on — skipping the first %d frames, then "
                     "%d per path; written to gcs_logs/frame_trace.jsonl on the GCS",
                     self.trace.skip, self.trace.limit)
        # Shallow on purpose. This queue is the frame's waiting room, and its
        # depth is latency: at the ~5 fps a Jetson master sustains, 30 slots is
        # six seconds of backlog, so a frame reached the scheduler already older
        # than any deadline worth having and was dropped without ever being
        # looked at. Six slots is ~1.2 s, which the deadline can actually absorb.
        self.task_queue = queue.Queue(maxsize=TASK_QUEUE_MAX)
        # Two stages, because decode and inference are different resources: the
        # decoder is CPU and releases the GIL, the model is the GPU. Splitting
        # them lets frame N+1 decode while frame N is still on the GPU, which is
        # the pipelining the ingest thread used to provide by decoding early —
        # except now it only ever decodes frames that are actually going to be
        # inferred here.
        self.local_decode_queue = queue.Queue(maxsize=LOCAL_DECODE_QUEUE_MAX)
        self.local_task_queue = queue.Queue(maxsize=LOCAL_INFER_QUEUE_MAX)
        self.results_queue = queue.Queue(maxsize=100)
        # Rows on their way to disk. Deep, and lossy when full by design: a
        # stalled SD card should cost log rows, never frames.
        self.csv_queue = queue.Queue(maxsize=5000)

        self.peers = {}                 # node_id -> dispatch state + profiler
        self.peers_lock = threading.Lock()

        self.gcs_sender = self.ctx.socket(zmq.PUSH)
        self.gcs_sender.setsockopt(zmq.LINGER, 0)
        self.gcs_sender.setsockopt(zmq.SNDHWM, 60)
        # No fail_fast here: the ground station does not move, and a moment of
        # buffering during a hiccup is better than a gap in the operator's feed.
        tune_for_failover(self.gcs_sender)
        self.gcs_sender.connect(f"tcp://{GCS_IP}:{GCS_PORT}")

        self.local_infer_ewma = EWMA(alpha=0.3, initial=0.050)
        self._local_busy = False
        self._last_sched_tick = time.monotonic()
        self._route_diag = collections.Counter()
        self._diag_reporter = PeriodicReporter(STATS_INTERVAL_SEC)
        self._sensors_seen = set()

        self.balancer = SwarmLoadBalancer(
            self.local_infer_ewma,
            switch_margin_sec=SWITCH_MARGIN_SEC,
            frame_deadline_sec=FRAME_DEADLINE_SEC,
        )

        self._stats_lock = threading.Lock()
        self._stats = {
            "ingested": 0, "delivered": 0, "dropped": 0, "logger_overflow": 0,
            "csv_overflow": 0,
            "late_results": 0, "forward_missing": 0,
            "by_node": collections.Counter(), "drop_reasons": collections.Counter(),
            "latencies": collections.deque(maxlen=600),
        }
        # Membership changes waiting to go to the ground station. Bounded and
        # lossy on purpose — see _announce_swarm.
        self.swarm_events = queue.Queue(maxsize=64)
        self._last_table = {}
        self._csv_file = None
        self._telemetry_file = None
        self._limit_hit = False     # MEC_RUN_FRAMES fires once, not every frame after

    # -- lifecycle -----------------------------------------------------

    def start(self):
        log.info("MASTER — scheduler=%s deadline=%dms run_frames=%s",
                 self._algorithm_name(), FRAME_DEADLINE_SEC * 1000,
                 RUN_FRAME_LIMIT or "unlimited")
        for name, target in (
            ("pi-ingest", self.ingest_loop),
            ("results-recv", self.results_loop),
            ("reaper", self.reaper_loop),
            ("gcs-writer", self.writer_loop),
            ("csv-writer", self.csv_loop),
            ("local-decode", self.local_decode_loop),
            ("local-infer", self.local_inference_loop),
            ("swarm-state", self.state_loop),
            ("scheduler", self.scheduler_loop),
        ):
            t = threading.Thread(target=self._guarded(name, target), daemon=True, name=name)
            t.start()
            self._threads.append(t)
        log.info("Listening on :%d for sensors, forwarding to GCS %s:%d",
                 PI_INGEST_PORT, GCS_IP, GCS_PORT)

    def stop(self):
        """Tear down so this process can take a different role.

        Order matters. The stop flag first so every loop sees it, then poison
        pills to release threads parked on a blocking queue.get() — those do not
        notice a ZMQ context dying — then the context itself.
        """
        if self._stop.is_set():
            return
        log.info("Master stopping…")
        self._stop.set()

        for q in (self.local_decode_queue, self.local_task_queue,
                  self.results_queue, self.csv_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        with self.peers_lock:
            for p in self.peers.values():
                try:
                    p["outgoing"].put_nowait(None)
                except queue.Full:
                    pass

        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []

        self.gcs_sender.close()
        with self.peers_lock:
            for p in self.peers.values():
                try:
                    p["socket"].close()
                except Exception:
                    pass
            self.peers.clear()

        # destroy(), not term(): term() waits for every socket in the context to
        # be closed and blocks forever if a thread was torn down still holding
        # one. destroy() closes them itself, and the threads are already joined.
        try:
            self.ctx.destroy(linger=0)
        except Exception:
            pass
        for f in (self._csv_file, self._telemetry_file):
            try:
                if f:
                    f.close()
            except Exception:
                pass

        with self._stats_lock:
            s = self._stats
            lat = sorted(s["latencies"])
            log.info("Session totals — %d received, %d delivered, %d dropped%s",
                     s["ingested"], s["delivered"], s["dropped"],
                     f", avg {sum(lat)/len(lat):.0f}ms" if lat else "")

    def _guarded(self, name, target):
        """Wrap a thread body so a crash is loud instead of silent.

        A thread dying quietly is the worst failure mode here: if the CSV writer
        goes down, results_queue fills and every producer blocks forever on put()
        with no log line and no exit — the whole master frozen while still
        apparently running.
        """
        def wrapper():
            try:
                target()
            except zmq.error.ContextTerminated:
                pass
            except Exception as e:
                log.error("Thread '%s' died: %s", name, e, exc_info=True)
        return wrapper

    def _algorithm_name(self):
        """Short name for logs and the results filename.

        The fixed-ratio scheduler carries its setting in the name — a file
        called results_fixed_003.csv would not tell you, or the report script,
        which ratio produced it, and the ratio is the only thing that
        distinguishes one fixed run from another.
        """
        name = type(self.balancer).__module__.replace("scheduler_", "")
        if name == "completion_time_scheduler":
            return "greedy_ect"
        if name == "fixed":
            return "fixed%d" % round(self.balancer.target * 100)
        return name

    # -- peers, from the network layer ---------------------------------

    def state_loop(self):
        """Track who is available, using what the network layer publishes.

        This replaces the registration handshake and the heartbeat monitor the
        master used to run. Both were rediscovering, more slowly, what the
        network layer already knows: who is present, at which address, and how
        healthy they are.
        """
        os.makedirs(STATE_DIR, exist_ok=True)
        serial = len(glob.glob(f"{STATE_DIR}/telemetry_*.csv")) + 1
        self._telemetry_file = open(f"{STATE_DIR}/telemetry_{serial:03d}.csv", "w", newline="")
        writer = csv.writer(self._telemetry_file)
        writer.writerow(TELEMETRY_CSV_COLUMNS)

        warned = False
        while not self._stop.is_set():
            state = read_swarm_state()
            if state is None:
                if not warned:
                    log.warning("No fresh %s — is swarm_net running? "
                                "Running master-only until it appears", NEIGHBORS_FILE)
                    warned = True
            else:
                warned = False
                table = state.get("nodes", {})
                self._last_table = table
                self._sync_peers(table)
                now = round(time.time(), 2)
                for node_id, e in table.items():
                    writer.writerow([now, node_id, e.get("gpu_temp_c", ""),
                                     e.get("battery_pct", "N/A"), e.get("cpu_pct", "N/A"),
                                     e.get("ram_pct", "N/A"), e.get("infer_ms", "N/A")])
                self._telemetry_file.flush()

            # Both stages, not just the GPU queue. This number is broadcast and
            # read by the other node's scheduler as this node's backlog, so
            # publishing only the shallow inference queue would advertise a
            # busy master as idle.
            publish_perf(self.node_id, MASTER,
                         round(self.local_infer_ewma.value * 1000, 1),
                         self._local_pending())
            self._stop.wait(STATE_POLL_SEC)

    def _sync_peers(self, table):
        """Add, update, or retire workers based on the neighbour table."""
        for node_id, entry in table.items():
            if node_id == self.node_id:
                continue

            usable = (entry.get("online") and entry.get("mec_active")
                      and entry.get("mec_role") == WORKER and entry.get("ip"))
            ip = entry.get("ip")

            with self.peers_lock:
                peer = self.peers.get(node_id)

                if not usable:
                    if peer and peer["alive"]:
                        peer["alive"] = False
                        self._announce_swarm("left", node_id, "no longer reachable")
                        log.warning("Worker '%s' is no longer available", node_id)
                    continue

                if peer is None:
                    profiler = NodeProfiler()
                    # Seed from what the peer reports about itself, so the first
                    # dispatch is an informed guess rather than a generic one.
                    if entry.get("infer_ms"):
                        profiler.infer_time.update(entry["infer_ms"] / 1000.0)
                    peer = {"ip": ip, "alive": True, "busy": 0, "inflight": {},
                            "profiler": profiler, "socket": self._worker_socket(ip),
                            "outgoing": queue.Queue(maxsize=OUTGOING_QUEUE_MAX),
                            # Consumed by scheduler_loop on its next tick: a
                            # worker that has just appeared gets a frame spent
                            # on measuring it rather than waiting for its seeded
                            # estimate to win a comparison on its own.
                            "force_probe": True}
                    self.peers[node_id] = peer
                    threading.Thread(
                        target=self._guarded(f"send-{node_id}",
                                             lambda n=node_id, p=peer: self.sender_loop(n, p)),
                        daemon=True, name=f"send-{node_id}").start()
                    self._announce_swarm("joined", node_id, "available at %s" % ip)
                    log.info("Worker '%s' available at %s", node_id, ip)

                elif peer["ip"] != ip:
                    # A worker that moved has restarted, so everything it had in
                    # flight is gone with it. Resetting the bookkeeping matters:
                    # otherwise it carries a phantom queue depth forever and the
                    # scheduler keeps avoiding it for work it is not doing.
                    log.info("Worker '%s' moved to %s — resetting its state", node_id, ip)
                    peer["socket"].close()
                    peer["socket"] = self._worker_socket(ip)
                    peer["ip"] = ip
                    peer["busy"] = 0
                    peer["inflight"].clear()
                    # Reliability is about a node's recent conduct, and a node
                    # that has just restarted has no recent conduct. Carrying the
                    # old score across meant a worker could come back healthy and
                    # still be excluded on the strength of failures its previous
                    # process committed.
                    peer["profiler"].reliability = EWMA(alpha=0.1, initial=1.0)
                    # A restarted worker re-warms its TensorRT engine, so its
                    # first inferences are transients again — see WARMUP_SAMPLES.
                    peer["profiler"].samples = 0
                    # And its link statistics describe a connection to a dead
                    # process. See the rejoin branch below for why network_rtt
                    # in particular cannot be allowed to survive this.
                    peer["profiler"].network_rtt = EWMA(alpha=0.3, initial=0.010)
                    peer["profiler"].rtt_jitter = EWMA(alpha=0.3, initial=0.003)
                    peer["force_probe"] = True
                    self._announce_swarm("rejoined", node_id, "restarted at %s" % ip)

                elif not peer["alive"]:
                    peer["alive"] = True
                    peer["busy"] = 0
                    peer["inflight"].clear()
                    peer["profiler"].reliability = EWMA(alpha=0.1, initial=1.0)
                    peer["profiler"].samples = 0
                    # Forget the link measurements too, not just reliability.
                    #
                    # network_rtt describes a TCP connection to a process that
                    # no longer exists, and it is the one statistic here that
                    # cannot heal on its own: it only takes a sample when a
                    # result arrives, so a value large enough to make the node
                    # fail the deadline test stops the very traffic that would
                    # correct it. Carrying it across a rejoin means a worker can
                    # come back perfectly healthy and still never be chosen,
                    # because of a number measured against its previous life.
                    #
                    # Reset to the constructor's seeds so a returning node is
                    # judged on what it does from here.
                    peer["profiler"].network_rtt = EWMA(alpha=0.3, initial=0.010)
                    peer["profiler"].rtt_jitter = EWMA(alpha=0.3, initial=0.003)
                    peer["force_probe"] = True
                    self._announce_swarm("rejoined", node_id, "link stats reset")
                    log.info("Worker '%s' is back — link stats reset", node_id)

                peer["profiler"].update_telemetry(
                    temp_celsius=entry.get("gpu_temp_c", 0.0) or 0.0,
                    battery_pct=entry.get("battery_pct"),
                    cpu_pct=entry.get("cpu_pct"),
                    ram_pct=entry.get("ram_pct"))

    def _worker_socket(self, ip):
        s = self.ctx.socket(zmq.PUSH)
        s.setsockopt(zmq.SNDHWM, 2)
        s.setsockopt(zmq.LINGER, 0)
        # fail_fast: a frame queued for a worker that is not currently connected
        # would arrive too late to be useful. Better to fail the send now, which
        # the sender loop records as a dispatch failure, and let the scheduler
        # route the next frame elsewhere.
        tune_for_failover(s, fail_fast=True)
        s.connect(f"tcp://{ip}:{WORK_PORT}")
        return s

    # -- stats ---------------------------------------------------------

    def _emit(self, frame_id, dets, node, latency, rtt_ms, gpu_ms, q_len, dec_ms, reason=""):
        """Hand a finished or failed frame to the CSV writer.

        Bounded wait, not an unbounded one: if the writer falls behind this loses
        a log row and says so, instead of stalling the thread that called it.
        Losing observability is bad; deadlocking the pipeline it describes is worse.
        """
        try:
            self.results_queue.put(
                (frame_id, dets, node, latency, rtt_ms, gpu_ms, q_len, dec_ms, reason),
                timeout=1.0)
        except queue.Full:
            with self._stats_lock:
                self._stats["logger_overflow"] += 1

    def _report(self):
        with self._stats_lock:
            s = self._stats
            delivered, dropped = s["delivered"], s["dropped"]
            lat = sorted(s["latencies"])
            by_node = dict(s["by_node"])
            overflow = s["logger_overflow"]
            csv_lost = s["csv_overflow"]
            late = s["late_results"]
            missing = s["forward_missing"]
            reasons = s["drop_reasons"].most_common(3)

        total = delivered + dropped
        if total == 0:
            log.info("No frames yet — waiting on the sensors")
            return

        offloaded = sum(c for n, c in by_node.items() if n not in ("MASTER", "DROPPED"))
        parts = [f"{total} frames"]
        if lat:
            parts.append(f"lat avg {sum(lat)/len(lat):.0f}ms p95 {percentile(lat, 95):.0f}ms")
        if delivered:
            parts.append(f"offload {100*offloaded/delivered:.0f}%")
        parts.append(f"drop {100*dropped/total:.1f}%")
        if csv_lost:
            # Said out loud rather than left silent: a results file missing rows
            # is only safe to analyse if you know how many are missing.
            parts.append(f"csv rows lost {csv_lost}")
        log.info(" | ".join(parts))

        rows = []
        for node_id, e in sorted(self._last_table.items()):
            bits = [node_id if e.get("online") else f"{node_id} (down)"]
            if e.get("gpu_temp_c"):
                bits.append("%.0fC" % e["gpu_temp_c"])
            if e.get("cpu_pct") is not None:
                bits.append("cpu %.0f%%" % e["cpu_pct"])
            if e.get("ram_pct") is not None:
                bits.append("ram %.0f%%" % e["ram_pct"])
            rows.append(" ".join(bits))
        if rows:
            log.info("  nodes: %s", " · ".join(rows))
        skews = self.clock_offsets.snapshot()
        if skews:
            log.info("  clock: %s", " · ".join(
                f"{s} {v*1000:+.0f}ms" for s, v in sorted(skews.items())))
        if reasons and dropped:
            log.info("  drops: %s", ", ".join(f"{r}×{c}" for r, c in reasons))
        if overflow:
            log.warning("  %d result rows lost — the CSV writer is behind", overflow)
        # Surfaced on the periodic line because both of these mean the results
        # CSV and the display disagree about what happened, and that is exactly
        # the disagreement that hid the post-rejoin fault. A handful right after
        # a worker rejoins is the engine warming up and is expected; a count
        # that keeps climbing is not.
        if late or missing:
            log.warning("  %d late results written off, %d results with no frame "
                        "to display", late, missing)

    # -- ingest --------------------------------------------------------

    def ingest_loop(self):
        """Accept frames from any number of sensors.

        Keyed "sensor_id:frame_id" so two Pis both counting from zero cannot
        overwrite each other.
        """
        rx = self.ctx.socket(zmq.PULL)
        rx.setsockopt(zmq.LINGER, 0)
        rx.setsockopt(zmq.RCVHWM, 500)
        rx.bind(f"tcp://0.0.0.0:{PI_INGEST_PORT}")

        while not self._stop.is_set():
            if not rx.poll(timeout=500):
                continue
            key = "unknown"
            try:
                parts = rx.recv_multipart()
                sid, fid, ts_bytes, img = parts[:4]
                recv_ts = time.time()
                # Fifth part is the Pi's own trace stamps. Older sensors send
                # four parts, so its absence is not an error.
                pi_stamps = {}
                if len(parts) > 4 and parts[4]:
                    try:
                        pi_stamps = json.loads(parts[4].decode())
                    except (ValueError, UnicodeDecodeError):
                        pass
                sensor = sid.decode()
                key = f"{sensor}:{fid.decode()}"
                if sensor not in self._sensors_seen:
                    self._sensors_seen.add(sensor)
                    log.info("Sensor '%s' streaming", sensor)
                try:
                    capture_ts = float(ts_bytes.decode())
                    # Every frame is also a clock sample. Feeding the estimator
                    # here, on the raw arrival time, is what keeps the deadline
                    # check independent of whether chrony has converged.
                    self.clock_offsets.observe(sensor, capture_ts, recv_ts)
                except (ValueError, UnicodeDecodeError):
                    capture_ts = recv_ts

                # Deliberately NOT decoded here. This thread's job is to get
                # frames off the socket; decoding on it made every frame wait
                # ~6.3 ms before the scheduler could even look at it, and that
                # decode was thrown away for every frame that was then offloaded
                # or dropped. It now happens after the decision, on the machine
                # that is going to use the pixels — see local_decode_loop.
                #
                # This also matters most exactly when things are worst: when the
                # link returns after an outage, a few hundred frames arrive at
                # once, and this loop can now drain them at socket speed instead
                # of 6.3 ms apiece.
                #
                # A corrupt JPEG is therefore no longer caught here. It surfaces
                # in local_decode_loop or on the worker, both of which drop it
                # with a reason attached rather than skipping it silently.
                self.frames.put(key, img, capture_ts)
                self.trace.begin(key, pi_stamps)
                self.trace.stamp(key, "master_recv", recv_ts)
                # Measured again here rather than trusting the Pi's figure, so
                # the size is present even for a sensor running the older
                # four-part build that sends no stamps at all.
                self.trace.note(key, "bytes", len(img))
                self._admit(key)
                with self._stats_lock:
                    self._stats["ingested"] += 1

            except Exception as e:
                log.error("Inbound frame error: %s", e)

    def _admit(self, key):
        """Queue a frame for scheduling, shedding the oldest if there is no room.

        The queue used to drop the frame that had just arrived and keep the
        thirty already waiting. For a live feed that is backwards: the newest
        frame is the only one anyone wants, and the old ones ahead of it are
        precisely why it could not be admitted. Worse, the scheduler then spent
        its time on frames that had aged past the deadline while queued, so they
        were dropped after being decoded and never reached the display — a full
        queue turned into a total outage rather than a reduced frame rate.

        Shedding from the front instead keeps the waiting room current: the
        scheduler always sees recent frames, they still have deadline left to
        spend, and load shows up as a lower frame rate rather than as nothing.
        """
        while True:
            try:
                self.task_queue.put_nowait(key)
                return
            except queue.Full:
                pass
            try:
                stale = self.task_queue.get_nowait()
            except queue.Empty:
                continue        # Drained underneath us; try the put again.
            self.frames.pop(stale)
            self._emit(stale, [], "DROPPED", -1, 0.0, 0.0,
                       self.task_queue.qsize(), 0.0, "SHED_STALE")

    # -- dispatch ------------------------------------------------------

    def sender_loop(self, node_id, peer):
        """Ship queued frames to one worker.

        One thread per worker, so a slow link to one node cannot stall dispatch
        to another and the scheduler never blocks on a socket.
        """
        while not self._stop.is_set():
            item = peer["outgoing"].get()
            if item is None:
                return
            key, frame_bytes, capture_ts, dec_ms, q_len = item
            dispatch_ts = time.time()
            self.trace.stamp(key, "dispatch", dispatch_ts)
            try:
                meta = json.dumps({"frame_id": key, "capture_ts": capture_ts,
                                   "dispatch_ts": dispatch_ts, "dec_ms": dec_ms,
                                   "q_len": q_len,
                                   # Per frame: true only for one the master is
                                   # actually tracing, so frames inside the
                                   # warm-up window or past the quota cost the
                                   # worker nothing.
                                   "trace": self.trace.tracking(key)}).encode()
                # NOBLOCK matters: a PUSH socket to a peer that has gone away
                # fills its buffer and then blocks forever, which used to wedge
                # this thread permanently — the worker stayed unusable even after
                # it came back.
                peer["socket"].send_multipart([meta, frame_bytes], flags=zmq.NOBLOCK)
                with self.peers_lock:
                    # Depth is recorded at dispatch, not read back at completion,
                    # because by then the queue this frame actually waited behind
                    # has already drained. results_loop needs it to tell the
                    # worker's queueing apart from the link's latency.
                    depth_at_dispatch = peer["busy"]
                    self.trace.note(key, "q_worker", depth_at_dispatch)
                    peer["busy"] += 1
                    peer["inflight"][key] = (dispatch_ts, depth_at_dispatch)
                    peer["profiler"].record_dispatch_outcome(True)
                log.debug("→ %s to %s", key, node_id)
            except zmq.error.Again:
                # The worker's pipe is full. That is backpressure, not failure —
                # SNDHWM here is 2 against a worker that needs ~160 ms a frame,
                # so this fires during ordinary congestion, and charging it to
                # reliability retired healthy nodes permanently. Fall the frame
                # back to local rather than binning it: the master can still
                # produce a result, and a late result beats no result.
                with self.peers_lock:
                    peer["profiler"].record_backpressure()
                self._fallback_local(key, dec_ms, q_len, "WORKER_LINK_BLOCKED")
            except Exception as e:
                log.error("Send to '%s' failed for %s: %s", node_id, key, e)
                with self.peers_lock:
                    peer["profiler"].record_dispatch_outcome(False)
                self._emit(key, [], "DROPPED", -1, 0.0, 0.0, q_len, dec_ms,
                           "WORKER_SEND_FAILED")

    def _fallback_local(self, key, dec_ms, q_len, reason):
        """Try to run a frame here after its intended worker refused it.

        Called from the sender threads, which is why it goes through the same
        bounded local queue as the scheduler rather than touching the model: the
        frame joins the back of the local queue and the local inference thread
        picks it up in order. Only if that queue is also full is the frame
        actually lost, and then the recorded reason is the one that explains why
        it could go nowhere.
        """
        # Under STRICT_OFFLOAD the frame was promised to the worker, and running
        # it here instead would quietly move it into the local column — the
        # reported split would then differ from the one the scheduler enforced.
        # Losing the frame is the honest outcome, and the reason names why.
        if STRICT_OFFLOAD:
            self.frames.pop(key)
            self._emit(key, [], "DROPPED", -1, 0.0, 0.0, q_len, dec_ms, reason)
            return

        entry = self.frames.get(key)
        if entry is not None:
            try:
                # The JPEG, not pixels — this frame was headed for a worker, so
                # nothing on this board has decoded it. It enters at the decode
                # stage like any other locally-run frame.
                self.local_decode_queue.put_nowait(
                    (key, entry["bytes"], entry["capture_ts"], dec_ms, q_len))
                log.debug("%s refused (%s) — running %s locally", reason, key, key)
                return
            except queue.Full:
                reason = "LOCAL_QUEUE_FULL"
        self.frames.pop(key)
        self._emit(key, [], "DROPPED", -1, 0.0, 0.0, q_len, dec_ms, reason)

    def reaper_loop(self):
        """Write off dispatches that never came back.

        This is what makes the reliability score mean anything. Without it, the
        only failures ever recorded are send errors — and an asynchronous PUSH to
        a dead peer does not raise one. A worker could stop answering entirely
        while its score sat at a perfect 1.0.
        """
        while not self._stop.is_set():
            now = time.time()
            expired = []
            with self.peers_lock:
                for node_id, peer in self.peers.items():
                    for key, (ts, _depth) in list(peer["inflight"].items()):
                        if now - ts > WORKER_RESULT_TIMEOUT_SEC:
                            del peer["inflight"][key]
                            peer["busy"] = max(0, peer["busy"] - 1)
                            peer["profiler"].record_dispatch_outcome(False)
                            expired.append((node_id, key))
            # Emitted outside the lock: _emit can wait, and holding the peer lock
            # while waiting would stall the scheduler.
            for node_id, key in expired:
                log.warning("No result from '%s' for %s after %.0fs — written off",
                            node_id, key, WORKER_RESULT_TIMEOUT_SEC)
                self._emit(key, [], "DROPPED", -1, 0.0, 0.0, 0, 0.0,
                           "WORKER_RESULT_TIMEOUT")
            self._stop.wait(1.0)

    def results_loop(self):
        """Collect detections coming back from workers."""
        rx = self.ctx.socket(zmq.PULL)
        rx.setsockopt(zmq.LINGER, 0)
        rx.setsockopt(zmq.RCVHWM, 100)
        rx.bind(f"tcp://0.0.0.0:{RESULTS_PORT}")

        while not self._stop.is_set():
            if not rx.poll(timeout=500):
                continue
            try:
                msg = json.loads(rx.recv_string())
                node_id = msg.get("worker_id")
                frame_id = msg.get("frame_id")
                capture_ts = msg.get("capture_ts", 0)
                dispatch_ts = msg.get("dispatch_ts", 0)
                dec_ms, q_len = msg.get("dec_ms", 0), msg.get("q_len", 0)
                infer_sec = msg.get("infer_time", 0.04)
                err = msg.get("error")

                with self.peers_lock:
                    peer = self.peers.get(node_id)
                    if peer is None:
                        log.debug("Result from unknown worker '%s' — ignored", node_id)
                        continue
                    peer["busy"] = max(0, peer["busy"] - 1)
                    tracked = peer["inflight"].pop(frame_id, None)
                    # None means "no idea", NOT "the depth was zero", and the
                    # difference is the whole bug below. Defaulting a missing
                    # entry to 0 let a frame the reaper had already written off
                    # pass the depth_at_dispatch == 0 test and be recorded as a
                    # pristine link measurement.
                    depth_at_dispatch = tracked[1] if tracked else None

                if err:
                    log.error("Worker '%s' failed on %s: %s", node_id, frame_id, err)
                    self._emit(frame_id, [], "DROPPED", -1, 0.0, 0.0, q_len, dec_ms,
                               "WORKER_INFERENCE_ERROR")
                    continue

                if tracked is None:
                    # The result came back after reaper_loop had already given
                    # up on this dispatch at WORKER_RESULT_TIMEOUT_SEC.
                    #
                    # This is the post-rejoin failure, and it is worth spelling
                    # out because every symptom of it points somewhere else.
                    # A Nano that has just rejoined re-warms its TensorRT engine,
                    # so its first inferences overrun the 2 s write-off. The
                    # reaper then does three things: emits the frame as DROPPED,
                    # tells the GCS, and pops the JPEG out of the frame store.
                    # Seconds later the real result arrives here and, until this
                    # branch existed, was processed as if nothing had happened:
                    #
                    #   · _forward found no JPEG and returned in silence, so the
                    #     frame never reached the display.  Offloaded frames
                    #     simply stopped appearing while local ones kept coming.
                    #   · the frame was counted a second time, now as delivered,
                    #     so the totals no longer summed to the frames ingested.
                    #   · worst of all, the missing inflight entry read as
                    #     depth 0, so the whole write-off delay — seconds of it —
                    #     was recorded as pure network RTT. At alpha 0.3 one such
                    #     sample moves network_rtt from 10 ms to well over a
                    #     second, estimate_time then exceeds the frame deadline
                    #     for every future frame, and enforce_deadline vetoes the
                    #     worker permanently. Nothing recovers it: network_rtt
                    #     only takes a sample when a result comes back, and no
                    #     result can come back from a node nothing is sent to.
                    #     StalenessProbe cannot break the loop either, because
                    #     its probe is vetoed by the same deadline test.
                    #
                    # That absorbing state is exactly "it offloads nothing and
                    # only master frames appear". The frame is already accounted
                    # for, so the only thing to do with it is the worker's own
                    # honest timing of its own inference, which carries none of
                    # the write-off delay and is what lets the node earn its way
                    # back once it is genuinely warm.
                    with self.peers_lock:
                        peer["profiler"].record_inference(infer_sec)
                    with self._stats_lock:
                        self._stats["late_results"] += 1
                    log.warning("Late result from '%s' for %s — already written "
                                "off, not displayed and not used to time the link",
                                node_id, frame_id)
                    self.trace.drop(frame_id)
                    continue

                recv_ts = time.time()
                self.trace.merge(frame_id, msg.get("stamps"))
                self.trace.stamp(frame_id, "result_recv", recv_ts)
                # Reported latency spans the Pi's clock and this node's. The
                # measured offset takes their skew out, leaving an error bounded
                # by the minimum one-way delay instead of by however far apart
                # the two clocks happen to be — which is what lets this figure
                # stand up with no time daemon running anywhere.
                total_latency = (recv_ts - capture_ts
                                 - self.clock_offsets.offset(frame_id.split(":", 1)[0]))
                # Network RTT with this node's own queueing removed.
                #
                # Subtracting infer_sec is not enough on its own. The worker
                # starts its timer after recv_multipart returns, so infer_sec
                # covers decode and inference but NOT the time the frame spent
                # waiting in its RCVHWM=2 buffer behind other frames. That wait
                # was landing in this EWMA and being read as link latency, which
                # is wrong twice over: estimate_time already charges for the
                # worker's queue in its (pending + 1) x infer_time term, so the
                # same congestion was counted once as compute and again as
                # network. It also does not decay — network_rtt only takes a
                # sample when a result comes back, so once a burst inflated it,
                # a node that stopped being chosen kept the inflated value with
                # nothing able to bring it down.
                #
                # Only frames that queued behind nothing are used to measure the
                # link, which is what depth_at_dispatch == 0 means. For those,
                # (recv - dispatch) - infer IS the network cost, exactly, with
                # nothing to estimate.
                #
                # The previous attempt subtracted an estimated
                # depth x infer_time instead, and estimating was the flaw: on a
                # cold worker the EWMA still read 132 ms while the frame had
                # really waited 460 ms behind a warming TensorRT engine, so
                # 330 ms of queueing landed in network_rtt regardless. Combined
                # with the jitter it drove the node's estimate to 786 ms against
                # a 600 ms deadline — permanently unschedulable, and unable to
                # correct itself because correction needs the results that being
                # unschedulable prevents.
                #
                # Measuring rarely and correctly beats measuring often and
                # wrongly. Under load few frames qualify, and network_rtt then
                # holds its last clean value rather than absorbing congestion —
                # and the probe guarantees depth-0 dispatches keep happening.
                if depth_at_dispatch == 0:
                    if dispatch_ts > 0:
                        pure_rtt = max(0.0, (recv_ts - dispatch_ts) - infer_sec)
                    else:
                        pure_rtt = max(0.0, total_latency - infer_sec)
                    with self.peers_lock:
                        peer["profiler"].record_rtt(pure_rtt)
                else:
                    pure_rtt = peer["profiler"].network_rtt.value

                with self.peers_lock:
                    peer["profiler"].record_inference(infer_sec)

                log.debug("← %s from %s (rtt %.0fms)", frame_id, node_id, pure_rtt * 1000)
                self._emit(frame_id, msg.get("detections", []), node_id.upper(),
                           total_latency, round(pure_rtt * 1000, 1),
                           round(infer_sec * 1000, 1), q_len, dec_ms)
            except Exception as e:
                log.error("Bad result message: %s", e)

    # -- output --------------------------------------------------------

    def writer_loop(self):
        """Forward finished frames to the GCS. Touches no disk.

        Disk was on this path until it caused the fault it was recording. This
        thread is the only consumer of results_queue AND the only caller of
        _forward, so whatever it spends per frame sets the drain rate for
        everything behind it. An fsync per row to a Jetson's SD card was far too
        expensive to sit there, and the cost landed exactly where it did the most
        harm: offloading roughly doubles the frame rate through here, so the
        thread fell behind precisely when the worker was being used.

        The cascade from that: results_queue fills, _emit blocks up to a second
        inside results_loop, peer["busy"] stays high because its decrement is
        stuck behind that, and the scheduler reads an idle worker as too loaded
        to meet the deadline. The display goes quiet at the same moment and for
        the same reason. One stalled write, both symptoms.

        Rows now go to csv_loop through a deep queue, so disk latency cannot
        reach the frame path at all.
        """
        reporter = PeriodicReporter(STATS_INTERVAL_SEC)
        while not self._stop.is_set():
            if reporter.due():
                self._report()
            # Drained here because this thread owns gcs_sender. Cheap: the
            # queue is empty on every tick except the handful where the swarm
            # actually changed shape.
            self._drain_swarm_events()
            try:
                item = self.results_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                fid, dets, node, lat, rtt_ms, gpu_ms, q_len, dec_ms, reason = item
            except ValueError:
                log.error("Malformed result row — skipped")
                continue

            try:
                latency_ms = round(lat * 1000, 1) if lat >= 0 else -1
                # Wall Clock and Node ID lead every row. Without them, results
                # from before and after a failover live in two files on two
                # machines with no shared axis, and recovery cannot be measured.
                self._queue_csv_row([round(time.time(), 3), self.node_id, fid, node,
                                     latency_ms, rtt_ms, gpu_ms, q_len, dec_ms, reason])

                with self._stats_lock:
                    self._stats["by_node"][node] += 1
                    if node == "DROPPED":
                        self._stats["dropped"] += 1
                        self._stats["drop_reasons"][reason or "UNKNOWN"] += 1
                    else:
                        self._stats["delivered"] += 1
                        self._stats["latencies"].append(latency_ms)
                    accounted = self._stats["delivered"] + self._stats["dropped"]

                self._check_run_limit(accounted)

                if node == "DROPPED":
                    self._notify_drop(fid, reason, q_len)
                    self.frames.pop(fid)
                    continue
                self._forward(fid, dets, node, lat)
            except Exception as e:
                # Never let one bad row take this thread down: if it dies,
                # results_queue fills and every producer blocks behind it.
                log.error("Result handling failed for %s: %s", fid, e)

    def _check_run_limit(self, accounted):
        """End the run once MEC_RUN_FRAMES frames have been accounted for.

        Raises SIGINT at itself rather than setting the stop event directly.
        That is deliberate: SIGINT is the path Ctrl+C already takes, so the run
        ends through shutdown code that gets exercised every single time you use
        this program, instead of through a second shutdown path that only ever
        runs at the end of a measurement — which is exactly where a bug would
        cost you the run you were measuring.

        The caller keeps going after this returns. The frame in hand is still
        forwarded and still written; the handler only sets an event, and the
        threads wind down on their own a moment later.
        """
        if not RUN_FRAME_LIMIT or self._limit_hit:
            return
        if accounted < RUN_FRAME_LIMIT:
            return
        self._limit_hit = True
        log.info("Reached MEC_RUN_FRAMES=%d (%d accounted) — ending the run",
                 RUN_FRAME_LIMIT, accounted)
        os.kill(os.getpid(), signal.SIGINT)

    def _queue_csv_row(self, row):
        """Hand one row to csv_loop. Never blocks, never raises.

        Lossy when the queue is full, and deliberately so: a stalled disk should
        cost log rows, not frames. The loss is counted and reported rather than
        being silent, because a results file that quietly lost rows is worse than
        one that says how many it lost.
        """
        if not CSV_ENABLED:
            return
        try:
            self.csv_queue.put_nowait(row)
        except queue.Full:
            with self._stats_lock:
                self._stats["csv_overflow"] += 1

    def csv_loop(self):
        """Own the results CSV, on its own thread, off the frame path."""
        if not CSV_ENABLED:
            log.warning("MEC_NO_CSV set — running with no results CSV")
            return
        os.makedirs(STATE_DIR, exist_ok=True)
        algo = self._algorithm_name()
        serial = len(glob.glob(f"{STATE_DIR}/results_{algo}_*.csv")) + 1
        path = f"{STATE_DIR}/results_{algo}_{serial:03d}.csv"
        self._csv_file = open(path, "w", newline="")
        writer = csv.writer(self._csv_file)
        writer.writerow(RESULTS_CSV_COLUMNS)
        self._csv_file.flush()
        log.info("Results → %s", path)

        last_flush = time.monotonic()
        while not self._stop.is_set():
            try:
                row = self.csv_queue.get(timeout=0.5)
            except queue.Empty:
                # Quiet moment, and nothing is waiting on this thread: land the
                # tail of the file now, so a run that ends quietly still ends
                # with a complete CSV.
                self._csv_file.flush()
                last_flush = time.monotonic()
                continue
            if row is None:
                break
            try:
                writer.writerow(row)
                now = time.monotonic()
                if now - last_flush >= CSV_FLUSH_INTERVAL_SEC:
                    self._csv_file.flush()
                    last_flush = now
            except Exception as e:
                log.error("CSV write failed: %s", e)

        # Drain whatever is still queued before the file closes.
        try:
            while True:
                row = self.csv_queue.get_nowait()
                if row is not None:
                    writer.writerow(row)
        except queue.Empty:
            pass
        try:
            self._csv_file.flush()
        except Exception:
            pass

    def _announce_swarm(self, state, node_id, detail=""):
        """Queue a membership change for the ground station.

        Deliberately does NOT send here. This runs on state_loop's thread while
        holding peers_lock, and gcs_sender belongs to writer_loop — ZMQ sockets
        are not thread safe, so touching it from two threads is a crash waiting
        for the demo rather than a bug that shows up in testing. The notice goes
        on a queue and writer_loop, which already owns the socket, sends it.

        Dropping on a full queue is correct: these are rare, and a membership
        notice is worth nothing if delivering it stalls the frame path.
        """
        try:
            self.swarm_events.put_nowait((state, node_id, detail, time.time()))
        except queue.Full:
            pass

    def _drain_swarm_events(self):
        """Send queued membership changes to the GCS. writer_loop's thread only."""
        while True:
            try:
                state, node_id, detail, ts = self.swarm_events.get_nowait()
            except queue.Empty:
                return
            try:
                notice = json.dumps({
                    "event": "worker", "state": state, "node": node_id,
                    "detail": detail, "ts": round(ts, 3),
                    "master": self.node_id,
                }).encode()
                self.gcs_sender.send_multipart([notice, b""], flags=zmq.NOBLOCK)
            except Exception:
                pass            # Best effort, exactly like a drop notice

    def _notify_drop(self, fid, reason, q_len):
        """Tell the GCS a frame is gone. Metadata only, no image.

        Lets the ground station log the drop with its real cause and skip past
        it, instead of holding playback open for a frame that is never arriving.
        """
        try:
            notice = json.dumps({"event": "drop", "frame_id": fid,
                                 "reason": reason or "UNKNOWN", "queue_len": q_len}).encode()
            self.gcs_sender.send_multipart([notice, b""], flags=zmq.NOBLOCK)
        except Exception:
            pass                # Best effort; a lost notice must not stall anything

    def _forward(self, fid, dets, node, lat):
        data = self.frames.pop(fid)
        if data is None:
            # A result with no frame to attach it to. This used to return in
            # complete silence, and that silence is why "offloaded frames stop
            # appearing on the display" was so hard to place: the CSV row had
            # already been written one line earlier in writer_loop, so the run
            # log showed the frame delivered by the worker while the GCS had
            # never been sent it. The two disagreed and nothing said so.
            #
            # The late-result branch in results_loop now catches the cause, so
            # reaching here means something else took the frame — a TTL
            # eviction under a backlog, most likely. Counted and logged either
            # way: a frame that is recorded as delivered but never displayed
            # must never again be invisible.
            with self._stats_lock:
                self._stats["forward_missing"] += 1
            log.warning("No stored frame for %s (ran on %s) — result recorded "
                        "but nothing to display", fid, node)
            return
        try:
            # The path is only known now: "MASTER" means the frame never left
            # this board (Time 01), anything else names the worker that ran it
            # (Time 02). claim() enforces the per-path quota, so a long run
            # keeps the first N of each and stops paying after that.
            t_gcs_send = time.time()
            self.trace.stamp(fid, "gcs_send", t_gcs_send)
            path = "local" if node == "MASTER" else "offload"
            traced = self.trace.export(fid) if self.trace.claim(fid, path) else None
            self.trace.drop(fid)

            payload = {
                "event": "frame", "source": node,
                "labels": sorted({d["cls"] for d in dets}), "detections": dets,
                "latency_ms": round(lat * 1000, 1),     # capture → inference done
                # The budget this master actually enforces. The GCS draws its
                # deadline line and sizes its jitter buffer from a constant of
                # its own, and the two silently disagreed for weeks — 0.6 s here
                # against 1.2 s there. Sending it lets the receiver notice.
                "deadline_ms": round(FRAME_DEADLINE_SEC * 1000, 1),
                "capture_ts": data["capture_ts"],       # legacy cross-clock figure
                # This node's clock as the frame goes to the wire. The GCS
                # differences it against its own arrival time and keeps the
                # minimum, which yields transport jitter without either machine
                # needing to agree with the other about the time of day.
                "sent_ts": t_gcs_send,
                "frame_id": fid,
            }
            if traced:
                payload["stamps"] = traced
            meta = json.dumps(payload).encode()
            self.gcs_sender.send_multipart([meta, data["bytes"]], flags=zmq.NOBLOCK)
        except zmq.error.Again:
            pass                # GCS buffer full — the live view skips a frame
        except Exception as e:
            log.error("GCS forward failed for %s: %s", fid, e)

    # -- inference -----------------------------------------------------

    def _local_pending(self):
        """Frames this node has already committed to running itself.

        Both stages plus the one on the GPU right now. The scheduler prices the
        local option as (pending + 1) x inference time, so undercounting here
        makes local look cheaper than it is and routes work to a node that is
        already behind. When decoding moved into its own stage this had to count
        both queues — counting only the inference queue would have hidden up to
        LOCAL_DECODE_QUEUE_MAX frames of committed work from the cost model.
        """
        return (self.local_decode_queue.qsize()
                + self.local_task_queue.qsize()
                + (1 if self._local_busy else 0))

    def local_decode_loop(self):
        """Decode frames the scheduler assigned here, and hand them to the GPU.

        This is the only place a master-bound frame is decoded, and it runs
        after the routing decision — so an offloaded frame is never decoded on
        this board, and a frame dropped at the deadline is never decoded at all.

        cv2.imdecode releases the GIL, so this genuinely runs alongside
        inference on another core rather than just interleaving with it.
        """
        while not self._stop.is_set():
            item = self.local_decode_queue.get()
            if item is None:
                return
            key, raw, capture_ts, dec_ms, q_len = item
            self.trace.stamp(key, "decode_start")
            try:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                log.error("Decode failed on %s: %s", key, e)
                img = None
            self.trace.stamp(key, "decode_done")

            # Corrupt frames used to be skipped silently on the ingest thread,
            # which meant a sensor sending garbage looked like a sensor sending
            # nothing. Now it is a drop with a reason, and it shows up in the
            # results CSV like every other one.
            if img is None:
                self.frames.pop(key)
                self._emit(key, [], "DROPPED", -1, 0.0, 0.0, q_len, dec_ms,
                           "CORRUPT_FRAME")
                continue

            # Blocking put, with a timeout so shutdown cannot wedge here if the
            # inference thread is already gone. The queue ahead is deliberately
            # shallow, so blocking is the normal way back-pressure reaches this
            # stage — it is not an error path.
            while not self._stop.is_set():
                try:
                    self.local_task_queue.put(
                        (key, img, capture_ts, dec_ms, q_len), timeout=0.2)
                    break
                except queue.Full:
                    continue

    def local_inference_loop(self):
        """Run inference on this node's own GPU."""
        import torch
        from utils.general import non_max_suppression, scale_coords
        from utils.dataloaders import letterbox

        while not self._stop.is_set():
            item = self.local_task_queue.get()
            if item is None:
                return
            key, img, capture_ts, dec_ms, q_len = item
            self._local_busy = True
            t0 = time.time()
            self.trace.stamp(key, "infer_start", t0)
            try:
                fmt = letterbox(img, new_shape=(640, 640), auto=False)[0]
                fmt = np.ascontiguousarray(fmt.transpose((2, 0, 1))[::-1])
                tensor = torch.from_numpy(fmt).to(self.device).half() / 255.0
                if len(tensor.shape) == 3:
                    tensor = tensor[None]

                pred = non_max_suppression(self.model(tensor), 0.60, 0.55, max_det=15)
                results = []
                if len(pred[0]) > 0:
                    pred[0][:, :4] = scale_coords(tensor.shape[2:], pred[0][:, :4],
                                                  img.shape).round()
                    results = [{"box": [int(x) for x in d[:4]], "conf": float(d[4]),
                                "cls": self.names[int(d[5])]} for d in pred[0]]

                duration = time.time() - t0
                self.trace.stamp(key, "infer_done")
                self.local_infer_ewma.update(duration)
                local_latency = (time.time() - capture_ts
                                 - self.clock_offsets.offset(key.split(":", 1)[0]))
                self._emit(key, results, "MASTER", local_latency,
                           0.0, round(duration * 1000, 1), q_len, dec_ms)
            except Exception as e:
                log.error("Local inference failed on %s: %s", key, e)
                self.local_infer_ewma.update(time.time() - t0)
                self._emit(key, [], "DROPPED", -1, 0.0, 0.0, q_len, dec_ms,
                           "LOCAL_INFERENCE_ERROR")
            finally:
                self._local_busy = False

    # -- scheduler -----------------------------------------------------

    def _log_routing_diag(self, candidates, estimates, remaining, decision):
        """Account for every frame that did not offload, and say which term did it.

        Aggregate result rows cannot answer this. A worker that has been ruled
        out simply stops appearing in the CSV, and "no rows from NANO2" reads the
        same whether it was excluded as unreliable, priced out by an inflated RTT
        estimate, or vetoed by the deadline. Those have different fixes, so the
        counters separate them:

          excluded    estimate_time returned inf — battery, RAM, or reliability
          infeasible  finishable in principle, but not before the deadline
          costlier    feasible and available; the scheduler preferred local
          chosen      offloaded

        A run that is 100% local with `costlier` dominant is a tuning problem.
        The same run with `infeasible` dominant is a deadline or capacity
        problem. With `excluded` dominant it is a health-signal problem. Set
        MEC_VERBOSE=1 for the per-node estimate breakdown behind these.
        """
        for wid in candidates:
            est = estimates.get(wid, math.inf)
            if math.isinf(est):
                bucket = "excluded"
            elif est > remaining:
                bucket = "infeasible"
            elif decision == wid:
                bucket = "chosen"
            else:
                bucket = "costlier"
            self._route_diag[f"{wid}:{bucket}"] += 1

        if not self._diag_reporter.due():
            return
        summary = ", ".join(f"{k}={v}" for k, v in sorted(self._route_diag.items()))
        log.info("Routing — local_est=%.0fms remaining=%.0fms | %s",
                 estimates.get("local", 0.0) * 1000, remaining * 1000, summary or "no workers")
        with self.peers_lock:
            for wid, p in self.peers.items():
                pr = p["profiler"]
                log.info("  %s: infer=%.0fms rtt=%.0fms ijit=%.0fms rjit=%.0fms "
                         "rel=%.2f busy=%d excluded=%s backpressure=%d",
                         wid, pr.infer_time.value * 1000, pr.network_rtt.value * 1000,
                         pr.infer_jitter.value * 1000, pr.rtt_jitter.value * 1000,
                         pr.reliability.value, p["busy"], pr.is_excluded(),
                         pr.backpressure_events)
        self._route_diag.clear()

    def scheduler_loop(self):
        """Route each frame to whichever node can finish it soonest."""
        while not self._stop.is_set():
            try:
                key = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            entry = self.frames.get(key)
            if entry is None:
                continue

            # Age since this frame arrived here, on this node's monotonic clock.
            #
            # Deliberately NOT measured from the Pi's capture_ts. That was a
            # subtraction across two machines' clocks, and it made the drop
            # decision depend on those clocks agreeing. They did not: this node
            # has come up with a clock years off, been promoted to master while
            # still wrong, and then been stepped by the time daemon mid-run.
            # Every one of those made "now - capture_ts" enormous and dropped
            # 100% of frames for minutes at a time, on a deadline that had not
            # actually elapsed. SensorClockOffset was built to correct that
            # subtraction and could not do it reliably enough to gate frames on.
            #
            # One clock, one subtraction, no daemon, no estimator. Monotonic
            # rather than time.time() so a clock correction on this very board
            # cannot move it either.
            #
            # The cost is that the deadline no longer covers the sensor hop:
            # a frame delayed in transit arrives looking fresh. That hop is not
            # measurable without cross-clock arithmetic, which is the thing this
            # is getting rid of. Reported latency still spans it — see the
            # total_latency and local_latency figures, which keep using
            # capture_ts and the offset estimate.
            frame_age = time.monotonic() - entry["arrival_mono"]
            frame_bytes = len(entry["bytes"])
            local_temp = self._last_table.get(self.node_id, {}).get("gpu_temp_c", 0.0) or 0.0

            tick_now = time.monotonic()
            sched_dt = min(1.0, tick_now - self._last_sched_tick)
            self._last_sched_tick = tick_now

            with self.peers_lock:
                candidates = {n: {"pending": p["outgoing"].qsize() + p["busy"],
                                  "profiler": p["profiler"]}
                              for n, p in self.peers.items() if p["alive"]}
                # Sample every candidate's depth every tick, including ones this
                # frame will not go to. That is what lets the queue-trend term
                # see a worker filling up one tick before the next dispatch to it
                # would have revealed the same thing.
                # A worker that has just joined is made overdue for a probe, so
                # the next routing decision spends one frame on it instead of
                # waiting for its seeded estimate to win a comparison against a
                # master that has real measurements behind it.
                #
                # This is what makes "it offloads again once the worker is back"
                # a guarantee rather than a likelihood. Without it the node is
                # merely *eligible*: it still has to beat the master on numbers
                # that are all seeds, and if it loses that comparison it is not
                # measured, so the seeds stand and it loses the next one too.
                # Round robin has no probe and needs none — rotation reaches
                # every live node regardless — hence the getattr rather than an
                # assumption about which scheduler is loaded.
                probe = getattr(self.balancer, "_probe", None)
                for wid, p in self.peers.items():
                    if p.get("force_probe") and p["alive"]:
                        p["force_probe"] = False
                        if probe is not None and hasattr(probe, "force"):
                            probe.force(wid)
                            log.info("Worker '%s' joined — forcing a probe so it "
                                     "is measured on the next frame", wid)

                for info in candidates.values():
                    info["profiler"].record_queue_observation(info["pending"])
                    # Heal reliability for anything still heartbeating. This
                    # belongs here rather than in a scheduler because it is
                    # health bookkeeping, and all three schedulers depend on the
                    # reliability floor not being permanent — see
                    # RELIABILITY_RECOVERY_PER_SEC.
                    info["profiler"].record_liveness_tick(sched_dt)

            self.trace.stamp(key, "sched_start")
            t0 = time.perf_counter()
            decision = self.balancer.pick(
                local_pending=self._local_pending(),
                remote_candidates=candidates, frame_age=frame_age,
                frame_bytes=frame_bytes, local_temp_celsius=local_temp)
            dec_ms = (time.perf_counter() - t0) * 1000
            self.trace.stamp(key, "sched_done")
            q_len = self.task_queue.qsize()
            self.trace.note(key, "q_sched", q_len)

            # Off unless MEC_VERBOSE=1. It re-derives every candidate's estimate
            # and takes the peer lock a second time on every frame, which is real
            # per-frame work in the loop whose throughput the results measure —
            # not something to leave running during a recorded run.
            if ROUTING_DIAG and candidates:
                diag_est = {"local": estimate_local_time(
                    self._local_pending(), self.local_infer_ewma, local_temp)}
                with self.peers_lock:
                    for wid, info in candidates.items():
                        diag_est[wid] = info["profiler"].estimate_time(
                            info["pending"], frame_bytes)
                self._log_routing_diag(candidates, diag_est,
                                       FRAME_DEADLINE_SEC - frame_age, decision)

            if decision == "drop":
                self.frames.pop(key)
                self._emit(key, [], "DROPPED", -1, 0.0, 0.0, q_len, dec_ms,
                           "DEADLINE_EXCEEDED")
                continue

            if decision != "local":
                with self.peers_lock:
                    peer = self.peers.get(decision)
                if peer is not None:
                    if STRICT_OFFLOAD:
                        # Wait for room rather than rerouting. The scheduler said
                        # this frame goes to the worker, so it goes to the worker
                        # or it dies here — either way the realized ratio is the
                        # one that was asked for. Bounded so a dead worker cannot
                        # wedge the scheduler thread.
                        deadline = time.monotonic() + FRAME_DEADLINE_SEC
                        while time.monotonic() < deadline and not self._stop.is_set():
                            try:
                                peer["outgoing"].put(
                                    (key, entry["bytes"], entry["capture_ts"],
                                     dec_ms, q_len), timeout=0.05)
                                break
                            except queue.Full:
                                continue
                        else:
                            with self.peers_lock:
                                peer["profiler"].record_backpressure()
                            self.frames.pop(key)
                            self._emit(key, [], "DROPPED", -1, 0.0, 0.0, q_len,
                                       dec_ms, "WORKER_QUEUE_FULL")
                        continue
                    try:
                        peer["outgoing"].put_nowait(
                            (key, entry["bytes"], entry["capture_ts"], dec_ms, q_len))
                        continue
                    except queue.Full:
                        # Backlogged, not broken — the queue-depth signal already
                        # tells the scheduler this node is loaded, and blaming
                        # reliability on top of that is what used to exclude a
                        # working node for the rest of the run.
                        with self.peers_lock:
                            peer["profiler"].record_backpressure()
                        log.debug("Worker '%s' backlogged — keeping %s local", decision, key)

            try:
                # Read before the put, so it counts frames already ahead of
                # this one rather than including it. _local_busy says whether
                # the GPU is mid-frame — one more wait this frame must serve
                # out, and the term that explained the 317 ms on 2026-09-04.
                # Both stages, for the same reason _local_pending counts both.
                self.trace.note(key, "q_local",
                                self.local_decode_queue.qsize()
                                + self.local_task_queue.qsize())
                self.trace.note(key, "local_busy", int(self._local_busy))
                # Still a JPEG. It is decoded one stage later, by whichever
                # thread is about to put it on the GPU.
                self.local_decode_queue.put_nowait(
                    (key, entry["bytes"], entry["capture_ts"], dec_ms, q_len))
            except queue.Full:
                self.frames.pop(key)
                self._emit(key, [], "DROPPED", -1, 0.0, 0.0, q_len, dec_ms,
                           "LOCAL_QUEUE_FULL")


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════

class WorkerRole:
    """Pull frames from the master, run inference, send detections back.

    Holds no state the swarm depends on and makes no routing decisions, which is
    what makes workers interchangeable — the master chooses between them purely
    on measured behaviour, and adding a fourth Nano needs no change anywhere.

    There is no registration handshake any more. The network layer already
    announced this node and is already broadcasting its health, and the master
    picks workers up from that.
    """

    def __init__(self, model, device, names, node_id, master_ip):
        self.node_id = node_id
        self.master_ip = master_ip
        self.model, self.device, self.names = model, device, names
        self._stop = threading.Event()
        self._threads = []
        self.ctx = zmq.Context()
        self.infer_ewma = EWMA(alpha=0.2, initial=0.050)
        self._done = 0
        self._failed = 0
        self._pending = 0

    def start(self):
        log.info("WORKER — master at %s, listening on :%d", self.master_ip, WORK_PORT)
        for name, target in (("perf", self.perf_loop), ("inference", self.inference_loop)):
            t = threading.Thread(target=self._guarded(name, target), daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self):
        if self._stop.is_set():
            return
        log.info("Worker stopping…")
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        try:
            self.ctx.destroy(linger=0)
        except Exception:
            pass
        log.info("Session totals — %d frames processed, %d failed", self._done, self._failed)

    def _guarded(self, name, target):
        def wrapper():
            try:
                target()
            except zmq.error.ContextTerminated:
                pass
            except Exception as e:
                log.error("Thread '%s' died: %s", name, e, exc_info=True)
        return wrapper

    def perf_loop(self):
        """Publish measured inference time for the network layer to broadcast.

        This is how the master learns what this node is capable of before ever
        sending it anything — the first dispatch is an informed estimate rather
        than a generic seed.
        """
        while not self._stop.is_set():
            publish_perf(self.node_id, WORKER,
                         round(self.infer_ewma.value * 1000, 1), self._pending)
            self._stop.wait(PERF_PUBLISH_SEC)

    def inference_loop(self):
        import torch
        from utils.general import non_max_suppression, scale_coords
        from utils.dataloaders import letterbox

        rx = self.ctx.socket(zmq.PULL)
        rx.setsockopt(zmq.LINGER, 0)
        # Only two frames may queue. A deep buffer would let a backlog build that
        # the master cannot see, so its estimate of this node would stay
        # optimistic while the real wait grew. Shallow means congestion shows up
        # as rising RTT almost immediately — a signal the scheduler acts on.
        rx.setsockopt(zmq.RCVHWM, 2)
        rx.bind(f"tcp://0.0.0.0:{WORK_PORT}")

        tx = self.ctx.socket(zmq.PUSH)
        tx.setsockopt(zmq.LINGER, 0)
        tx.setsockopt(zmq.SNDHWM, 10)
        # The master can move to another board underneath us. Same address, new
        # machine — so the old connection has to be noticed and dropped quickly.
        tune_for_failover(tx)
        tx.connect(f"tcp://{self.master_ip}:{RESULTS_PORT}")

        reporter = PeriodicReporter(STATS_INTERVAL_SEC)
        window = []

        while not self._stop.is_set():
            if reporter.due():
                if window:
                    log.info("%d frames | infer avg %.0fms",
                             self._done, 1000 * sum(window) / len(window))
                    window = []
                else:
                    log.info("Idle — no work from the master this interval")

            if not rx.poll(timeout=500):
                continue

            frame_id, capture_ts, dispatch_ts, dec_ms, q_len = "unknown", 0.0, 0.0, 0.0, 0
            try:
                parts = rx.recv_multipart()
                # Before the JSON parse, so the stamp covers everything this
                # board does with the frame — the master differences it against
                # its own dispatch_ts to get the master→worker hop.
                t_worker_recv = time.time()
                if len(parts) != 2:
                    raise ValueError(f"expected 2 message parts, got {len(parts)}")
                meta = json.loads(parts[0].decode())
                frame_id = meta.get("frame_id", "unknown")
                capture_ts = meta.get("capture_ts", 0.0)
                dispatch_ts = meta.get("dispatch_ts", 0.0)
                dec_ms, q_len = meta.get("dec_ms", 0.0), meta.get("q_len", 0)
                want_trace = bool(meta.get("trace"))

                self._pending = 1
                t0 = time.time()
                frame = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("cv2.imdecode returned None — corrupt JPEG")

                img = letterbox(frame, new_shape=(640, 640), auto=False)[0]
                img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
                tensor = torch.from_numpy(img).to(self.device).half() / 255.0
                if len(tensor.shape) == 3:
                    tensor = tensor[None]

                pred = non_max_suppression(self.model(tensor), 0.60, 0.55, max_det=15)
                results = []
                if len(pred[0]) > 0:
                    pred[0][:, :4] = scale_coords(tensor.shape[2:], pred[0][:, :4],
                                                  frame.shape).round()
                    for *xyxy, conf, cls in pred[0]:
                        results.append({"box": [int(x) for x in xyxy], "conf": float(conf),
                                        "cls": self.names[int(cls)]})

                duration = time.time() - t0
                t_worker_done = time.time()
                self.infer_ewma.update(duration)
                self._done += 1
                window.append(duration)
                reply = {"worker_id": self.node_id, "frame_id": frame_id,
                         "capture_ts": capture_ts, "dispatch_ts": dispatch_ts,
                         "dec_ms": dec_ms, "q_len": q_len,
                         "detections": results, "infer_time": duration}
                if want_trace:
                    reply["stamps"] = {"worker_recv": t_worker_recv,
                                       "worker_done": t_worker_done}
                self._reply(tx, reply)
                log.debug("Processed %s in %.0fms (%d detections)",
                          frame_id, duration * 1000, len(results))

            except Exception as e:
                self._failed += 1
                log.error("Inference failed on '%s': %s", frame_id, e)
                # Always reply, even on failure. Silence would leave the master
                # counting this frame as in flight until its reaper times it out,
                # and cost this node reliability for a frame it did answer for.
                self._reply(tx, {"worker_id": self.node_id, "frame_id": frame_id,
                                 "capture_ts": capture_ts, "dispatch_ts": dispatch_ts,
                                 "dec_ms": dec_ms, "q_len": q_len,
                                 "detections": [], "infer_time": 0.0, "error": str(e)})
            finally:
                self._pending = 0

    def _reply(self, tx, payload):
        try:
            tx.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
        except zmq.error.Again:
            log.error("Could not return %s — the master's buffer is full",
                      payload.get("frame_id"))
        except Exception as e:
            log.error("Could not return %s: %s", payload.get("frame_id"), e)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def own_ip(master_ip):
    """The address the master will see us on.

    Opening a UDP socket toward the master and reading back the local end picks
    the right interface without parsing `ip addr`, and re-reads correctly after
    re-associating to a network another Nano rebuilt.
    """
    s = pysocket.socket(pysocket.AF_INET, pysocket.SOCK_DGRAM)
    try:
        s.connect((master_ip, 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def start_role(role, model, device, names, node_id, master_ip):
    if role == MASTER:
        node = MasterRole(model, device, names, node_id)
    else:
        node = WorkerRole(model, device, names, node_id, master_ip)
    node.start()
    return node


def main():
    parser = argparse.ArgumentParser(description="UAV swarm MEC node")
    parser.add_argument("--role", choices=[MASTER, WORKER],
                        help="force a role instead of reading it from the network layer")
    parser.add_argument("--node-id", help="override the node name (default: from swarm_net)")
    parser.add_argument("--master-ip", default="192.168.50.1",
                        help="master address when forcing the worker role")
    args = parser.parse_args()

    stop = threading.Event()

    def handle_signal(signum, _frame):
        log.info("Stopping (signal %d)", signum)
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    warn_if_clock_unsynced()

    # Loaded once, before any role is taken, and reused for the life of the
    # process. The single biggest lever on failover time: a cold TensorRT load is
    # 10-30s on a Nano and would otherwise dominate every role change.
    model, device, names = load_model()

    # ── Forced role: ignore the network layer entirely ──
    if args.role:
        node_id = args.node_id or pysocket.gethostname()
        log.info("Role forced to %s (node=%s)", args.role.upper(), node_id)
        node = start_role(args.role, model, device, names, node_id, args.master_ip)
        try:
            while not stop.is_set():
                stop.wait(1.0)
        finally:
            node.stop()
        return

    # ── Normal path: follow the network layer ──
    node = None
    current = None
    announced = False

    try:
        while not stop.is_set():
            state = read_swarm_state()
            if state is None:
                if not announced:
                    log.warning("Waiting for the network layer — no fresh %s. "
                                "Check: systemctl status swarm_net", NEIGHBORS_FILE)
                    announced = True
                stop.wait(STATE_POLL_SEC)
                continue
            announced = False

            role = state.get("role")
            if role not in (MASTER, WORKER):
                stop.wait(STATE_POLL_SEC)
                continue

            if role != current:
                if current is not None:
                    log.warning("Network role changed: %s → %s", current.upper(), role.upper())
                node_id = args.node_id or state.get("node_id") or pysocket.gethostname()
                master_ip = state.get("master_address", args.master_ip)
                if role == MASTER:
                    log.info("Role: MASTER — this node holds %s, so it owns the network "
                             "and schedules the swarm", master_ip)
                else:
                    log.info("Role: WORKER — the master is at %s, this node runs "
                             "inference for it (as %s)", master_ip, own_ip(master_ip))
                node = start_role(role, model, device, names, node_id, master_ip)
                current = role

            # Hold until the network layer says otherwise. Polling the published
            # state rather than coordinating directly keeps the two programs
            # independent: either can restart without the other noticing.
            while not stop.is_set():
                fresh = read_swarm_state()
                if fresh is None or fresh.get("role") != current:
                    break
                stop.wait(STATE_POLL_SEC)

            if not stop.is_set() and node is not None:
                log.warning("This node's role changed — switching over")
                node.stop()
                node, current = None, None

    finally:
        if node is not None:
            node.stop()
        log.info("Stopped — the network is untouched")


if __name__ == "__main__":
    main()
