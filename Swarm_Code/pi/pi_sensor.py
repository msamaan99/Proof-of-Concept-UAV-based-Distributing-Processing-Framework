"""
pi_sensor.py

Sensor node for the UAV swarm distributed processing framework.  Runs on each
Raspberry Pi.

  1. Capture frames from the Pi Camera using Picamera2.
  2. JPEG-compress and stream them to the MEC master over ZMQ PUSH.
  3. Reconnect the camera with exponential backoff if it disappears.
  4. Count and report dropped frames rather than losing them silently.

Nothing here needs to change when the master fails over to the other Nano.  The
master's address is the access point's gateway address, so whichever board is
running the network is reachable at the same place — and ZMQ reconnects a PUSH
socket on its own once the endpoint comes back.  A failover shows up here as a
burst of backpressure drops and then normal service.

Camera backend:
  Picamera2 (the native libcamera binding) for CSI Pi Cameras, which is the
  supported path on Raspberry Pi OS Bookworm and later.  Falls back to OpenCV
  V4L2 for USB cameras when Picamera2 is unavailable.
"""

import fcntl
import json
import logging
import os
import signal
import sys
import threading
import time

import cv2
import zmq

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("MEC_VERBOSE") else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sensor")


class PeriodicReporter:
    """Timer for summary log lines, so nothing logs per frame.

    At 10 FPS a per-frame line is ten a second, which buries anything that
    matters. Counters plus one aggregated line on a timer replaced it.
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


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration — change SENSOR_ID per Pi
# ═══════════════════════════════════════════════════════════════════════════

# The master's address is the access point's gateway address, so whichever Nano
# is currently running the network is reachable here.  Nothing on the Pi changes
# when the master fails over to the other board.
# Overridable only so the watchdog and reconnect paths can be exercised against
# a local endpoint without the rig powered up.  The default is the real value
# and nothing on either Pi sets the variable.
MASTER_IP = os.environ.get("PI_MASTER_IP", "192.168.50.1")
PI_INGEST_PORT = int(os.environ.get("PI_INGEST_PORT", "5000"))

# Unique per Pi (pi1, pi2, pi3 …).  The master keys frames as
# "sensor_id:frame_id", so two Pis both counting from zero cannot collide — but
# only if this is actually unique.
#
# Set with the environment variable, do not edit this line per board:
#
#     SENSOR_ID=pi2 python3 pi_sensor.py
#
# Editing it means keeping two divergent copies of this file, and the failure
# mode when you get it wrong is silent and destructive — both Pis claim "pi1",
# their frame ids interleave in one FrameStore keyspace, and each overwrites the
# other's frames. The run looks like it worked and every number in it is wrong.
# An env var is visible in the command you ran and in the startup log line.
SENSOR_ID = os.environ.get("PI_SENSOR_ID", "pi1").strip()

# Frames per second offered by THIS Pi. PI_TARGET_FPS overrides this number;
# the value written here is only the fallback used when that variable is unset.
#
# 20 against a master that needs ~50 ms a frame is 100% of one board and about
# half the pair, so offloading is necessary but the deadline check is not firing
# on every frame. At 30 the deadline gate was overriding the schedulers often
# enough that all three converged and the comparison stopped discriminating.
#
# _open_camera asks the camera for this rate explicitly rather than accepting
# whatever it came up in. Whether it was granted is printed at startup, and the
# achieved rate is printed every interval, so a shortfall shows in the log
# instead of being assumed away.
TARGET_FPS = float(os.environ.get("PI_TARGET_FPS", "20"))

# 0-100. Override with PI_JPEG_QUALITY.
#
# 80 rather than 95, because the payload curve is steep at the top and the
# detection quality curve is flat there. Measured on a 640x480 frame of
# comparable detail:
#
#     q75    25 kB     6.2 Mbps at 30 fps
#     q80    31 kB     7.7 Mbps      <-- here
#     q95   107 kB    26.4 Mbps
#
# The last fifteen points of quality cost 3.4x the bytes. YOLOv5n letterboxes to
# 640x640 and runs at conf 0.60, well above the threshold where q80 artefacts
# change a detection, so those bytes buy nothing the model can use — they only
# compete with the master's offload and GCS traffic on the same radio.
JPEG_QUALITY = int(os.environ.get("PI_JPEG_QUALITY", "80"))

# Never drop a captured frame to protect the frame rate. Override with PI_STRICT=0.
#
# The sender used to be NOBLOCK: when the master fell behind, the frame was
# discarded and the frame id was NOT advanced, so the loss left no gap in the
# sequence and nothing downstream could see it had happened. That is the right
# behaviour for a live feed and the wrong behaviour for a measurement, because
# it silently converts "the master is saturated" into "the Pi sent fewer frames".
#
# Strict mode blocks until the frame goes or SEND_TIMEOUT_MS expires. Back
# pressure then shows up honestly as a lower achieved rate rather than as
# invisible shedding, and anything that still fails to send is counted and
# logged rather than absorbed.
STRICT_SEND = os.environ.get("PI_STRICT", "1") != "0"
SEND_TIMEOUT_MS = 1000

CAMERA_INDEX = 0                # /dev/video index, USB fallback only
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

STATS_INTERVAL_SEC = 10.0       # How often to print the one-line summary

CAMERA_MAX_RETRIES = 10         # Consecutive reconnect attempts before giving up
CAMERA_RETRY_BASE_SEC = 1.0     # Doubles each attempt
CAMERA_RETRY_CAP_SEC = 30.0

# Send buffer depth.  At 10 FPS each buffered frame is another 100 ms of age by
# the time it is looked at, so the buffer stays small: better to drop a frame the
# master would have received too late anyway than to deliver a stale one and push
# every frame behind it further out of date.
ZMQ_SEND_HWM = 4                # ≈400 ms of buffer before backpressure

# ── Failover reconnection ──────────────────────────────────────────────────
# This socket is the one thing in the swarm that never restarts, so it has to
# survive the master moving between Nanos on its own.
#
# The address never changes — 192.168.50.1 is the access point's gateway address,
# and whichever Nano holds the access point holds it — so there is nothing to
# rediscover.  What has to be handled is the dead TCP connection to the Nano that
# just died.
#
# Left at ZMQ's defaults that takes far longer than it should.  Heartbeats are
# off, so ZMQ only learns the peer is gone when TCP says so, and Linux retries an
# established connection for 15 attempts with doubling backoff — 0.2s, 0.4s,
# 0.8s … 12.8s, 25.6s.  If the new access point comes up at t=10s, the next
# retransmit might not fire until t=25s, and only then does the RST from the new
# Nano tear the old connection down.
#
# ZMTP heartbeats fix that properly: the dead peer is detected in HEARTBEAT_TIMEOUT
# regardless of what TCP is doing, the connection is dropped, and reconnection
# attempts start immediately — so the link is restored within RECONNECT_IVL_MAX
# of the network coming back rather than within a TCP backoff window.
ZMQ_HEARTBEAT_IVL_MS = 1000     # Ping the master every second
ZMQ_HEARTBEAT_TIMEOUT_MS = 3000 # No reply for 3s → the peer is gone

# Retry at a fixed interval with no exponential backoff.  Backoff is the wrong
# shape here: an outage lasts as long as it takes the other Nano to rebuild the
# network, and by the end of that window a backed-off socket would be waiting
# seconds between attempts and add that delay on top of the real recovery.  A
# failed connect on a down link returns immediately, so retrying five times a
# second through a 20-second outage costs nothing worth saving.
ZMQ_RECONNECT_IVL_MS = 200
ZMQ_RECONNECT_IVL_MAX_MS = 0    # 0 = constant interval, no backoff

# ── Progress watchdog ──────────────────────────────────────────────────────
# The feed has stopped in the field with the process still running, and
# restarting it by hand brought it straight back.  That is the signature of
# state inside this process that cannot recover on its own, and there are three
# candidates that all look identical from outside:
#
#   · Picamera2 wedges.  capture_array() has no timeout and can block forever
#     when the CSI pipeline stalls; the capture loop never comes back round, so
#     even the interval summary stops printing.
#   · The ZMQ connection goes stale.  Heartbeats catch a peer that stops
#     answering, but not every way a link can rot — and a PUSH socket with
#     IMMEDIATE set will sit there reporting no peers indefinitely.
#   · A blocking send against a zombie access point: associated, routed, and
#     nothing at the other end.
#
# Diagnosing which one it was after the fact needs logs from the moment it
# happened, and on demo day there will not be any.  So instead of identifying
# the fault, this detects the *symptom* they share — the process stops making
# progress — and applies the escalating version of what a human restart does.
#
# Two separate timers, because the two faults need different responses:
#
#   SEND_STALL   nothing has been delivered for this long, but the loop is
#                still turning.  The camera is fine and the socket is not, so
#                the socket is rebuilt in place.  Cheap, and invisible to the
#                master beyond a reconnect.
#   LOOP_STALL   the capture loop itself has not come round.  Nothing in-process
#                can fix that, because the thread that would do the fixing is
#                the one that is stuck.  The process exits and systemd restarts
#                it clean, which is exactly the manual recovery, minus the human.
#
# The loop turns every 1/TARGET_FPS — 50 ms at 20 fps — so 20 s is four hundred
# missed iterations.  Nothing normal comes close, including strict-mode sends
# that block for the full SEND_TIMEOUT_MS: those still return and still tick.
WATCHDOG_ENABLED = os.environ.get("PI_WATCHDOG", "1") != "0"
WATCHDOG_POLL_SEC = 2.0
SEND_STALL_SEC = float(os.environ.get("PI_SEND_STALL_SEC", "10.0"))
LOOP_STALL_SEC = float(os.environ.get("PI_LOOP_STALL_SEC", "20.0"))

# Exit code used when the watchdog gives up, chosen to be distinguishable in
# `systemctl status` from a crash (1) or a signal.
WATCHDOG_EXIT_CODE = 3

# Two sensors streaming under the same id is the single most destructive thing
# that can happen to a run: the master keys FrameStore on "sensor:frame_id", so
# the two sequences interleave and silently overwrite each other.  The run
# completes, looks plausible, and every number in it is wrong.
#
# Once this is a systemd service that is no longer hypothetical — starting it
# by hand out of habit while the service is already running does exactly that.
# An advisory lock on a fixed path makes the second one refuse to start.
LOCK_PATH = os.environ.get("PI_LOCK_PATH", "/tmp/pi_sensor.lock")

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False


class SensorNode:
    """Camera capture and frame streaming to the MEC master.

    Owns the ZMQ socket, the camera handle, the frame-rate governor, and clean
    shutdown on SIGINT/SIGTERM.
    """

    def __init__(self):
        self._running = True
        self._frame_id = 0
        self._drop_streak = 0
        self._total_drops = 0
        self._sent_since_report = 0
        self._dropped_since_report = 0
        self._frame_time = 1.0 / TARGET_FPS
        self._use_picamera2 = False

        # Watchdog state.  Both stamps are monotonic, so a chrony step — and
        # this rig steps its clocks deliberately — cannot make the watchdog
        # think the loop has hung, or hide a hang that really happened.
        self._loop_tick = time.monotonic()
        self._last_send_ok = time.monotonic()
        self._need_socket_rebuild = False
        self._socket_rebuilds = 0

        self.ctx = zmq.Context()
        self.sender = None
        self._build_socket()

        self.picam2 = None
        self.cap = None

        # The watchdog is NOT started here.  Opening the camera legitimately
        # takes seconds — a settle sleep, five warm-up frames, and a retry
        # ladder that backs off to half a minute between attempts — and none of
        # that ticks the capture loop.  A watchdog running during it would read
        # a perfectly healthy startup as a hang and kill the process, on repeat.
        # It starts in run(), where a stalled loop actually means something.

        if not self._open_camera() and not self._reconnect_camera():
            log.error("No working camera — cannot start")
            self.sender.close()
            self.ctx.term()
            sys.exit(1)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # -------------------------------------------------------------------
    #  Transport
    # -------------------------------------------------------------------

    def _build_socket(self):
        """Create and connect the PUSH socket.

        Split out of __init__ so the watchdog can have it rebuilt without
        restarting the process.  Called only from the capture thread — ZMQ
        sockets are not thread safe, which is why the watchdog sets a flag
        rather than touching this itself.
        """
        if self.sender is not None:
            # LINGER is 0, so this discards anything still queued rather than
            # blocking on a peer that is by definition not responding.  Those
            # frames are seconds old and would be dropped by the master's
            # deadline check the moment they arrived.
            try:
                self.sender.close()
            except Exception:
                pass
            self.sender = None

        self.sender = self.ctx.socket(zmq.PUSH)
        # Strict mode: a deeper queue and a bounded blocking send, so a brief
        # stall costs latency instead of frames. Live mode keeps the shallow
        # buffer, where staying current matters more than keeping every frame.
        self.sender.setsockopt(zmq.SNDHWM, 64 if STRICT_SEND else ZMQ_SEND_HWM)
        if STRICT_SEND:
            self.sender.setsockopt(zmq.SNDTIMEO, SEND_TIMEOUT_MS)
        self.sender.setsockopt(zmq.LINGER, 0)

        # Fail fast rather than queue for a master that is not there. Without
        # this, frames captured during an outage sit in the buffer and get
        # delivered on reconnect — seconds stale, instantly dropped by the
        # deadline check, having displaced fresh frames on the way.
        self.sender.setsockopt(zmq.IMMEDIATE, 1)

        # See the notes on these constants above: this is what turns a failover
        # from a possible half-minute wait into a couple of seconds.
        self.sender.setsockopt(zmq.HEARTBEAT_IVL, ZMQ_HEARTBEAT_IVL_MS)
        self.sender.setsockopt(zmq.HEARTBEAT_TIMEOUT, ZMQ_HEARTBEAT_TIMEOUT_MS)
        self.sender.setsockopt(zmq.RECONNECT_IVL, ZMQ_RECONNECT_IVL_MS)
        self.sender.setsockopt(zmq.RECONNECT_IVL_MAX, ZMQ_RECONNECT_IVL_MAX_MS)

        # Belt and braces at the TCP layer, for the case where the peer vanishes
        # without ever closing the connection.
        self.sender.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.sender.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 5)
        self.sender.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 2)

        self.sender.connect(f"tcp://{MASTER_IP}:{PI_INGEST_PORT}")

    # -------------------------------------------------------------------
    #  Watchdog
    # -------------------------------------------------------------------

    def _watchdog_loop(self):
        """Watch the capture loop for progress and escalate when it stops.

        Deliberately does almost nothing itself.  It touches no socket and no
        camera handle — both are owned by the capture thread and neither is
        thread safe — so its whole vocabulary is one flag and one hard exit.
        """
        while self._running:
            time.sleep(WATCHDOG_POLL_SEC)
            now = time.monotonic()

            loop_age = now - self._loop_tick
            if loop_age > LOOP_STALL_SEC:
                # The capture thread is blocked somewhere with no timeout —
                # almost certainly inside the camera driver.  It cannot act on
                # a flag, and there is no safe way to interrupt it from here,
                # so the only honest move is to die and let systemd restart.
                #
                # os._exit rather than sys.exit: sys.exit raises in *this*
                # thread, which the interpreter would then try to unwind while
                # the main thread is stuck in a C call, and the process would
                # hang in exactly the state being escaped.
                log.error("capture loop has not advanced for %.0fs — the camera "
                          "or the send has wedged.  Exiting so systemd restarts "
                          "a clean process.", loop_age)
                sys.stderr.flush()
                os._exit(WATCHDOG_EXIT_CODE)

            send_age = now - self._last_send_ok
            # Only a loop that is demonstrably turning can have a transport
            # fault worth rebuilding for.  Without this gate a wedged loop
            # trips both timers and logs a transport diagnosis for a camera
            # fault, sending whoever reads the journal after the demo in
            # precisely the wrong direction.  A strict-mode send blocks for at
            # most SEND_TIMEOUT_MS, so a live loop always ticks well inside
            # this window.
            loop_alive = loop_age < SEND_STALL_SEC
            if loop_alive and send_age > SEND_STALL_SEC and not self._need_socket_rebuild:
                # The loop is turning but nothing is getting out.  That is a
                # transport fault, and rebuilding the socket is the in-process
                # equivalent of the restart that has been fixing this by hand.
                log.warning("no frame delivered for %.0fs while the loop is still "
                            "running — asking for a socket rebuild", send_age)
                self._need_socket_rebuild = True

    # -------------------------------------------------------------------
    #  Camera
    # -------------------------------------------------------------------

    def _signal_handler(self, signum, _frame):
        log.info("Signal %d received — shutting down", signum)
        self._running = False

    def _release_camera(self):
        """Release whichever camera handle is currently held."""
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self._use_picamera2 = False
        time.sleep(0.5)              # Let the kernel finish releasing the device

    def _open_camera(self):
        """Open the camera, trying Picamera2 first and V4L2 second."""
        self._release_camera()

        if HAS_PICAMERA2:
            try:
                self.picam2 = Picamera2()
                # FrameDurationLimits pins the sensor's frame interval, in
                # microseconds, min and max equal. Without it Picamera2 picks a
                # duration from the exposure it happens to settle on, and a dim
                # room silently caps the sensor well below the requested rate.
                us = int(1_000_000 / TARGET_FPS)
                self.picam2.configure(self.picam2.create_preview_configuration(
                    main={"size": (FRAME_WIDTH, FRAME_HEIGHT)},
                    controls={"FrameDurationLimits": (us, us)}))
                self.picam2.start()

                time.sleep(1.0)      # Let auto-exposure settle before testing
                test = self.picam2.capture_array()
                if test is not None and test.size > 0:
                    self._use_picamera2 = True
                    log.info("Camera open — Picamera2 %dx%d", FRAME_WIDTH, FRAME_HEIGHT)
                    return True

                log.warning("Picamera2 opened but returned an empty frame")
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
            except Exception as e:
                log.warning("Picamera2 unavailable: %s", e)
                if self.picam2 is not None:
                    try:
                        self.picam2.stop()
                        self.picam2.close()
                    except Exception:
                        pass
                    self.picam2 = None

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        # MJPG before the size, and both before FPS. Most USB webcams only offer
        # 30 fps at 640x480 in MJPG; in the default YUYV mode the same camera
        # caps at 10-15 fps and quietly ignores a higher request. This one line
        # is usually the whole difference between asking for 30 and getting it.
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        # One frame of driver buffer. The default queues several, so read()
        # returns a frame captured some tens of milliseconds ago and the
        # cap_start stamp describes the wrong instant — latency that looks like
        # network delay but was added before the frame ever left the Pi.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if self.cap.isOpened():
            granted = self.cap.get(cv2.CAP_PROP_FPS)
            if granted and abs(granted - TARGET_FPS) > 0.5:
                log.warning("Camera granted %.1f fps, not the %.1f requested — "
                            "this is the ceiling, not a setting to argue with",
                            granted, TARGET_FPS)
            for _ in range(5):       # Discard warm-up frames
                time.sleep(0.2)
                ok, _ = self.cap.read()
                if ok:
                    self._use_picamera2 = False
                    log.info("Camera open — V4L2 index %d %dx%d",
                             CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
                    return True
            self.cap.release()
            self.cap = None

        log.error("No camera backend worked")
        return False

    def _capture_frame(self):
        """Grab one BGR frame, or None if the capture failed."""
        if self._use_picamera2 and self.picam2 is not None:
            try:
                rgb = self.picam2.capture_array()
                if rgb is not None and rgb.size > 0:
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                log.warning("Capture failed: %s", e)
            return None

        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None

        return None

    def _reconnect_camera(self):
        """Retry opening the camera with exponential backoff."""
        for attempt in range(1, CAMERA_MAX_RETRIES + 1):
            delay = min(CAMERA_RETRY_BASE_SEC * (2 ** (attempt - 1)), CAMERA_RETRY_CAP_SEC)
            log.warning("Camera reconnect %d/%d in %.0fs",
                        attempt, CAMERA_MAX_RETRIES, delay)
            # Sleep in slices, ticking the watchdog through each one.  The
            # backoff reaches 30 s, which is longer than LOOP_STALL_SEC, so a
            # single sleep would look exactly like a hang and the watchdog
            # would kill a process that is recovering correctly and saying so
            # in the log.  A handled fault is not the watchdog's business.
            waited = 0.0
            while waited < delay and self._running:
                time.sleep(min(1.0, delay - waited))
                waited += 1.0
                self._loop_tick = time.monotonic()
                self._last_send_ok = time.monotonic()
            if self._open_camera():
                log.info("Camera reconnected on attempt %d", attempt)
                return True

        log.error("Camera did not come back after %d attempts", CAMERA_MAX_RETRIES)
        return False

    # -------------------------------------------------------------------
    #  Main loop
    # -------------------------------------------------------------------

    def run(self):
        """Capture, compress, send — at the configured frame rate.

        Each message is multipart: [sensor_id, frame_id, capture_ts, jpeg_bytes].
        capture_ts is stamped here, at the moment of capture, because it is the
        start of every latency measurement in the system — the master and the
        Phase 0 baseline both measure from this timestamp, which is the only
        thing that makes their numbers comparable.
        """
        backend = "Picamera2" if self._use_picamera2 else "V4L2"
        log.info("Streaming to %s:%d — id=%s %d FPS, JPEG q%d, %s",
                 MASTER_IP, PI_INGEST_PORT, SENSOR_ID, TARGET_FPS, JPEG_QUALITY, backend)

        reporter = PeriodicReporter(STATS_INTERVAL_SEC)
        window_start = time.time()

        # Started here rather than in the constructor: from this point on, a
        # loop that stops turning is a fault, which is the only condition the
        # watchdog can distinguish.
        if WATCHDOG_ENABLED:
            self._loop_tick = time.monotonic()
            self._last_send_ok = time.monotonic()
            threading.Thread(target=self._watchdog_loop, name="watchdog",
                             daemon=True).start()
            log.info("watchdog armed — socket rebuild at %.0fs without a "
                     "delivery, restart at %.0fs without a loop tick",
                     SEND_STALL_SEC, LOOP_STALL_SEC)

        while self._running:
            loop_start = time.time()

            # Proof of life for the watchdog.  Updated before anything that can
            # block, so whatever the iteration goes on to get stuck in, the
            # stall is attributed to this iteration and not to the next one.
            self._loop_tick = time.monotonic()

            if self._need_socket_rebuild:
                self._socket_rebuilds += 1
                log.warning("rebuilding the ZMQ socket (#%d) — reconnecting to "
                            "%s:%d", self._socket_rebuilds, MASTER_IP, PI_INGEST_PORT)
                self._build_socket()
                self._need_socket_rebuild = False
                # Restart the clock rather than leaving it stale, so a rebuild
                # that does not help is retried on the same interval instead of
                # firing again on the very next poll.
                self._last_send_ok = time.monotonic()

            if reporter.due():
                span = time.time() - window_start
                fps = self._sent_since_report / span if span > 0 else 0
                # Achieved against target, every interval. A run that quietly
                # delivered 19 fps while the write-up claims 30 is the failure
                # this line exists to prevent — the shortfall has to be visible
                # while the run is happening, not inferred from the data later.
                shortfall = ("" if fps >= TARGET_FPS * 0.95
                             else "  ** %.0f%% of target" % (100.0 * fps / TARGET_FPS))
                log.info("%d sent | %.1f/%.0f fps | %d dropped this interval | "
                         "%d total%s",
                         self._frame_id, fps, TARGET_FPS,
                         self._dropped_since_report, self._total_drops, shortfall)
                self._sent_since_report = 0
                self._dropped_since_report = 0
                window_start = time.time()

            t_cap_start = time.time()
            frame = self._capture_frame()
            if frame is None:
                log.warning("Frame grab failed — reconnecting the camera")
                if not self._reconnect_camera():
                    # Exit rather than return quietly.  Under systemd this is
                    # the difference between a board that comes back on its own
                    # and one that is dark until someone notices — and the
                    # retry ladder above has already spent minutes on it, which
                    # is long enough for a genuinely transient fault to clear.
                    log.error("Camera permanently lost — exiting for a restart")
                    self.close()
                    sys.exit(WATCHDOG_EXIT_CODE)
                continue
            t_cap_done = time.time()

            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                log.warning("JPEG encode failed on frame %d — skipped", self._frame_id)
                continue
            t_enc_done = time.time()

            try:
                # capture_ts keeps its exact old meaning — the instant the frame
                # is handed to ZMQ — so every existing latency figure and the
                # Phase 0 baseline stay comparable. The trace stamps ride in a
                # fifth part the master may ignore.
                t_pi_send = time.time()
                stamps = json.dumps({
                    "cap_start": t_cap_start,   # shutter
                    "cap_done": t_cap_done,     # frame in hand, before JPEG
                    "enc_done": t_enc_done,     # conversion finished
                    "pi_send": t_pi_send,       # handed to the socket
                    # Payload size, measured where it is created. Every network
                    # segment downstream is only interpretable against it — a
                    # 40 ms hop means nothing until you know whether it moved
                    # 30 kB or 300 kB.
                    "bytes": int(buffer.size),
                    "wh": f"{frame.shape[1]}x{frame.shape[0]}",
                }).encode()

                # Strict mode blocks (bounded by SNDTIMEO); otherwise the old
                # NOBLOCK behaviour, which discards the frame the moment the
                # master is behind.
                self.sender.send_multipart([
                    SENSOR_ID.encode(),
                    str(self._frame_id).encode(),
                    str(t_pi_send).encode(),
                    buffer.tobytes(),
                    stamps,
                ], flags=0 if STRICT_SEND else zmq.NOBLOCK)

                # Only counts up on frames that were actually handed to ZMQ, so
                # the sequence has no gaps for frames dropped here.  The receiver
                # relies on that when it works out delivery ratio.
                self._frame_id += 1
                self._sent_since_report += 1
                self._drop_streak = 0
                self._last_send_ok = time.monotonic()

            except zmq.error.Again:
                # Send buffer full: the master is behind, or — during a failover —
                # briefly not there at all.  Dropping the newest frame is correct
                # either way; queueing it would only make everything behind it
                # older still.
                self._drop_streak += 1
                self._total_drops += 1
                self._dropped_since_report += 1
                if self._drop_streak == 30:
                    log.warning("30 consecutive drops — the master is unreachable "
                                "or saturated")

            sleep_time = self._frame_time - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.close()

    def close(self):
        """Release the camera and the ZMQ context.

        The totals printed here are needed to compute true end-to-end delivery
        ratio: the receiver can only see frames that were sent, and frames
        dropped in this process leave no gap in the sequence for it to notice.
        """
        log.info("Session totals — %d frames sent, %d dropped",
                 self._frame_id, self._total_drops)
        self._release_camera()
        self.sender.close()
        self.ctx.term()


def _claim_single_instance():
    """Refuse to start if another sensor process already holds the lock.

    The lock is held for the lifetime of the process by an open file
    descriptor, so it is released by the kernel however this exits — including
    a kill -9 or the watchdog's os._exit.  Nothing has to clean it up, and a
    stale lock file left behind by a crash does not block the next start.

    Returned rather than discarded so the descriptor stays referenced; letting
    it be garbage collected would close it and drop the lock.
    """
    fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.error("another pi_sensor is already running (lock %s held).", LOCK_PATH)
        log.error("Two sensors under one id interleave their frame numbers in "
                  "the master's store and overwrite each other — the run would "
                  "look fine and every number in it would be wrong.")
        log.error("Check with:  systemctl status pi-sensor")
        sys.exit(2)
    fd.write("%d\n" % os.getpid())
    fd.flush()
    return fd


if __name__ == "__main__":
    _lock = _claim_single_instance()
    SensorNode().run()
