"""Per-frame timestamp tracing across Pi → master → (worker) → GCS.

Answers two questions with real numbers rather than estimates:

  Time 01  capture → convert → network → decide → queue → infer on the master
           → network → GCS
  Time 02  the same frame path, but the scheduler diverted the frame to the
           other Nano, so it also carries master→worker, the worker's own
           decode+inference, and worker→master

Every stamp is `time.time()` — wall clock, not `perf_counter` — because the
interesting intervals span four machines and a monotonic counter is meaningless
across processes. That only works because chrony holds the rig inside ~15 µs;
see BUILD_GUIDE's time-sync section. If chrony is not synchronised the
cross-device segments here are worthless, so `report()` prints a warning when a
segment comes out negative, which is the signature of clock skew.

Design note — why a side table instead of extra function arguments: the frame
already travels through half a dozen queues as a fixed-arity tuple
`(key, img, capture_ts, dec_ms, q_len)`. Threading a stamps dict through all of
them means touching every put and get, in a file where those tuples are unpacked
in five places. Keying stamps by frame id in one bounded dict leaves every
existing signature alone — a stage just calls `trace.stamp(key, "infer_start")`.

Off unless MEC_TRACE_N is set. The Pi always attaches its own four stamps
because they cost ~120 bytes against a ~100 kB JPEG, and because a trace that
needs enabling in three places at once is a trace nobody successfully collects.
"""

import collections
import json
import os
import threading
import time

def _parse_n(raw):
    """MEC_TRACE_N accepts a count, or 'all' for every frame."""
    raw = (raw or "").strip().lower()
    if raw in ("all", "inf", "-1"):
        return 10 ** 9          # effectively unbounded; the ring buffer still caps memory
    try:
        return int(raw)
    except ValueError:
        return 0


# How many frames to keep per path. Two gives you one example of each path.
# 'all' traces every frame after the skip window, which is what you want for
# averages — a single frame of each path is an anecdote, not a measurement.
TRACE_N = _parse_n(os.environ.get("MEC_TRACE_N", "0"))

# Full tables are printed for at most this many frames per path, however many
# are traced. Rendering 500 tables into the GCS log would cost more than the
# tracing does, and every frame is written to the JSONL regardless.
PRINT_N = int(os.environ.get("MEC_TRACE_PRINT", "2"))
_printed = collections.Counter()
_print_lock = threading.Lock()


def should_print(path):
    """Rate-limit the full-table output. Safe to call from the GCS thread."""
    with _print_lock:
        if _printed[path] >= PRINT_N:
            return False
        _printed[path] += 1
        return True

# Frames to let past before tracing anything. Without this you measure the
# cold start, not the pipeline: on 2026-09-04 the first local frame spent
# 364.7 ms in inference against 92.2 ms once TensorRT was warm, and the first
# offloaded frame 549 ms against 118.7 ms. Worse, the frame *behind* each of
# those inherits the wait — frame 2's "local queue wait" was 317 ms, and it
# started inference 0.2 ms after frame 1 finished. Nothing about those numbers
# describes steady state.
#
# 100 frames at 10 FPS is ten seconds, which clears the engine load, the first
# CUDA allocations and the GCS's Tk startup backlog.
TRACE_SKIP = int(os.environ.get("MEC_TRACE_SKIP", "100"))

# Bound on the side table. Frames that are dropped or never reach the GCS would
# otherwise accumulate; the oldest is evicted rather than grown.
_CAPACITY = 256


# ── the segments, in pipeline order ────────────────────────────────────────
#
# (label, from_stamp, to_stamp, path)  —  path: "both", "local", "offload"
SEGMENTS = [
    ("capture",                "cap_start",    "cap_done",     "both"),
    ("conversion (JPEG)",      "cap_done",     "enc_done",     "both"),
    ("pi handoff to ZMQ",      "enc_done",     "pi_send",      "both"),
    ("network  Pi → master",   "pi_send",      "master_recv",  "both"),
    # Decode used to sit here, between arrival and the scheduler, and was paid
    # by every frame including the ones that were then offloaded or dropped. It
    # now happens after the decision and only on the local path, which is why
    # the offload timeline below has no decode-on-master segment at all.
    ("wait for scheduler",     "master_recv",  "sched_start",  "both"),
    ("decision (Lyapunov)",    "sched_start",  "sched_done",   "both"),

    ("decode queue wait",      "sched_done",   "decode_start", "local"),
    ("decode on master",       "decode_start", "decode_done",  "local"),
    ("GPU queue wait",         "decode_done",  "infer_start",  "local"),
    ("inference (master GPU)", "infer_start",  "infer_done",   "local"),
    ("forward to GCS",         "infer_done",   "gcs_send",     "local"),

    ("dispatch queue",         "sched_done",   "dispatch",     "offload"),
    ("network  master → worker", "dispatch",   "worker_recv",  "offload"),
    ("worker decode + inference", "worker_recv", "worker_done", "offload"),
    ("network  worker → master", "worker_done", "result_recv", "offload"),
    ("forward to GCS",         "result_recv",  "gcs_send",     "offload"),

    ("network  master → GCS",  "gcs_send",     "gcs_recv",     "both"),
]

# Counts, not instants. Recorded so "the queue wait was 317 ms" can be read
# alongside how many frames that actually was — on 2026-09-04 a 317 ms wait
# turned out to be a single frame that was still warming up, which is a very
# different finding from a backlog.
#
# (field, label, path)
QUEUE_FIELDS = [
    ("q_sched",    "scheduler queue at decision",   "both"),
    ("q_local",    "frames ahead in local queue",   "local"),
    # A 0/1 flag. Averaged over many frames its mean is the fraction that
    # arrived to a busy GPU, which is why the label says so.
    ("local_busy", "GPU busy on arrival (0/1)",     "local"),
    ("q_worker",   "frames in flight to worker",    "offload"),
]

# Network segments that actually move the JPEG, and so have a meaningful
# throughput. worker → master is excluded on purpose: it carries detections as
# JSON, a few hundred bytes, so dividing the payload size by it would invent a
# number. Same reason the on-board segments are absent — nothing crosses a wire.
PAYLOAD_HOPS = [
    ("network  Pi → master",     "pi_send",  "master_recv"),
    ("network  master → worker", "dispatch", "worker_recv"),
    ("network  master → GCS",    "gcs_send", "gcs_recv"),
]


def _throughput_mbs(nbytes, seconds):
    """MB/s, or None when the interval is too small to divide by."""
    if not nbytes or seconds is None or seconds <= 1e-6:
        return None
    return (nbytes / 1e6) / seconds


class FrameTrace:
    """Bounded side table of timestamps, keyed by frame id.

    Thread-safe: the master stamps from the ingest, scheduler, sender, results
    and inference threads concurrently.
    """

    # The paths claim() distinguishes. Used to tell when every quota is full.
    PATHS = ("local", "offload")

    def __init__(self, limit=None, skip=None, capacity=_CAPACITY):
        self.limit = TRACE_N if limit is None else limit
        self.skip = TRACE_SKIP if skip is None else skip
        self._capacity = capacity
        self._lock = threading.Lock()
        self._stamps = collections.OrderedDict()
        self._claimed = collections.Counter()   # path → how many kept so far
        self._seen = 0                          # frames ingested this run

    @property
    def enabled(self):
        return self.limit > 0

    def _quotas_full(self):
        return all(self._claimed[p] >= self.limit for p in self.PATHS)

    def begin(self, key, stamps=None):
        """Start tracing a frame, seeding it with stamps made elsewhere.

        Silently declines during the warm-up window and once every quota is
        full, so the cost of leaving tracing enabled for a whole run is one
        counter increment per frame.
        """
        if not self.enabled:
            return
        with self._lock:
            self._seen += 1
            if self._seen <= self.skip or self._quotas_full():
                return
            while len(self._stamps) >= self._capacity:
                self._stamps.popitem(last=False)
            self._stamps[key] = dict(stamps or {})

    def tracking(self, key):
        """Is this frame actually being traced?

        The master asks before telling a worker to stamp, so a skipped or
        past-quota frame costs the worker nothing at all.
        """
        if not self.enabled:
            return False
        with self._lock:
            return key in self._stamps

    def stamp(self, key, name, ts=None):
        if not self.enabled:
            return
        with self._lock:
            entry = self._stamps.get(key)
            if entry is not None and name not in entry:
                # First write wins. A frame that falls back to local after a
                # worker refuses it would otherwise overwrite its own
                # sched_done and report a negative queue wait.
                entry[name] = ts if ts is not None else time.time()

    def note(self, key, name, value):
        """Record a non-timestamp fact about a frame — a queue depth, a flag.

        Same storage as stamp(), separate method so a reader can tell the two
        apart: everything stamp() writes is an instant on some machine's clock,
        everything note() writes is a count.
        """
        if not self.enabled:
            return
        with self._lock:
            entry = self._stamps.get(key)
            if entry is not None and name not in entry:
                entry[name] = value

    def merge(self, key, stamps):
        """Fold in stamps that were made in another process (worker, Pi)."""
        if not self.enabled or not stamps:
            return
        with self._lock:
            entry = self._stamps.get(key)
            if entry is not None:
                for k, v in stamps.items():
                    entry.setdefault(k, v)

    def claim(self, key, path):
        """Should this frame stay traced? Enforces the per-path quota.

        Called once the path is known — that is, at the point the frame is about
        to leave for the GCS. Frames beyond the quota are dropped from the table
        so a long run does not keep paying for stamps nobody will read.
        """
        if not self.enabled:
            return False
        with self._lock:
            if key not in self._stamps:
                return False
            if self._claimed[path] >= self.limit:
                self._stamps.pop(key, None)
                return False
            self._claimed[path] += 1
            self._stamps[key]["path"] = path
            return True

    def export(self, key):
        with self._lock:
            entry = self._stamps.get(key)
            return dict(entry) if entry is not None else None

    def drop(self, key):
        with self._lock:
            self._stamps.pop(key, None)


# ── reporting ──────────────────────────────────────────────────────────────

def report(stamps, frame_id=""):
    """Render one traced frame as an absolute-timestamp table plus segments."""
    path = stamps.get("path", "local")
    label = "Time 02 — offloaded to worker Nano" if path == "offload" \
            else "Time 01 — processed on the master"

    lines = []
    lines.append("")
    lines.append("=" * 74)
    lines.append(f"{label}    frame {frame_id}")
    lines.append("=" * 74)

    # Absolute stamps first. The deltas below are derived from these, and when a
    # delta looks wrong this is the table that shows which machine disagreed.
    lines.append("")
    lines.append(f"  {'stamp':<26} {'absolute (UTC)':<28} {'epoch':>18}")
    lines.append("  " + "-" * 72)
    ordered = [n for _, n, _, _ in SEGMENTS] + [n for _, _, n, _ in SEGMENTS]
    seen = []
    for name in ordered:
        if name in stamps and name not in seen:
            seen.append(name)
    for name in sorted(seen, key=lambda n: stamps[n]):
        t = stamps[name]
        human = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
        human += f".{int((t % 1) * 1e6):06d}"
        lines.append(f"  {name:<26} {human:<28} {t:>18.6f}")

    lines.append("")
    lines.append(f"  {'segment':<28} {'ms':>10}   {'% of total':>10}")
    lines.append("  " + "-" * 72)

    total = None
    if "cap_start" in stamps and "gcs_recv" in stamps:
        total = stamps["gcs_recv"] - stamps["cap_start"]

    skew_warning = False
    for seg_label, a, b, seg_path in SEGMENTS:
        if seg_path != "both" and seg_path != path:
            continue
        if a not in stamps or b not in stamps:
            continue
        dt = stamps[b] - stamps[a]
        if dt < 0:
            skew_warning = True
        pct = f"{dt / total * 100:9.1f}%" if total and total > 0 else " " * 10
        lines.append(f"  {seg_label:<28} {dt * 1000:>10.3f}   {pct}")

    if total is not None:
        lines.append("  " + "-" * 72)
        lines.append(f"  {'TOTAL  capture → GCS':<28} {total * 1000:>10.3f}")
        # The pipeline's own latency. When the sensor is the bottleneck the
        # capture stamp is mostly the loop blocking for the next frame — time
        # during which this frame does not yet exist, so charging it as latency
        # overstates the figure. Measured on 2026-09-04: 3.5 ms at 10 fps
        # against 52.6 ms once the loop outran a 16 fps camera, for an
        # unchanged pipeline.
        if "cap_done" in stamps:
            ready = stamps["gcs_recv"] - stamps["cap_done"]
            lines.append(f"  {'TOTAL  frame ready → GCS':<28} {ready * 1000:>10.3f}")

    # Payload, and what each wire segment achieved moving it.
    nbytes = stamps.get("bytes")
    if nbytes:
        lines.append("")
        lines.append(f"  {'payload':<28} {'':>10}")
        lines.append("  " + "-" * 72)
        wh = stamps.get("wh")
        lines.append(f"  {'JPEG size':<28} {nbytes / 1024:>10.1f} KB"
                     + (f"   ({wh})" if wh else ""))
        for hop_label, a, b in PAYLOAD_HOPS:
            if a not in stamps or b not in stamps:
                continue
            mbs = _throughput_mbs(nbytes, stamps[b] - stamps[a])
            if mbs is not None:
                lines.append(f"  {hop_label:<28} {mbs:>10.2f} MB/s")

    # What the waiting was actually made of. A long wait behind one frame is a
    # slow frame; the same wait behind six is a backlog.
    depths = [(lbl, stamps[f]) for f, lbl, p in QUEUE_FIELDS
              if f in stamps and p in ("both", path)]
    if depths:
        lines.append("")
        lines.append(f"  {'queue depth':<38} {'frames':>8}")
        lines.append("  " + "-" * 72)
        for lbl, val in depths:
            shown = ("yes" if val else "no") if lbl.startswith("GPU") else val
            lines.append(f"  {lbl:<38} {str(shown):>8}")
        ahead = stamps.get("q_local")
        busy = stamps.get("local_busy")
        if path == "local" and ahead is not None and busy is not None:
            # Decision → GPU, minus this frame's own decode. Decode sits inside
            # that span now, and it is work on this frame rather than time spent
            # behind other frames — leaving it in would inflate the per-frame
            # queueing cost by the decode time on every single frame.
            wait_ms = None
            if "infer_start" in stamps and "sched_done" in stamps:
                wait_ms = (stamps["infer_start"] - stamps["sched_done"]) * 1000
                if "decode_start" in stamps and "decode_done" in stamps:
                    wait_ms -= (stamps["decode_done"] - stamps["decode_start"]) * 1000
            n = ahead + (1 if busy else 0)
            if wait_ms is not None and n:
                lines.append(f"  → {wait_ms:.1f} ms spent behind {n} frame(s)"
                             f" = {wait_ms / n:.1f} ms each")
            elif wait_ms is not None:
                lines.append(f"  → {wait_ms:.1f} ms with an empty queue and an idle GPU")

    if skew_warning:
        lines.append("")
        lines.append("  ** a segment came out NEGATIVE — the two clocks it spans")
        lines.append("     disagree. Check: ./chrony_setup.sh check on both ends.")

    lines.append("")
    return "\n".join(lines)


def _stats(values):
    """n, mean, median, p95, min, max — no numpy, this runs on a Jetson."""
    vs = sorted(values)
    n = len(vs)
    mean = sum(vs) / n
    mid = vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2
    # Nearest-rank p95. With n < 20 this is just the max, which is honest —
    # it does not invent a percentile the sample cannot support.
    p95 = vs[min(n - 1, int(round(0.95 * n)) - 1 if n > 1 else 0)]
    return n, mean, mid, p95, vs[0], vs[-1]


def aggregate(rows, path):
    """Per-segment statistics across every traced frame of one path.

    Segments are recomputed from the stamps rather than read from the ms_
    columns, so a file written by an older build still aggregates.
    """
    same = [r for r in rows if r.get("path") == path]
    if not same:
        return None

    lines = []
    label = "Time 02 — offloaded to worker Nano" if path == "offload" \
            else "Time 01 — processed on the master"
    lines.append("")
    lines.append("=" * 86)
    lines.append(f"{label}    {len(same)} frames")
    lines.append("=" * 86)
    lines.append("")
    lines.append(f"  {'segment':<28} {'n':>4} {'mean':>9} {'median':>9} "
                 f"{'p95':>9} {'min':>9} {'max':>9}")
    lines.append("  " + "-" * 84)

    for seg_label, a, b, seg_path in SEGMENTS:
        if seg_path != "both" and seg_path != path:
            continue
        vals = [(r[b] - r[a]) * 1000 for r in same if a in r and b in r]
        if not vals:
            continue
        n, mean, mid, p95, lo, hi = _stats(vals)
        lines.append(f"  {seg_label:<28} {n:>4} {mean:>9.3f} {mid:>9.3f} "
                     f"{p95:>9.3f} {lo:>9.3f} {hi:>9.3f}")

    totals = [(r["gcs_recv"] - r["cap_start"]) * 1000
              for r in same if "gcs_recv" in r and "cap_start" in r]
    if totals:
        n, mean, mid, p95, lo, hi = _stats(totals)
        lines.append("  " + "-" * 84)
        lines.append(f"  {'TOTAL  capture → GCS':<28} {n:>4} {mean:>9.3f} "
                     f"{mid:>9.3f} {p95:>9.3f} {lo:>9.3f} {hi:>9.3f}")
    # See report(): once the loop outruns the camera, the capture stamp is
    # mostly waiting for a frame that does not exist yet. This is the row to
    # quote as pipeline latency.
    ready = [(r["gcs_recv"] - r["cap_done"]) * 1000
             for r in same if "gcs_recv" in r and "cap_done" in r]
    if ready:
        n, mean, mid, p95, lo, hi = _stats(ready)
        lines.append(f"  {'TOTAL  frame ready → GCS':<28} {n:>4} {mean:>9.3f} "
                     f"{mid:>9.3f} {p95:>9.3f} {lo:>9.3f} {hi:>9.3f}")

    sizes = [r["bytes"] for r in same if r.get("bytes")]
    if sizes:
        lines.append("")
        lines.append(f"  {'payload':<28} {'n':>4} {'mean':>9} {'median':>9} "
                     f"{'p95':>9} {'min':>9} {'max':>9}")
        lines.append("  " + "-" * 84)
        n, mean, mid, p95, lo, hi = _stats([b / 1024.0 for b in sizes])
        lines.append(f"  {'JPEG size (KB)':<28} {n:>4} {mean:>9.1f} {mid:>9.1f} "
                     f"{p95:>9.1f} {lo:>9.1f} {hi:>9.1f}")
        for hop_label, a, b in PAYLOAD_HOPS:
            rates = [m for m in (_throughput_mbs(r.get("bytes"), r[b] - r[a])
                                 for r in same if a in r and b in r)
                     if m is not None]
            if rates:
                n, mean, mid, p95, lo, hi = _stats(rates)
                # p95 of a rate is the fast tail, so min is the one that matters
                # for a deadline — it is the worst the link did.
                short = hop_label.replace("network  ", "") + " (MB/s)"
                lines.append(f"  {short:<28} {n:>4} {mean:>9.2f} "
                             f"{mid:>9.2f} {p95:>9.2f} {lo:>9.2f} {hi:>9.2f}")

    depths = [(f, lbl) for f, lbl, p in QUEUE_FIELDS if p in ("both", path)]
    shown = False
    for field, lbl in depths:
        vals = [r[field] for r in same if field in r]
        if not vals:
            continue
        if not shown:
            lines.append("")
            lines.append(f"  {'queue depth (frames)':<28} {'n':>4} {'mean':>9} "
                         f"{'median':>9} {'p95':>9} {'min':>9} {'max':>9}")
            lines.append("  " + "-" * 84)
            shown = True
        n, mean, mid, p95, lo, hi = _stats([float(v) for v in vals])
        lines.append(f"  {lbl:<28} {n:>4} {mean:>9.2f} {mid:>9.2f} "
                     f"{p95:>9.2f} {lo:>9.2f} {hi:>9.2f}")

    neg = sum(1 for r in same for _, a, b, p in SEGMENTS
              if p in ("both", path) and a in r and b in r and r[b] < r[a])
    if neg:
        lines.append("")
        lines.append(f"  ** {neg} negative segment(s) across these frames —")
        lines.append("     clock skew. Do not quote these numbers.")

    lines.append("")
    return "\n".join(lines)


def coverage(rows):
    """Account for every frame the Pi sent, not just the ones that arrived.

    frame_id is `sensor:N` where N is the Pi's own monotonic counter, bumped
    only on frames actually handed to ZMQ. So within the traced span, any id
    absent from the file is a frame that was sent and never reached the GCS —
    dropped at a queue, past its deadline, or lost to a link outage. Counting
    the rows that arrived tells you nothing about that; this does.
    """
    lines = [""]
    by_sensor = {}
    for r in rows:
        fid = r.get("frame_id", "")
        if ":" not in fid:
            continue
        s, n = fid.rsplit(":", 1)
        if n.isdigit():
            by_sensor.setdefault(s, set()).add(int(n))
    if not by_sensor:
        return "  (no parseable frame ids — cannot account for loss)"

    for sensor, ids in sorted(by_sensor.items()):
        lo, hi = min(ids), max(ids)
        span = hi - lo + 1
        missing = sorted(set(range(lo, hi + 1)) - ids)
        pct = len(ids) / span * 100

        # Collapse consecutive missing ids into runs — 40 singletons and one
        # 40-frame hole are different failures and should not read the same.
        runs, start, prev = [], None, None
        for m in missing:
            if start is None:
                start = prev = m
            elif m == prev + 1:
                prev = m
            else:
                runs.append((start, prev)); start = prev = m
        if start is not None:
            runs.append((start, prev))

        lines.append(f"  {sensor}: ids {lo}–{hi} ({span} sent) · "
                     f"{len(ids)} traced · {len(missing)} lost · {pct:.1f}% coverage")
        if runs:
            longest = max(runs, key=lambda r: r[1] - r[0])
            singles = sum(1 for a, b in runs if a == b)
            lines.append(f"     {len(runs)} gap(s); {singles} single frame(s); "
                         f"longest {longest[1]-longest[0]+1} frames at id {longest[0]}")
            shown = ", ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in runs[:12])
            lines.append(f"     {shown}{' …' if len(runs) > 12 else ''}")
            if longest[1] - longest[0] + 1 >= 20:
                lines.append("     ** a gap that long is a link outage, not queue "
                             "shedding — check the GCS event log for 'feed lost'")
    return "\n".join(lines)


def write_row(path_dir, stamps, frame_id=""):
    """Append one traced frame to runtime/frame_trace.jsonl, best effort."""
    try:
        os.makedirs(path_dir, exist_ok=True)
        row = dict(stamps)
        row["frame_id"] = frame_id
        for seg_label, a, b, seg_path in SEGMENTS:
            if seg_path != "both" and seg_path != stamps.get("path", "local"):
                continue
            if a in stamps and b in stamps:
                row[f"ms_{a}__{b}"] = round((stamps[b] - stamps[a]) * 1000, 3)
        if "cap_start" in stamps and "gcs_recv" in stamps:
            row["ms_total"] = round(
                (stamps["gcs_recv"] - stamps["cap_start"]) * 1000, 3)
        with open(os.path.join(path_dir, "frame_trace.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        pass            # tracing must never take the pipeline down


if __name__ == "__main__":
    # Re-render a trace file:  python3 frame_trace.py gcs_logs/frame_trace.jsonl
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "gcs_logs/frame_trace.jsonl"
    try:
        rows = [json.loads(line) for line in open(src) if line.strip()]
    except FileNotFoundError:
        print(f"no trace file at {src} — was the GCS running?")
        raise SystemExit(1)

    if not rows:
        print(f"{src} is empty")
        raise SystemExit(1)

    counts = collections.Counter(r.get("path", "?") for r in rows)
    print(f"{len(rows)} traced frames in {src}   "
          + "  ".join(f"{p}={n}" for p, n in sorted(counts.items())))
    print(coverage(rows))

    for path in ("local", "offload"):
        same = [r for r in rows if r.get("path") == path]
        if not same:
            print(f"\n** no '{path}' frames — "
                  + ("the scheduler never offloaded; raise TARGET_FPS on the Pi"
                     if path == "offload" else "nothing ran locally"))
            continue
        if len(same) >= 5:
            # Enough to average. One frame is an anecdote; the spread between
            # median and p95 is what says whether the mean means anything.
            print(aggregate(rows, path))
        else:
            print(f"\n** only {len(same)} '{path}' frame(s) — too few to average, "
                  "showing the fastest")
            best = min(same, key=lambda r: r.get("ms_total", float("inf")))
            print(report(best, best.get("frame_id", "")))

    loc = [r["ms_total"] for r in rows
           if r.get("path") == "local" and "ms_total" in r]
    off = [r["ms_total"] for r in rows
           if r.get("path") == "offload" and "ms_total" in r]
    if len(loc) >= 5 and len(off) >= 5:
        ml, mo = sum(loc) / len(loc), sum(off) / len(off)
        print("=" * 86)
        print(f"  Time 01 (local)    mean {ml:9.3f} ms   over {len(loc)} frames")
        print(f"  Time 02 (offload)  mean {mo:9.3f} ms   over {len(off)} frames")
        print(f"  offloading costs   {mo - ml:+9.3f} ms   ({(mo / ml - 1) * 100:+.1f}%)")
        print("=" * 86)
