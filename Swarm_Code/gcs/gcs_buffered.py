"""
gcs_buffered.py

Ground control station — dual-screen view, BUFFERED playback.

One screen per Raspberry Pi sensor, side by side. Each screen carries its own
video and its own measurements, so the two cameras can be compared directly
instead of sharing one set of swarm-wide numbers. The layout is deliberately
identical to gcs.py, down to the metric columns, so a recording of
one build can be held against a recording of the other without allowing for any
difference in presentation. A summary
strip along the bottom keeps the totals that really are swarm properties.

This is the buffered variant. Frames land in a per-sensor jitter buffer keyed on
capture sequence number, and the display loop shows them in capture order:

  * A frame that arrives early waits for its predecessors.
  * A frame the master reported as dropped is stepped over immediately.
  * A gap with no drop notice is stepped over after JITTER_BUFFER_TIMEOUT_SEC.

The cost is the buffer wait, which is measured and folded into display latency,
so the number on screen tells the truth about what buffering costs you. The
matching unbuffered build is gcs.py — same layout, same wire
format, so the two can be run back to back on the same swarm and compared.

Latency is composed from single-clock terms rather than subtracted across
machines: processing (measured on the master, already skew-corrected) plus
transport jitter above the observed floor plus the buffer wait and render
measured here. The transport floor itself cannot be separated from clock skew by
one-way observation, so it is excluded and reported as excluded.

Run:  python3 gcs_buffered.py
"""

import csv
import glob
import json
import logging
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from collections import deque

import cv2
import numpy as np
import zmq
from PIL import Image, ImageTk

import frame_trace

# Shared with gcs.py. Anything that has to hold the same value in
# both builds lives there, so the two cannot drift apart the way the deadline
# did — see the note at the top of gcs_common.py.
from gcs_common import (  # noqa: F401 — palette/consts used throughout
    FRAME_DEADLINE_MS, JITTER_BUFFER_TIMEOUT_SEC, LAST_MSG_TIMEOUT,
    GCS_PORT, HOP_STEP_THRESHOLD_SEC, CLOCK_DISAGREE_MS,
    CLOCK_DISAGREE_QUIET_SEC, EVENT_ROWS_MAX, FPS_SAMPLES,
    CHART_REPAINT_MS, SENSORS, WINDOW_SIZE,
    PAPER, PANEL, RULE, INK, MUTED, TEAL, VIOLET,
    OK, WARN, BAD, PANEL_RGB,
    SensorScreenMixin, GroundStationMixin,
    latency_colour, sane_proc_ms, check_reported_deadline, _pick_font)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gcs-buf")

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════


# Screens drawn at startup, before any frame arrives, so the split layout is on
# screen while you are still bringing the swarm up — and a camera that never
# appears is an obviously empty screen rather than something silently missing.
#
# Not a whitelist: a sensor that turns up unlisted gets a screen of its own.
# Match the SENSOR_ID values in pi_sensor.py on each Pi.

WINDOW_TITLE = "Ground control — dual sensor — buffered"

# JITTER_BUFFER_TIMEOUT_SEC is imported from gcs_common, derived from the frame
# budget rather than written down beside it. Any frame that will arrive at all
# arrives within that budget, since the scheduler drops it before dispatch
# otherwise; drop notices let us skip most waits outright, so the timeout is only
# a backstop for notices that themselves went missing. It has to sit *above* the
# budget — it was 1.0 s against a 1.2 s deadline, abandoning frames the master
# still considered live.

# A buffer deeper than this is not jitter any more: it means frames are arriving
# faster than this laptop can draw them, and every frame still held is adding its
# full queueing time to the latency on screen. Past the limit, playback jumps to
# the tail and keeps JITTER_BUFFER_KEEP frames, so the display stays near-live
# instead of falling further behind with every second.
JITTER_BUFFER_MAX = 15
JITTER_BUFFER_KEEP = 5



# ── Palette ────────────────────────────────────────────────────────────────
# Colour is split by job so no hue ever does two things at once.
#   Cool hues = identity — which node did the work
#   Warm hues = status   — how the system is doing right now




# Still stamped into the event log so a saved CSV names the build that wrote
# it, even though the window no longer carries a badge — the direct build does
# the same, and the two headers are now identical on screen.
MODE_LABEL = "BUFFERED"




# ═══════════════════════════════════════════════════════════════════════════
#  One screen per sensor
# ═══════════════════════════════════════════════════════════════════════════

class SensorScreen(SensorScreenMixin):
    """A single Pi's screen: the video, and its own numbers beneath it.

    Everything here is per-sensor on purpose. Sharing one latency readout
    between two cameras hides exactly the case this display exists to show —
    one feed healthy while the other is starved.
    """

    def __init__(self, parent, sensor, mono, sans):
        self.sensor = sensor
        self.mono = mono
        self.sans = sans

        self.shown = 0
        self.dropped = 0
        self.skipped = 0
        self.render_stamps = deque(maxlen=FPS_SAMPLES)

        # The Tk image this screen draws into. Held and reused across frames;
        # allocating a fresh PhotoImage every frame costs measurably more (see
        # the render path) and is only actually needed when the size changes.
        self.photo = None
        self.photo_size = None

        self.frame = tk.Frame(parent, bg=PANEL, highlightthickness=1,
                              highlightbackground=RULE)

        # ── Title row: who this is, and who processed its last frame ──
        head = tk.Frame(self.frame, bg=PANEL)
        head.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(head, text=sensor, font=(mono, 13, "bold"),
                 fg=INK, bg=PANEL).pack(side="left")
        self.node = tk.Label(head, text="—", font=(sans, 10), fg=MUTED, bg=PANEL)
        self.node.pack(side="right")
        tk.Label(head, text="processed on", font=(sans, 9),
                 fg=MUTED, bg=PANEL).pack(side="right", padx=(0, 6))

        # ── Big number: display latency for this camera ──
        lat_row = tk.Frame(self.frame, bg=PANEL)
        lat_row.pack(fill="x", padx=12, pady=(0, 6))
        self.latency = tk.Label(lat_row, text="—", font=(mono, 30), fg=MUTED, bg=PANEL)
        self.latency.pack(side="left")
        tk.Label(lat_row, text="display latency\nexcludes transport floor",
                 font=(sans, 8), fg=MUTED, bg=PANEL, justify="left").pack(
                     side="left", padx=(10, 0), pady=(8, 0))

        # ── Bottom furniture packed before the video ──
        # A Label's requested size follows the image it holds, so a video packed
        # first claims everything left and squeezes these out entirely.
        self.metrics = tk.Frame(self.frame, bg=PANEL)
        self.metrics.pack(side="bottom", fill="x", padx=12, pady=(6, 10))


        self.video = tk.Label(self.frame, bg=PANEL, fg=MUTED, font=(sans, 10),
                              text=f"waiting for {sensor}")
        self.video.pack(side="top", expand=True, fill="both", padx=1, pady=(0, 1))

        # Metrics laid out as a fixed grid: labels never move as values change,
        # so a glance at the same spot always reads the same quantity.
        self.values = {}
        # One row of five, in the same order and the same columns as the direct
        # build, so a buffered recording and a direct one line up frame for
        # frame when they are compared. "Skipped" sits where the direct build
        # puts "Gaps": both count sequence numbers that never reached the
        # screen, one because playback stepped over them, the other because
        # nothing ever arrived.
        names = ("Processing", "Feed rate", "Shown", "Dropped", "Skipped")
        for col, name in enumerate(names):
            self.values[name] = self._metric(self.metrics, name, col, 0)
        for col in range(3):
            self.metrics.grid_columnconfigure(col, weight=1, uniform="metric")


    # -- numbers -------------------------------------------------------



    def note_skip(self, count):
        self.skipped += count
        self.values["Skipped"].config(text=f"{self.skipped:,}", fg=WARN)

    def note_depth(self, depth):
        """Buffer depth is no longer shown.

        The direct build has no counterpart, and the screens are kept identical
        so the two recordings can be compared without allowing for a layout
        difference. Depth is not lost as a measurement: the wait it causes is
        already inside display latency, which both builds show in the same
        place. Kept as a no-op so the caller in the display loop needs no
        special case.
        """



    # -- trace ---------------------------------------------------------



# ═══════════════════════════════════════════════════════════════════════════
#  Ground station
# ═══════════════════════════════════════════════════════════════════════════

class DualGroundStation(GroundStationMixin):
    # Supplied to GroundStationMixin, which cannot know them.
    SCREEN_CLASS = SensorScreen
    MODE_LABEL = "BUFFERED"
    log = log

    """Two screens, one per Pi, fed from the MEC master over ZMQ PULL."""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(1100, 660)
        self.root.configure(bg=PAPER)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._shutting_down = False
        self._zmq_ctx = zmq.Context()

        # Monospace for measured values is the core typographic rule here: every
        # number that comes off the system is set in it, and nothing else is.
        self.mono = _pick_font(
            ["JetBrains Mono", "IBM Plex Mono", "SF Mono", "Menlo",
             "DejaVu Sans Mono", "Consolas", "Courier New"], "Courier")
        self.sans = _pick_font(
            ["Inter", "IBM Plex Sans", "Segoe UI", "SF Pro Text",
             "DejaVu Sans", "Helvetica"], "Helvetica")

        # ── Swarm-wide counters ──
        self.frames_shown = 0
        self.frames_dropped = 0
        self.count_local = 0
        self.count_remote = 0
        self.last_msg_time = time.time()
        self._link_down_since = None

        self.screens = {}       # sensor -> SensorScreen
        self._split_dirty = False

        # ── Transport jitter, without a shared clock ──
        # Differencing arrival against the master's send stamp gives hop − skew.
        # The minimum over a window is the quietest such sample, so subtracting
        # it leaves hop − min_hop: queueing and jitter above the floor, with the
        # skew cancelled. The floor itself cannot be separated from skew by
        # one-way observation, so it is excluded and stated as excluded.
        self._hop_lock = threading.Lock()
        self._hop_samples = deque(maxlen=400)
        self._clock_disagree_at = {}
        self._deadline_checked = False

        # ── Jitter buffer ──
        # Written by the ZMQ thread, read and mutated by the Tk display loop, so
        # every touch is under this lock. Without it the display loop's scan can
        # collide with an insert and raise "dictionary changed size during
        # iteration" — intermittently, under load.
        self._buffer_lock = threading.Lock()
        self._jitter_buffer = {}      # sensor -> {frame_id: payload}
        self._next_display_id = {}    # sensor -> next frame_id to show
        self._dropped_ids = {}        # sensor -> ids the master reported dropped

        # tkinter is not thread-safe, and root.after() from another thread only
        # happens to work while mainloop runs. One queue removes the whole class
        # of problem: the ZMQ thread posts, the Tk thread drains.
        self._inbox = queue.Queue(maxsize=500)

        self._init_event_log()
        self._build_ui()

        for sensor in SENSORS:
            self._ensure_screen(sensor, announce=False)

        threading.Thread(target=self._zmq_receiver, daemon=True, name="zmq-recv").start()
        self.root.after(1000, self._check_link_status)
        self.root.after(CHART_REPAINT_MS, self._repaint_charts)
        self.root.after(33, self._display_loop)

    # -------------------------------------------------------------------
    #  Clock-free latency composition
    # -------------------------------------------------------------------



    def _composed_latency(self, data, now):
        """processing + transport jitter + buffer wait, each on one clock."""
        proc_ms = data.get("proc_latency")
        if proc_ms is None:
            return None
        recv_ts = data["receive_time"]
        jitter_ms = self._hop_excess(data.get("sent_ts"), recv_ts) * 1000
        held_ms = max(0.0, now - recv_ts) * 1000
        return proc_ms + jitter_ms + held_ms


    # -------------------------------------------------------------------
    #  Event log — one CSV per run, auto-numbered like the master's output
    # -------------------------------------------------------------------

    def _init_event_log(self):
        # Next to this file, not in whatever directory you happened to launch
        # from, so the logs are always in one predictable place.
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcs_logs")
        os.makedirs(log_dir, exist_ok=True)
        serial = len(glob.glob(os.path.join(log_dir, "dual_buffered_*.csv"))) + 1
        self.event_log_path = os.path.join(log_dir, f"dual_buffered_{serial:03d}.csv")
        self._event_file = open(self.event_log_path, "w", newline="")
        self._event_csv = csv.writer(self._event_file)
        self._event_csv.writerow(
            ["Wall Clock", "Epoch", "Elapsed (s)", "Event", "Frame ID", "Detail"])
        self._event_file.flush()
        self._start_time = time.time()
        log.info("Event log: %s", self.event_log_path)


    # -------------------------------------------------------------------
    #  UI
    # -------------------------------------------------------------------


    def _build_feed_tab(self, parent):
        # ── Swarm summary strip ──
        # Only quantities that genuinely belong to the system rather than to one
        # camera live down here; everything per-camera is up in its own screen.
        #
        # Packed BEFORE the stage even though it sits below it. pack hands out
        # space in call order, so a stage packed first with expand=True claims
        # the whole tab and this strip never renders at all.
        summary = tk.Frame(parent, bg=PAPER)
        summary.pack(side="bottom", fill="x", pady=(14, 6))
        tk.Frame(parent, bg=RULE, height=1).pack(side="bottom", fill="x")

        # ── The two screens ──
        self.stage = tk.Frame(parent, bg=PAPER)
        self.stage.pack(side="top", expand=True, fill="both", pady=(16, 0))
        self.stage.grid_rowconfigure(0, weight=1)

        self.total_shown = self._summary_stat(summary, "Frames shown")
        self.total_dropped = self._summary_stat(summary, "Frames dropped")
        self.elapsed_val = self._summary_stat(summary, "Session")

        split_box = tk.Frame(summary, bg=PAPER)
        split_box.pack(side="right", fill="x", expand=True, padx=(30, 0))
        self.split_label = tk.Label(split_box, text="awaiting frames",
                                    font=(self.sans, 9), fg=MUTED, bg=PAPER)
        self.split_label.pack(anchor="e")
        # The bar reuses the two node hues, so it doubles as the legend for the
        # coloured segments in both traces above it.
        self.split_canvas = tk.Canvas(split_box, height=8, bg=PAPER,
                                      highlightthickness=0)
        self.split_canvas.pack(fill="x", pady=(4, 0))






    # -------------------------------------------------------------------
    #  Link health
    # -------------------------------------------------------------------

    def _check_link_status(self):
        """Watch for the feed stopping and restarting.

        "feed lost" and "feed resumed" are the measurement for a failover drill:
        recovery time is the gap between them, taken from the one machine that
        stayed up and kept a continuous clock through the whole event.
        """
        if self._shutting_down:
            return

        now = time.time()
        elapsed = now - self.last_msg_time
        if elapsed > LAST_MSG_TIMEOUT:
            self.status_label.config(text=f"No data for {elapsed:.0f}s")
            self.status_dot.config(fg=BAD)
            if self._link_down_since is None:
                self._link_down_since = self.last_msg_time
                self._record("feed lost", detail=f"nothing for {LAST_MSG_TIMEOUT:.0f}s",
                             tone="drop")
                log.warning("Feed lost — no frames for %.0fs", elapsed)
        else:
            self.status_label.config(text="Receiving")
            self.status_dot.config(fg=OK)

        for screen in self.screens.values():
            screen.note_idle(now)
        self.elapsed_val.config(text=time.strftime("%M:%S", time.gmtime(now - self._start_time)))

        self.root.after(1000, self._check_link_status)


    # -------------------------------------------------------------------
    #  Frame render
    # -------------------------------------------------------------------

    def _update_screen(self, sensor, source_node, img_bytes, display_ms, proc_ms,
                       detections):
        """Draw one frame into its sensor's screen and update the swarm totals."""
        if self._shutting_down:
            return
        self._note_traffic()

        screen = self._ensure_screen(sensor)
        is_local = str(source_node).upper() == "MASTER"
        node_colour = TEAL if is_local else VIOLET

        if display_ms is not None:
            colour = latency_colour(display_ms, OK, WARN, BAD)
            screen.latency.config(text=f"{display_ms:.0f} ms", fg=colour)
        else:
            # Blanked rather than left alone. Skipping the update would hold the
            # previous frame's figure on screen while frames of unknown age keep
            # arriving, which reads as a healthy latency that nothing measured.
            # The trace is left untouched: a frame whose age is unknown has no
            # place on a plot of latency over time.
            screen.latency.config(text="—", fg=INK)

        screen.node.config(text=str(source_node), fg=node_colour)
        screen.values["Processing"].config(
            text=f"{proc_ms:.0f} ms" if proc_ms is not None else "—")

        if is_local:
            self.count_local += 1
        else:
            self.count_remote += 1
        self._split_dirty = True

        if not img_bytes:
            return
        try:
            cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if cv_img is None:
                self._record("corrupt frame", frame_id=sensor,
                             detail="image failed to decode", tone="warn")
                return

            for det in detections:
                x1, y1, x2, y2 = det.get("box", [0, 0, 0, 0])
                label = f"{det.get('cls', 'object')} {det.get('conf', 0.0):.2f}"
                # BGR form of TEAL, so boxes match the local-node hue
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (122, 108, 12), 2)
                cv2.putText(cv_img, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 108, 12), 1)

            cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # Size to the space the video label has actually been allotted.
            # The usual objection — that a Label's requested size follows its
            # image, so measuring the label and then filling it feeds back — does
            # not bite here: the cell width comes from the window through a
            # uniform grid, and an image bigger than the cell is clipped rather
            # than granted, so the measurement settles on the first frame instead
            # of growing. Deriving it from the stage instead needs a constant for
            # the height of the header, the latency line, the trace and the
            # metrics grid, and that constant is wrong the moment any of them
            # changes — which is how frames ended up clipped by 70px.
            #
            # Before the first layout pass winfo_ returns 1, so fall back to a
            # stage-derived estimate until Tk has placed things.
            vid_w, vid_h = screen.video.winfo_width(), screen.video.winfo_height()
            if vid_w > 80 and vid_h > 80:
                target_w, target_h = vid_w, vid_h
            else:
                count = max(len(self.screens), 1)
                target_w = max(max(self.stage.winfo_width(), 900) // count - 34, 220)
                target_h = max(max(self.stage.winfo_height(), 420) - 282, 160)

            orig_h, orig_w = cv_rgb.shape[:2]
            ratio = min(target_w / orig_w, target_h / orig_h)
            new_w, new_h = max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio))

            # cv2 for the scale and numpy for the letterbox, rather than PIL for
            # both: measured at 0.6 ms against 9 ms for a LANCZOS resize of the
            # same frame. With two cameras at 10 fps, every millisecond spent
            # here is one the Tk thread does not have — and on a downscale
            # INTER_AREA is indistinguishable from LANCZOS at this size.
            interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
            small = cv2.resize(cv_rgb, (new_w, new_h), interpolation=interp)
            canvas = np.empty((target_h, target_w, 3), np.uint8)
            canvas[:] = PANEL_RGB
            off_x, off_y = (target_w - new_w) // 2, (target_h - new_h) // 2
            canvas[off_y:off_y + new_h, off_x:off_x + new_w] = small

            # Paste into the existing Tk image rather than allocating a new one
            # per frame: measured at 20 ms against 28 ms per pane including Tk's
            # own redraw. Allocation is then paid only when the target size
            # actually changes, which means on a window resize.
            pil = Image.fromarray(canvas)
            if screen.photo is None or screen.photo_size != (target_w, target_h):
                screen.photo = ImageTk.PhotoImage(image=pil)
                screen.photo_size = (target_w, target_h)
                # Hold the reference on the widget too, or Tk garbage-collects it
                screen.video.config(image=screen.photo, text="")
                screen.video.image = screen.photo
            else:
                screen.photo.paste(pil)

            self.frames_shown += 1
            screen.note_render(time.time())
            self.total_shown.config(text=f"{self.frames_shown:,}")
        except Exception as e:
            self._record("render failed", frame_id=sensor, detail=str(e), tone="warn")

    # -------------------------------------------------------------------
    #  ZMQ receiver
    # -------------------------------------------------------------------

    def _zmq_receiver(self):
        receiver = self._zmq_ctx.socket(zmq.PULL)
        receiver.setsockopt(zmq.LINGER, 0)
        receiver.setsockopt(zmq.RCVTIMEO, 1000)
        receiver.bind(f"tcp://0.0.0.0:{GCS_PORT}")
        log.info("Listening on port %d (buffered)", GCS_PORT)

        while not self._shutting_down:
            try:
                meta_b, img_b = receiver.recv_multipart()
            except zmq.error.Again:
                continue
            except zmq.error.ContextTerminated:
                break
            except Exception as e:
                log.error("Receive failed: %s", e)
                continue

            try:
                meta = json.loads(meta_b.decode())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                log.error("Malformed metadata from master: %s", e)
                continue

            raw_fid = meta.get("frame_id", "")

            # ── Drop notice: no image, just the record ──
            if meta.get("event") == "drop":
                self._post(("drop", raw_fid, meta.get("reason", "UNKNOWN"),
                            meta.get("queue_len", "")))
                continue

            source = meta.get("source", "unknown")
            # Screened here, at the one point a master's figure enters, so
            # nothing downstream has to wonder whether it is real.
            proc_latency = sane_proc_ms(meta.get("latency_ms"))
            # Checked once per run: if the master enforces a different budget
            # than this receiver assumes, the deadline line and the buffer
            # timeout are both wrong and nothing else would say so.
            if not self._deadline_checked:
                self._deadline_checked = True
                warning = check_reported_deadline(meta.get("deadline_ms"))
                if warning:
                    log.warning("%s", warning)
            capture_ts = meta.get("capture_ts")
            sent_ts = meta.get("sent_ts")
            detections = meta.get("detections", [])

            recv_ts = time.time()
            self._observe_hop(sent_ts, recv_ts)

            # End of the line for a traced frame. The master attaches stamps to
            # only the first N of each path, so this fires a handful of times per
            # run and costs nothing the rest of it. Stamping here rather than
            # after the draw is deliberate: this is the arrival, and anything
            # later measures Tk's scheduling, not the pipeline.
            stamps = meta.get("stamps")
            if stamps:
                stamps["gcs_recv"] = recv_ts
                # Every traced frame is written; only the first couple per path
                # are rendered, so tracing a whole run does not turn the log
                # into thousands of tables.
                if frame_trace.should_print(stamps.get("path")):
                    log.info("%s", frame_trace.report(stamps, raw_fid))
                frame_trace.write_row(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "gcs_logs"), stamps, raw_fid)

            try:
                sensor, fid_str = raw_fid.split(":")
                fid_num = int(fid_str)
            except (ValueError, IndexError):
                # No usable sequence number — nothing to order against, so show it
                # straight away rather than holding it for a slot it does not have.
                display = self._composed_latency(
                    {"proc_latency": proc_latency, "receive_time": recv_ts,
                     "sent_ts": sent_ts}, recv_ts)
                self._post(("frame", "unkeyed", source, img_b,
                            display if display is not None else proc_latency,
                            proc_latency, detections))
                continue

            with self._buffer_lock:
                if sensor not in self._next_display_id:
                    self._next_display_id[sensor] = fid_num
                self._jitter_buffer.setdefault(sensor, {})[fid_num] = {
                    "source": source,
                    "img_b": img_b,
                    "proc_latency": proc_latency,
                    "capture_ts": capture_ts,
                    "sent_ts": sent_ts,
                    "detections": detections,
                    "receive_time": recv_ts,
                }

        receiver.close()
        log.info("Receiver stopped")

    def _post(self, item):
        """Hand work from the ZMQ thread to the Tk thread. Never blocks.

        If the UI falls far enough behind to fill this, dropping the newest notice
        is the right trade — stalling the receiver would back-pressure all the way
        up to the master.
        """
        try:
            self._inbox.put_nowait(item)
        except queue.Full:
            pass

    def _drain_inbox(self):
        """Run queued work on the Tk thread. Called from the display loop."""
        for _ in range(64):         # Bounded, so one burst cannot stall the UI
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                return
            try:
                if item[0] == "drop":
                    self._handle_drop(*item[1:])
                elif item[0] == "frame":
                    self._update_screen(*item[1:])
            except Exception as e:
                log.error("UI update failed: %s", e)

    def _handle_drop(self, raw_fid, reason, queue_len):
        """Log a drop and let the display loop step straight past it."""
        self._note_traffic()
        self.frames_dropped += 1
        self.total_dropped.config(text=f"{self.frames_dropped:,}", fg=BAD)

        detail = f"{reason} · queue {queue_len}" if queue_len != "" else str(reason)
        self._record("frame dropped", frame_id=raw_fid, detail=detail, tone="drop")

        try:
            sensor, fid_str = raw_fid.split(":")
            fid_num = int(fid_str)
        except (ValueError, IndexError):
            return

        # Charge the drop to the camera it came from, so a struggling sensor shows
        # up on its own screen rather than buried in a swarm-wide total.
        screen = self.screens.get(sensor)
        if screen is not None:
            screen.note_drop()

        with self._buffer_lock:
            self._dropped_ids.setdefault(sensor, set()).add(fid_num)

    # -------------------------------------------------------------------
    #  Ordered playback — this is the whole difference from the direct build
    # -------------------------------------------------------------------

    def _display_loop(self):
        """Show frames in capture order, stepping past ones that never come.

        Runs on the Tk thread. Everything it reads is produced by the ZMQ thread,
        so the entire selection step happens under the buffer lock and only the
        render — which touches no shared state — happens outside it.
        """
        if self._shutting_down:
            return

        self._drain_inbox()

        ready = []      # (sensor, payload) chosen this tick, rendered after the lock
        skips = []      # (sensor, from_id, to_id, waited) to log after the lock
        overruns = []   # (sensor, from_id, to_id, count) to log after the lock
        depths = {}     # sensor -> frames still held, for the Buffered readout

        with self._buffer_lock:
            # Every sensor advances independently, so one camera stalling never
            # holds up the other's feed.
            for sensor, next_id in list(self._next_display_id.items()):
                buffer = self._jitter_buffer.get(sensor, {})
                dropped = self._dropped_ids.get(sensor, set())

                for k in [k for k in list(buffer) if k < next_id]:
                    buffer.pop(k)

                # Frames are arriving faster than this laptop can draw them.
                # Holding the backlog would keep the display drifting further
                # into the past — the buffer is meant to absorb jitter, not to
                # become a recording. Jump to the tail and say so in the log.
                if len(buffer) > JITTER_BUFFER_MAX:
                    keep_from = max(buffer) - JITTER_BUFFER_KEEP + 1
                    stale = [k for k in buffer if k < keep_from]
                    for k in stale:
                        buffer.pop(k)
                    overruns.append((sensor, next_id, keep_from - 1, len(stale)))
                    next_id = keep_from
                    self._next_display_id[sensor] = next_id

                # The master told us these are gone — advance past them at once
                # rather than holding the feed open for something never coming.
                while next_id in dropped:
                    dropped.discard(next_id)
                    next_id += 1
                    self._next_display_id[sensor] = next_id

                if next_id in buffer:
                    ready.append((sensor, buffer.pop(next_id)))
                    self._next_display_id[sensor] = next_id + 1
                elif buffer:
                    oldest = min(buffer)
                    waited = time.time() - buffer[oldest]["receive_time"]
                    if oldest > next_id and waited > JITTER_BUFFER_TIMEOUT_SEC:
                        skips.append((sensor, next_id, oldest - 1, waited))
                        self._next_display_id[sensor] = oldest

                depths[sensor] = len(buffer)

        for sensor, first, last, waited in skips:
            screen = self.screens.get(sensor)
            if screen is not None:
                screen.note_skip(last - first + 1)
            self._record("playback skip", frame_id=f"{sensor}:{first}",
                         detail=f"no notice for {first}–{last}, waited {waited:.1f}s",
                         tone="warn")

        for sensor, first, last, count in overruns:
            screen = self.screens.get(sensor)
            if screen is not None:
                screen.note_skip(count)
            self._record("buffer overrun", frame_id=f"{sensor}:{first}",
                         detail=(f"held more than {JITTER_BUFFER_MAX} frames — dropped "
                                 f"{count} ({first}–{last}) to catch up to live"),
                         tone="warn")

        for sensor, depth in depths.items():
            screen = self.screens.get(sensor)
            if screen is not None:
                screen.note_depth(depth)

        for sensor, data in ready:
            now = time.time()
            display = self._composed_latency(data, now)
            if display is None:
                display = data["proc_latency"]

            # The old cross-clock figure, kept only so the two can be compared
            # while a time daemon happens to be running. Agreement means the
            # composition is sound; divergence names the clocks as the reason.
            cap_ts = data.get("capture_ts")
            if cap_ts and display is not None:
                legacy_ms = (now - cap_ts) * 1000
                if abs(legacy_ms - display) > CLOCK_DISAGREE_MS:
                    self._note_clock_disagreement(sensor, legacy_ms, display)

            self._update_screen(sensor, data["source"], data["img_b"], display,
                                data["proc_latency"], data["detections"])

        self.root.after(33, self._display_loop)

    # -------------------------------------------------------------------
    #  Shutdown
    # -------------------------------------------------------------------

    def _on_close(self):
        self._record("session ended",
                     detail=f"{self.frames_shown} shown, {self.frames_dropped} dropped")
        self._shutting_down = True
        try:
            self._event_file.close()
        except Exception:
            pass
        log.info("Session ended — %d shown, %d dropped. Log saved to %s",
                 self.frames_shown, self.frames_dropped, self.event_log_path)
        threading.Thread(target=self._zmq_ctx.term, daemon=True, name="zmq-term").start()
        self.root.after(500, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = DualGroundStation(root)
    root.mainloop()
