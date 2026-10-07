"""
gcs.py

Ground control station — dual-screen view, DIRECT playback (no jitter buffer).

Same two screens, same wire format and same measurements as gcs_buffered.py.
The one difference is the playback policy, and it is the point of shipping both:

  BUFFERED   frames are held in a per-sensor buffer and shown in capture order.
             Smooth, correctly ordered, and every frame pays the buffer wait.

  DIRECT     every frame is drawn the moment it arrives, in arrival order.
             Lowest achievable display latency; reordering is not attempted, so
             a frame that overtakes another is shown out of sequence and the
             sequence numbers are used only to *count* that, never to fix it.

Because nothing waits, this build can report "Gaps" — sequence numbers that
never turned up at all, with no drop notice to account for them. Frames that
arrive behind one already shown are still logged to the event tab as "out of
order", though the screens no longer keep a running count. Both are the price
of the latency the buffer was buying down.

The handoff queue between the ZMQ thread and Tk is intentionally shallow
(INBOX_DEPTH). It exists because tkinter is not thread-safe, not to smooth the
feed, and when it fills the *oldest* frame is discarded rather than the newest —
a display with no buffer should show the present, not work through a backlog.

Latency is composed from single-clock terms rather than subtracted across
machines: processing (measured on the master, already skew-corrected) plus
transport jitter above the observed floor plus the handoff and render measured
here. The transport floor cannot be separated from clock skew by one-way
observation, so it is excluded and reported as excluded.

Run:  python3 gcs.py
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

# Shared with gcs_buffered.py. Anything that has to hold the same value in
# both builds lives there, so the two cannot drift apart the way the deadline
# did — see the note at the top of gcs_common.py.
from gcs_common import (  # noqa: F401 — palette/consts used throughout
    FRAME_DEADLINE_MS, JITTER_BUFFER_TIMEOUT_SEC, LAST_MSG_TIMEOUT,
    GCS_PORT, HOP_STEP_THRESHOLD_SEC, CLOCK_DISAGREE_MS,
    CLOCK_DISAGREE_QUIET_SEC, EVENT_ROWS_MAX, FPS_SAMPLES,
    CHART_REPAINT_MS, SENSORS, WINDOW_SIZE,
    PAPER, PANEL, RULE, INK, MUTED, TEAL, VIOLET, TEAL_WASH,
    OK, WARN, BAD, PANEL_RGB,
    SensorScreenMixin, GroundStationMixin,
    latency_colour, sane_proc_ms, check_reported_deadline, _pick_font)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gcs-dir")

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════


# Screens drawn at startup, before any frame arrives, so the split layout is on
# screen while you are still bringing the swarm up — and a camera that never
# appears is an obviously empty screen rather than something silently missing.
#
# Not a whitelist: a sensor that turns up unlisted gets a screen of its own.
# Match the SENSOR_ID values in pi_sensor.py on each Pi.

WINDOW_TITLE = "Ground control — dual sensor — direct"

# Thread handoff only, never smoothing. Deep enough that a burst of two or three
# frames across both cameras rides through a single Tk tick, shallow enough that
# the display can never be seconds behind the link.
INBOX_DEPTH = 12
DISPLAY_TICK_MS = 16            # Drain as fast as Tk will usefully redraw


ORDER_QUIET_SEC = 10            # Rate limit on out-of-order/gap event rows

# ── Palette ────────────────────────────────────────────────────────────────
# Colour is split by job so no hue ever does two things at once.
#   Cool hues = identity — which node did the work
#   Warm hues = status   — how the system is doing right now




# Still stamped into the event log so a saved CSV names the build that wrote
# it, even though the window no longer carries a badge.
MODE_LABEL = "DIRECT"




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
        self.gaps = 0
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
        tk.Label(lat_row, text="display latency", font=(sans, 8),
                 fg=MUTED, bg=PANEL).pack(side="left", padx=(10, 0), pady=(14, 0))

        # ── Bottom furniture packed before the video ──
        # A Label's requested size follows the image it holds, so a video packed
        # first claims everything left and squeezes these out entirely.
        self.metrics = tk.Frame(self.frame, bg=PANEL)
        self.metrics.pack(side="bottom", fill="x", padx=12, pady=(6, 10))

        # The video sits in a box whose size comes from the parent, never from
        # the picture inside it. A Label's requested size follows its image, and
        # now that the video is the tallest thing on the screen that request
        # would drive the whole frame: the window could grow but never shrink,
        # and the image would overflow its cell. Switching propagation off makes
        # this box a fixed viewport that the label simply fills.
        self.video_box = tk.Frame(self.frame, bg=PANEL)
        self.video_box.pack(side="top", expand=True, fill="both", padx=1, pady=(0, 1))
        self.video_box.pack_propagate(False)

        self.video = tk.Label(self.video_box, bg=PANEL, fg=MUTED, font=(sans, 10),
                              text=f"waiting for {sensor}")
        self.video.pack(expand=True, fill="both")

        # One row rather than two: labels never move as values change, so a glance
        # at the same spot always reads the same quantity, and a single row leaves
        # the height it used to take to the video above it.
        self.values = {}
        names = ("Processing", "Feed rate", "Shown", "Dropped", "Gaps")
        for col, name in enumerate(names):
            self.values[name] = self._metric(self.metrics, name, col, 0)
            self.metrics.grid_columnconfigure(col, weight=1, uniform="metric")


    # -- numbers -------------------------------------------------------



    def note_gap(self, count):
        self.gaps += count
        self.values["Gaps"].config(text=f"{self.gaps:,}", fg=WARN)




# ═══════════════════════════════════════════════════════════════════════════
#  Ground station
# ═══════════════════════════════════════════════════════════════════════════

class DualGroundStation(GroundStationMixin):
    # Supplied to GroundStationMixin, which cannot know them.
    SCREEN_CLASS = SensorScreen
    MODE_LABEL = "DIRECT"
    log = log

    """Two screens, one per Pi, drawn straight off the wire with no reordering."""

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

        # ── Sequence tracking, for measurement only ──
        # There is no buffer to reorder against, so these numbers describe the
        # feed rather than repair it. Touched on the ZMQ thread alone.
        self._highest_fid = {}      # sensor -> highest frame_id seen so far
        self._dropped_fids = {}     # sensor -> ids the master said it dropped
        self._order_noted_at = {}   # sensor -> last time an order event was logged

        # Frames the UI could not keep up with. Written by the ZMQ thread, read
        # once a second by the Tk thread — a plain counter is enough for that.
        self._shed = 0

        # tkinter is not thread-safe, and root.after() from another thread only
        # happens to work while mainloop runs. This queue is the handoff, kept
        # shallow so it never becomes the buffer this build is meant not to have.
        self._inbox = queue.Queue(maxsize=INBOX_DEPTH)

        self._init_event_log()
        self._build_ui()

        for sensor in SENSORS:
            self._ensure_screen(sensor, announce=False)

        threading.Thread(target=self._zmq_receiver, daemon=True, name="zmq-recv").start()
        self.root.after(1000, self._check_link_status)
        self.root.after(CHART_REPAINT_MS, self._repaint_charts)
        self.root.after(DISPLAY_TICK_MS, self._display_loop)

    # -------------------------------------------------------------------
    #  Clock-free latency composition
    # -------------------------------------------------------------------



    def _composed_latency(self, proc_ms, sent_ts, recv_ts, now):
        """processing + transport jitter + handoff, each measured on one clock.

        The handoff term is what the buffered build spends on ordering. Here it
        is the few milliseconds between arrival and the next Tk tick, and it is
        still measured rather than assumed away — an unbuffered display that
        quietly under-reported its own latency would be worth nothing as a
        comparison against the buffered one.
        """
        if proc_ms is None:
            return None
        jitter_ms = self._hop_excess(sent_ts, recv_ts) * 1000
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
        serial = len(glob.glob(os.path.join(log_dir, "dual_direct_*.csv"))) + 1
        self.event_log_path = os.path.join(log_dir, f"dual_direct_{serial:03d}.csv")
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
        # Unique to this build: frames the link delivered but the UI could not
        # draw in time. In the buffered build these would have queued instead.
        self.shed_val = self._summary_stat(summary, "Shed by UI")
        self.elapsed_val = self._summary_stat(summary, "Session")

        split_box = tk.Frame(summary, bg=PAPER)
        split_box.pack(side="right", fill="x", expand=True, padx=(30, 0))
        self.split_label = tk.Label(split_box, text="awaiting frames",
                                    font=(self.sans, 9), fg=MUTED, bg=PAPER)
        self.split_label.pack(anchor="e")
        # The bar reuses the two node hues, so it doubles as the legend for the
        # hues used for the node labels on each screen.
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

        shed = self._shed
        self.shed_val.config(text=f"{shed:,}", fg=(WARN if shed else INK))
        self.elapsed_val.config(text=time.strftime("%M:%S", time.gmtime(now - self._start_time)))

        self.root.after(1000, self._check_link_status)


    # -------------------------------------------------------------------
    #  Frame render
    # -------------------------------------------------------------------

    def _update_screen(self, sensor, source_node, img_bytes, display_ms, proc_ms,
                       detections, order):
        """Draw one frame into its sensor's screen and update the swarm totals.

        *order* is "ok", "late" (arrived behind a frame already shown) or a
        positive int (that many sequence numbers skipped ahead). Nothing is
        corrected here — with no buffer there is nothing to correct against —
        the counters simply record what direct playback let through.
        """
        if self._shutting_down:
            return
        self._note_traffic()

        screen = self._ensure_screen(sensor)
        if isinstance(order, int) and order > 0:
            screen.note_gap(order)

        is_local = str(source_node).upper() == "MASTER"
        node_colour = TEAL if is_local else VIOLET

        if display_ms is not None:
            colour = latency_colour(display_ms, OK, WARN, BAD)
            screen.latency.config(text=f"{display_ms:.0f} ms", fg=colour)
        else:
            # Blanked rather than left alone. Skipping the update would hold the
            # previous frame's figure on screen while frames of unknown age keep
            # arriving, which reads as a healthy latency that nothing measured.
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

            # Size to the space the video box has actually been allotted.
            # The usual objection — that a Label's requested size follows its
            # image, so measuring the label and then filling it feeds back — does
            # not bite here: the cell width comes from the window through a
            # uniform grid, and an image bigger than the cell is clipped rather
            # than granted, so the measurement settles on the first frame instead
            # of growing. Deriving it from the stage instead needs a constant for
            # the height of the header, the latency line and the metrics row,
            # and that constant is wrong the moment any of them
            # changes — which is how frames ended up clipped by 70px.
            #
            # Before the first layout pass winfo_ returns 1, so fall back to a
            # stage-derived estimate until Tk has placed things.
            vid_w, vid_h = screen.video_box.winfo_width(), screen.video_box.winfo_height()
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
    #  ZMQ receiver — straight from the wire to the screen
    # -------------------------------------------------------------------

    def _classify_order(self, sensor, fid_num):
        """Describe this frame's place in the sequence. Never reorders anything.

        Returns "late" for a frame behind one already shown, an int for how many
        sequence numbers vanished, or "ok". Ids the master told us it dropped are
        excluded: those are already counted as drops, and charging them here too
        would make the feed look worse than it is. What is left is the frame that
        went missing with no explanation at all — exactly what the buffered build
        would have waited JITTER_BUFFER_TIMEOUT_SEC for before giving up.
        """
        highest = self._highest_fid.get(sensor)
        if highest is None:
            self._highest_fid[sensor] = fid_num
            return "ok"
        if fid_num <= highest:
            return "late"
        self._highest_fid[sensor] = fid_num

        jump = fid_num - highest - 1
        if jump == 0:
            return "ok"
        # A jump this large is a sensor restart or a long outage, not a handful
        # of lost frames. Walking it id by id would stall the receiver, and the
        # notices could not account for it anyway.
        if jump > 1000:
            return jump

        notified = self._dropped_fids.get(sensor, set())
        missing = 0
        for fid in range(highest + 1, fid_num):
            if fid in notified:
                notified.discard(fid)       # accounted for, and keeps the set small
            else:
                missing += 1
        return missing or "ok"

    def _note_order_event(self, sensor, fid_num, order):
        """Log ordering trouble, rate limited so a bad link cannot flood the tab.

        Limited per sensor *and per kind*: reordering on a busy link is frequent,
        and a single shared timer would let it mask the gap rows entirely — which
        is the report you actually want, since a gap is a frame that is gone.
        """
        kind = "late" if order == "late" else "gap"
        now = time.time()
        if now - self._order_noted_at.get((sensor, kind), 0.0) < ORDER_QUIET_SEC:
            return
        self._order_noted_at[(sensor, kind)] = now
        if order == "late":
            detail = (f"arrived behind {self._highest_fid.get(sensor)} — shown out of "
                      f"sequence, no buffer to reorder against")
            self._record("out of order", frame_id=f"{sensor}:{fid_num}",
                         detail=detail, tone="warn")
        else:
            self._record("sequence gap", frame_id=f"{sensor}:{fid_num}",
                         detail=f"{order} frame(s) missing with no drop notice",
                         tone="warn")

    def _zmq_receiver(self):
        receiver = self._zmq_ctx.socket(zmq.PULL)
        receiver.setsockopt(zmq.LINGER, 0)
        receiver.setsockopt(zmq.RCVTIMEO, 1000)
        receiver.bind(f"tcp://0.0.0.0:{GCS_PORT}")
        log.info("Listening on port %d (direct, no buffer)", GCS_PORT)

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
            # Nothing is waiting on this frame here, so the notice is purely a
            # counter and a log line. In the buffered build it also releases the
            # playback slot the missing frame was holding open.
            if meta.get("event") == "drop":
                # Remembered on this thread, where the ordering check reads it,
                # so the gap this frame is about to leave is not counted twice.
                try:
                    d_sensor, d_fid = raw_fid.split(":")
                    self._dropped_fids.setdefault(d_sensor, set()).add(int(d_fid))
                except (ValueError, IndexError):
                    pass
                self._post(("drop", raw_fid, meta.get("reason", "UNKNOWN"),
                            meta.get("queue_len", "")))
                continue

            # ── Swarm membership change: no image, no frame id ──
            # The master announces a worker appearing or going away. Worth a
            # row of its own because a failover and a rejoin are the two moments
            # the offload split legitimately changes, and without a marker in
            # this log the change has to be inferred from the source column of
            # whatever frames happened to follow.
            if meta.get("event") == "worker":
                self._post(("worker", meta.get("node", "?"),
                            meta.get("state", "?"), meta.get("detail", "")))
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
                order = self._classify_order(sensor, fid_num)
                if order != "ok":
                    self._note_order_event(sensor, fid_num, order)
            except (ValueError, IndexError):
                # No usable sequence number. Nothing here depends on one, so the
                # frame goes through unchanged with its ordering unclassified.
                sensor, order = "unkeyed", "ok"

            composed = self._composed_latency(proc_latency, sent_ts, recv_ts, recv_ts)

            # True end-to-end when the clocks can carry it; composed when they
            # cannot.
            #
            # capture_ts is the Pi's clock and recv_ts is this machine's, so
            # subtracting them is only meaningful while chrony holds the rig
            # together. It does now — 1,166 traced frames on 2026-09-07 with not
            # one negative segment — so the honest number is the subtraction:
            # camera shutter to arrival here, with the master→GCS transport
            # included rather than excluded as the composed figure has to.
            #
            # The two normally differ by the transport floor alone, a few ms,
            # which is well inside CLOCK_DISAGREE_MS. A larger gap means a clock
            # has moved, and then the composed figure — immune to skew — is the
            # one to show. Without that fallback a clock step puts a 13-second
            # latency on the operator's screen, which is what happened on
            # 2026-09-05.
            display = composed
            if capture_ts and composed is not None:
                legacy_ms = (recv_ts - capture_ts) * 1000
                if abs(legacy_ms - composed) <= CLOCK_DISAGREE_MS:
                    display = legacy_ms
                else:
                    self._post(("clock", sensor, legacy_ms, composed))

            self._post(("frame", sensor, source, img_b,
                        display if display is not None else proc_latency,
                        proc_latency, detections, order, recv_ts, sent_ts))

        receiver.close()
        log.info("Receiver stopped")

    def _post(self, item):
        """Hand work to the Tk thread. Never blocks the receiver.

        This is the one place the two builds diverge on queue policy. A backlog
        on a display with no jitter buffer is a contradiction: it would mean
        showing old frames while newer ones sit behind them. So when the handoff
        is full the OLDEST item is discarded to make room, and the discard is
        counted. Blocking instead would back-pressure all the way up to the
        master; dropping the newest would leave the display stuck in the past.
        """
        try:
            self._inbox.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            stale = self._inbox.get_nowait()
            if stale[0] == "frame":
                self._shed += 1
        except queue.Empty:
            pass
        try:
            self._inbox.put_nowait(item)
        except queue.Full:
            # The Tk thread refilled it between the two calls. Losing this frame
            # is the same trade as the one above, so take it and move on.
            if item[0] == "frame":
                self._shed += 1

    def _display_loop(self):
        """Draw whatever has arrived. No ordering, no waiting, no lookahead.

        This is the entire playback policy of this build, and the reason it is
        four lines against the buffered version's forty: there is no buffer to
        scan, no next-expected id to track and no timeout to expire. Every frame
        the receiver handed over is drawn in the order it came off the wire.
        """
        if self._shutting_down:
            return

        # Bounded, so one burst cannot stall the UI. INBOX_DEPTH is smaller than
        # this, so in practice the queue is always emptied each tick.
        for _ in range(INBOX_DEPTH * 2):
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                break
            try:
                if item[0] == "drop":
                    self._handle_drop(*item[1:])
                elif item[0] == "worker":
                    self._handle_worker(*item[1:])
                elif item[0] == "clock":
                    self._note_clock_disagreement(*item[1:])
                elif item[0] == "frame":
                    (sensor, source, img_b, display, proc, dets,
                     order, recv_ts, sent_ts) = item[1:]
                    # Recompute against the moment of drawing rather than the
                    # moment of arrival, so the handoff wait is inside the figure
                    # instead of being quietly excluded from it.
                    now = time.time()
                    fresh = self._composed_latency(proc, sent_ts, recv_ts, now)
                    self._update_screen(sensor, source, img_b,
                                        fresh if fresh is not None else display,
                                        proc, dets, order)
            except Exception as e:
                log.error("UI update failed: %s", e)

        self.root.after(DISPLAY_TICK_MS, self._display_loop)

    def _handle_worker(self, node, state, detail):
        """Log a worker joining or leaving the swarm.

        Given its own row rather than folded into the frame log because this is
        the event that explains the ones around it: the offload share changing,
        a burst of drops, the source column suddenly showing one node instead of
        two. Reading that back afterwards without a marker for the moment the
        swarm changed shape means guessing.
        """
        self._note_traffic()
        label = {"joined": "worker joined",
                 "rejoined": "worker rejoined",
                 "left": "worker left"}.get(state, "worker %s" % state)
        tone = "drop" if state == "left" else "ok"
        self._record(label, frame_id=node, detail=detail, tone=tone)

    def _handle_drop(self, raw_fid, reason, queue_len):
        """Log a drop. With no buffer there is no playback slot to release."""
        self._note_traffic()
        self.frames_dropped += 1
        self.total_dropped.config(text=f"{self.frames_dropped:,}", fg=BAD)

        detail = f"{reason} · queue {queue_len}" if queue_len != "" else str(reason)
        self._record("frame dropped", frame_id=raw_fid, detail=detail, tone="drop")

        try:
            sensor, _ = raw_fid.split(":")
        except (ValueError, IndexError):
            return

        # Charge the drop to the camera it came from, so a struggling sensor shows
        # up on its own screen rather than buried in a swarm-wide total.
        screen = self.screens.get(sensor)
        if screen is not None:
            screen.note_drop()

    # -------------------------------------------------------------------
    #  Shutdown
    # -------------------------------------------------------------------

    def _on_close(self):
        self._record("session ended",
                     detail=(f"{self.frames_shown} shown, {self.frames_dropped} dropped, "
                             f"{self._shed} shed by UI"))
        self._shutting_down = True
        try:
            self._event_file.close()
        except Exception:
            pass
        log.info("Session ended — %d shown, %d dropped, %d shed. Log saved to %s",
                 self.frames_shown, self.frames_dropped, self._shed, self.event_log_path)
        threading.Thread(target=self._zmq_ctx.term, daemon=True, name="zmq-term").start()
        self.root.after(500, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = DualGroundStation(root)
    root.mainloop()
