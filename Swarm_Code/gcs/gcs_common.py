"""Constants shared by the two ground-control builds.

Both gcs.py and gcs_buffered.py import from here, so a value
that has to be the same in both is written once. It exists because it wasn't:
FRAME_DEADLINE_MS drifted to four different values across five files while the
nodes ran a fifth, and the latency colours on screen were tuned against a budget
that had since doubled.

The frame deadline itself is not really ours. It is enforced on the master, in
mec_node.py, and everything here is downstream of that. Keep FRAME_DEADLINE_MS
equal to mec_node.py's FRAME_DEADLINE_SEC; when the master reports its own value
on the wire the receivers check it against this one and say so if they disagree,
so a future drift shows up as a log line instead of a wrong dashed line.
"""

import csv
import os
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

# ── Frame budget ──────────────────────────────────────────────────────────
#
# MUST equal mec_node.py's FRAME_DEADLINE_SEC (× 1000). The master drops a frame
# older than this before dispatch, so nothing beyond it can ever reach a display.
#
# Reads the same MEC_DEADLINE_MS the master reads, so one variable set on both
# machines keeps them aligned. They still have to be set separately — the two
# programs run on different boards — and check_reported_deadline() below exists
# for exactly the case where somebody sets one and forgets the other.
FRAME_DEADLINE_MS = float(os.environ.get("MEC_DEADLINE_MS", "1200"))


# ── Latency colours ───────────────────────────────────────────────────────
#
# Derived from the budget rather than fixed, which is the whole point. They were
# 150 ms and 350 ms, chosen when the master was a laptop and the budget was 600
# ms. Against a 1200 ms budget those thresholds painted 71% of a healthy run red
# — frames comfortably inside their deadline, shown in the colour that means
# "past deadline". A recording of that run reads as a system in permanent
# failure, and the first thing anyone concludes from it is wrong.
#
# As fractions of the budget: comfortable below a quarter, worth watching to
# three fifths, genuinely at risk beyond that.
LATENCY_OK_FRACTION = 0.25
LATENCY_WARN_FRACTION = 0.60

LATENCY_OK_MS = FRAME_DEADLINE_MS * LATENCY_OK_FRACTION      # 300 ms
LATENCY_WARN_MS = FRAME_DEADLINE_MS * LATENCY_WARN_FRACTION  # 720 ms


def latency_colour(display_ms, ok, warn, bad):
    """Pick a colour for a latency figure, as a fraction of the frame budget.

    The palette is passed in because the two builds define their own; only the
    thresholds are shared.
    """
    if display_ms < LATENCY_OK_MS:
        return ok
    if display_ms < LATENCY_WARN_MS:
        return warn
    return bad


# ── Jitter buffer ─────────────────────────────────────────────────────────
#
# How long the buffered build waits for a frame that has not arrived before it
# gives up and moves on. It has to exceed the frame budget: a frame the master
# still considers live is one that can still turn up, and abandoning it early
# throws away a frame that was going to arrive and counts it as lost. It was
# 1.0 s against a 0.6 s budget, which was correct, and stayed 1.0 s when the
# budget became 1.2 s, which was not.
#
# The margin covers the master→GCS hop for a frame dispatched at the very edge
# of its deadline.
JITTER_BUFFER_MARGIN_SEC = 0.4
JITTER_BUFFER_TIMEOUT_SEC = FRAME_DEADLINE_MS / 1000.0 + JITTER_BUFFER_MARGIN_SEC


# ── Reported latency sanity ───────────────────────────────────────────────
#
# Ceiling on the processing figure a master reports, above which it is not a
# slow frame but a broken clock.
#
# The master derives that figure by subtracting the Pi's capture stamp from its
# own clock. When a master comes up with a wrong clock — no RTC, promoted mid-
# flight, stepped by a time daemon — the subtraction spans two epochs and the
# number arrives as tens of billions of milliseconds. It has been rendered on
# this dashboard verbatim: "149294645155 ms", which is four and a half years.
#
# Nothing on this link is slow in seconds, let alone minutes, so anything past
# this is a clock artefact and not a measurement.
PLAUSIBLE_LATENCY_MS = 60_000


def sane_proc_ms(value):
    """The master's processing figure, or None if it cannot be one.

    Rejects non-numbers, negatives (the master's clock behind the Pi's) and
    anything past the plausible ceiling (ahead of it). Returning None routes the
    frame down the same path as a master that sent no figure at all, so one bad
    clock costs a dash on the display instead of a fictional number in the
    trace, the average and the CSV.
    """
    if value is None:
        return None
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return None
    if ms != ms or ms < 0.0 or ms > PLAUSIBLE_LATENCY_MS:   # ms != ms catches NaN
        return None
    return ms


def check_reported_deadline(reported_ms):
    """Compare the master's own frame budget against ours.

    Returns a message to log once, or None when they agree or the master is old
    enough not to report one. This is the drift alarm: the nodes ran a 0.6 s
    budget for weeks while this side assumed 1.2 s, and nothing anywhere said so.
    """
    if reported_ms is None:
        return None
    try:
        theirs = float(reported_ms)
    except (TypeError, ValueError):
        return None
    if abs(theirs - FRAME_DEADLINE_MS) < 1.0:
        return None
    return (f"Frame budget disagreement: master enforces {theirs:.0f} ms, this "
            f"receiver assumes {FRAME_DEADLINE_MS:.0f} ms. The deadline line on "
            f"the traces and the jitter-buffer timeout are both wrong until "
            f"FRAME_DEADLINE_MS in gcs_common.py matches mec_node.py.")


# ═══════════════════════════════════════════════════════════════════════════
#  Shared presentation and plumbing
# ═══════════════════════════════════════════════════════════════════════════
#
# Everything below was duplicated, byte for byte, in gcs.py and
# gcs_buffered.py. Two copies of the same code is two places for a fix to
# land in one of, which is how the frame budget came to hold four values across
# five files. The builds differ in playback policy; they were never meant to
# differ in how a number is drawn or an event row is written.
#
# What stayed behind in each build is what genuinely differs: SensorScreen (the
# buffered one carries a latency trace), the display loop, the receive loop, and
# the queue policy. Those are the comparison, so they stay visible in the file
# that implements them.
#
# Three names cannot live here because they are per-build. Each station supplies
# them as class attributes:
#
#   SCREEN_CLASS  the build's own SensorScreen
#   MODE_LABEL    "DIRECT" or "BUFFERED", stamped into the event log
#   log           that build's logger, so lines stay attributable

# ── Shared plumbing constants ─────────────────────────────────────────────
# Identical in both builds; the extracted methods below reference them.
GCS_PORT = 6000                 # Must match GCS_PORT in mec_node.py
LAST_MSG_TIMEOUT = 5.0          # Seconds with no message → link flagged down
HOP_STEP_THRESHOLD_SEC = 1.0    # Beyond this, a hop sample is a clock step
CLOCK_DISAGREE_MS = 50          # Composed vs cross-clock gap worth reporting
CLOCK_DISAGREE_QUIET_SEC = 30   # Report it at most this often, per sensor
EVENT_ROWS_MAX = 2000           # Cap in-memory event rows; the CSV keeps all
FPS_SAMPLES = 30                # Render stamps kept for the feed-rate figure
CHART_REPAINT_MS = 160          # Traces and split bar redraw on this timer
WINDOW_SIZE = "1420x820"

# Screens drawn at startup, before any frame arrives. Not a whitelist: a sensor
# that turns up unlisted gets a screen of its own. Match SENSOR_ID in
# pi_sensor.py on each Pi.
SENSORS = ["pi1", "pi2"]

# Fill under the buffered build's latency trace. Unused by the direct build,
# which draws no trace.
TEAL_WASH = "#C6E2E9"


# ── Palette ────────────────────────────────────────────────────────────────
# Colour is split by job so no hue ever does two things at once.
#   Cool hues = identity — which node did the work
#   Warm hues = status   — how the system is doing right now
PAPER = "#D6E7F8"       # Blue ground
PANEL = "#F1F7FE"       # Raised surfaces
RULE = "#A9C7E4"        # Hairlines
INK = "#0F2438"         # Primary text — deep navy
MUTED = "#5A7B99"       # Labels, secondary text

TEAL = "#0C6C7A"        # Local  — work done on the master itself
VIOLET = "#63459A"      # Remote — work offloaded to a worker

OK = "#1F8A57"          # Within deadline
WARN = "#BE7C10"        # Approaching deadline
BAD = "#D24B32"         # Past deadline, or dropped

PANEL_RGB = (0xF1, 0xF7, 0xFE)


def _pick_font(candidates, fallback):
    """Return the first installed family from candidates, else fallback."""
    try:
        available = set(tkfont.families())
    except Exception:
        return fallback
    for name in candidates:
        if name in available:
            return name
    return fallback


class SensorScreenMixin:
    """Per-sensor readouts that do not depend on playback policy."""

    def _metric(self, parent, label, col, row):
        tk.Label(parent, text=label, font=(self.sans, 8), fg=MUTED, bg=PANEL,
                 anchor="w").grid(row=row, column=col, sticky="w", pady=(4, 0))
        val = tk.Label(parent, text="—", font=(self.mono, 12), fg=INK, bg=PANEL,
                       anchor="w")
        val.grid(row=row + 1, column=col, sticky="w")
        return val

    def note_render(self, now):
        self.shown += 1
        self.render_stamps.append(now)
        self.values["Shown"].config(text=f"{self.shown:,}")
        self.values["Feed rate"].config(text=self._fps_text())

    def note_drop(self):
        self.dropped += 1
        self.values["Dropped"].config(text=f"{self.dropped:,}", fg=BAD)

    def _fps_text(self):
        if len(self.render_stamps) < 2:
            return "—"
        span = self.render_stamps[-1] - self.render_stamps[0]
        if span <= 0:
            return "—"
        return f"{(len(self.render_stamps) - 1) / span:4.1f} fps"

    def note_idle(self, now):
        """Blank the feed rate once frames stop, so a stale figure never lingers."""
        if self.render_stamps and now - self.render_stamps[-1] > 3.0:
            self.render_stamps.clear()
            self.values["Feed rate"].config(text="—")
            self.latency.config(fg=MUTED)


class GroundStationMixin:
    """Station behaviour shared by both builds.

    Requires from the concrete class: SCREEN_CLASS, MODE_LABEL, log.
    """

    def _observe_hop(self, sent_ts, recv_ts):
        """Record one (arrival − master send) sample. Both raw, skew included.

        A sample far from the current floor is a clock step, not jitter, and
        samples either side of a step describe different time bases. Keeping
        both would poison the minimum for a full window, so the history goes.
        """
        if not sent_ts:
            return
        sample = recv_ts - sent_ts
        with self._hop_lock:
            if self._hop_samples:
                floor = min(self._hop_samples)
                if abs(sample - floor) > HOP_STEP_THRESHOLD_SEC:
                    self._hop_samples.clear()
            self._hop_samples.append(sample)

    def _hop_excess(self, sent_ts, recv_ts):
        """Transport delay above the floor, in seconds. Skew-free."""
        if not sent_ts:
            return 0.0
        with self._hop_lock:
            if not self._hop_samples:
                return 0.0
            floor = min(self._hop_samples)
        return max(0.0, (recv_ts - sent_ts) - floor)

    def _note_clock_disagreement(self, sensor, legacy_ms, composed_ms):
        """Report a gap between composed and cross-clock latency, rarely.

        A persistent gap is the size of the clock error the composition is immune
        to and the naive subtraction was not — worth having in the record.
        """
        now = time.time()
        if now - self._clock_disagree_at.get(sensor, 0.0) < CLOCK_DISAGREE_QUIET_SEC:
            return
        self._clock_disagree_at[sensor] = now
        self._record("clock disagreement", frame_id=sensor,
                     detail=(f"cross-clock {legacy_ms:.0f}ms vs composed "
                             f"{composed_ms:.0f}ms — {legacy_ms - composed_ms:+.0f}ms "
                             f"of clock error"),
                     tone="warn")

    def _record(self, event, frame_id="", detail="", tone=None):
        """Write one row to the CSV and mirror it into the Events tab.

        The raw epoch sits alongside the human clock so this log joins against
        the masters' results CSVs without reparsing times.
        """
        now = time.time()
        wall = time.strftime("%H:%M:%S", time.localtime(now))
        elapsed = round(now - self._start_time, 2)

        self._event_csv.writerow([wall, round(now, 3), elapsed, event, frame_id, detail])
        self._event_file.flush()

        if self._shutting_down:
            return
        row = self.events.insert("", "end",
                                 values=(wall, f"{elapsed:.2f}", event, frame_id, detail),
                                 tags=(tone,) if tone else ())
        if self._follow_events.get():
            self.events.see(row)
        if len(self.events.get_children()) > EVENT_ROWS_MAX:
            self.events.delete(self.events.get_children()[0])

    def _summary_stat(self, parent, label):
        box = tk.Frame(parent, bg=PAPER)
        box.pack(side="left", padx=(0, 30))
        tk.Label(box, text=label, font=(self.sans, 9), fg=MUTED, bg=PAPER).pack(anchor="w")
        val = tk.Label(box, text="—", font=(self.mono, 15), fg=INK, bg=PAPER)
        val.pack(anchor="w")
        return val

    def _ensure_screen(self, sensor, announce=True):
        """Return this sensor's screen, creating a column for it on first sight.

        Two Pis fill the stage as an even split. A third camera would take a
        third of it with no layout code of its own — the uniform group keeps the
        columns equal whatever turns up.

        *announce* is False for the screens drawn at startup from SENSORS: those
        are placeholders, and logging "sensor appeared" for a camera that has
        sent nothing would be a lie in the event record.
        """
        screen = self.screens.get(sensor)
        if screen is not None:
            return screen

        column = len(self.screens)
        screen = self.SCREEN_CLASS(self.stage, sensor, self.mono, self.sans)
        # padx on every cell, not just the inner edge: put the whole gutter on
        # one side and the uniform group still hands out equal columns, but the
        # padding comes out of that column and the screens end up 12px apart in
        # width. Split evenly and they match.
        screen.frame.grid(row=0, column=column, sticky="nsew", padx=6)
        self.stage.grid_columnconfigure(column, weight=1, uniform="screen")
        self.screens[sensor] = screen

        if announce:
            self._record("sensor appeared", frame_id=sensor,
                         detail="new screen", tone="ok")
        return screen

    def _build_events_tab(self, parent):
        bar = tk.Frame(parent, bg=PAPER)
        bar.pack(fill="x", pady=(16, 10))

        tk.Label(bar, text="Saving to", font=(self.sans, 9),
                 fg=MUTED, bg=PAPER).pack(side="left")
        tk.Label(bar, text=self.event_log_path, font=(self.mono, 9),
                 fg=INK, bg=PAPER).pack(side="left", padx=(6, 0))

        self._follow_events = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Follow new events", variable=self._follow_events,
                       font=(self.sans, 9), fg=MUTED, bg=PAPER, activebackground=PAPER,
                       activeforeground=INK, selectcolor=PANEL, bd=0,
                       highlightthickness=0).pack(side="right")

        wrap = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=RULE)
        wrap.pack(expand=True, fill="both", pady=(0, 16))

        cols = ("time", "elapsed", "event", "frame", "detail")
        self.events = ttk.Treeview(wrap, columns=cols, show="headings",
                                   style="Events.Treeview")
        for col, head, width, anchor in (
            ("time", "Time", 90, "w"),
            ("elapsed", "Elapsed", 80, "e"),
            ("event", "Event", 140, "w"),
            ("frame", "Frame", 120, "w"),
            ("detail", "Detail", 520, "w"),
        ):
            self.events.heading(col, text=head, anchor=anchor)
            self.events.column(col, width=width, anchor=anchor, stretch=(col == "detail"))

        self.events.tag_configure("drop", foreground=BAD)
        self.events.tag_configure("warn", foreground=WARN)
        self.events.tag_configure("ok", foreground=OK)

        scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.events.yview)
        self.events.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.events.pack(expand=True, fill="both", padx=1, pady=1)

        self._record("session started",
                     detail=f"{self.MODE_LABEL} · listening on port {GCS_PORT}", tone="ok")

    def _draw_split(self):
        """Stacked bar: share of frames handled locally vs offloaded."""
        c = self.split_canvas
        c.delete("all")
        w = c.winfo_width() or 300
        h = c.winfo_height() or 8
        total = self.count_local + self.count_remote
        if total == 0:
            c.create_rectangle(0, 0, w, h, fill=RULE, outline="")
            return
        cut = w * (self.count_local / total)
        c.create_rectangle(0, 0, cut, h, fill=TEAL, outline="")
        c.create_rectangle(cut, 0, w, h, fill=VIOLET, outline="")
        pct = round(100 * self.count_remote / total)
        self.split_label.config(text=f"{100 - pct}% local · {pct}% offloaded")

    def _note_traffic(self):
        """Called on every inbound message. Closes out a gap if one was open."""
        self.last_msg_time = time.time()
        if self._link_down_since is not None:
            gap = time.time() - self._link_down_since
            self._record("feed resumed", detail=f"recovered after {gap:.1f}s", tone="ok")
            self.log.info("Feed resumed after %.1fs", gap)
            self._link_down_since = None

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TNotebook", background=PAPER, borderwidth=0,
                        highlightthickness=0, tabmargins=(0, 6, 0, 0))
        style.layout("TNotebook", [])       # drop clam's raised client frame
        style.configure("TNotebook.Tab", background=PAPER, foreground=MUTED,
                        font=(self.sans, 10), padding=(2, 6), borderwidth=0,
                        highlightthickness=0)
        style.map("TNotebook.Tab",
                  background=[("selected", PAPER), ("active", PAPER)],
                  foreground=[("selected", INK), ("active", INK)],
                  expand=[("selected", (0, 0, 0, 0))])
        style.configure("Events.Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=INK, font=(self.mono, 9), rowheight=22, borderwidth=0)
        style.configure("Events.Treeview.Heading", background=PAPER, foreground=MUTED,
                        font=(self.sans, 9), relief="flat", borderwidth=0)
        style.map("Events.Treeview", background=[("selected", "#B8D4EE")],
                  foreground=[("selected", INK)])

        # ── Header ──
        header = tk.Frame(self.root, bg=PAPER)
        header.pack(fill="x", padx=26, pady=(18, 0))

        tk.Label(header, text="Ground control", font=(self.sans, 17),
                 fg=INK, bg=PAPER).pack(side="left")

        self.status_dot = tk.Label(header, text="●", font=(self.sans, 11),
                                   fg=WARN, bg=PAPER)
        self.status_dot.pack(side="right", padx=(8, 0))
        self.status_label = tk.Label(header, text="Connecting", font=(self.sans, 10),
                                     fg=MUTED, bg=PAPER)
        self.status_label.pack(side="right")

        tk.Frame(self.root, bg=RULE, height=1).pack(fill="x", padx=26, pady=(13, 0))

        # ── Tabs ──
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(expand=True, fill="both", padx=26, pady=(0, 18))

        feed_tab = tk.Frame(self.tabs, bg=PAPER)
        events_tab = tk.Frame(self.tabs, bg=PAPER)
        self.tabs.add(feed_tab, text="   Feed   ")
        self.tabs.add(events_tab, text="   Events   ")

        # Events first: _record() needs its table to exist, and writing the
        # opening row leaves that tab selected. Land the operator on the video.
        self._build_events_tab(events_tab)
        self._build_feed_tab(feed_tab)
        self.tabs.select(feed_tab)

    def _repaint_charts(self):
        """Redraw the split bar on a timer rather than on every frame.

        Time the Tk thread spends drawing is time frames spend in the handoff
        queue, and past INBOX_DEPTH that is a shed frame rather than a queued
        one — so nothing that can be repainted lazily is repainted per frame.
        """
        if self._shutting_down:
            return
        if self._split_dirty:
            self._draw_split()
            self._split_dirty = False
        self.root.after(CHART_REPAINT_MS, self._repaint_charts)
