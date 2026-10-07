"""
completion_time_scheduler.py

Core load-balancing engine for the UAV swarm MEC framework.

Provides:
  - EWMA:               Exponentially weighted moving average for stable stat tracking.
  - NodeProfiler:       Per-node profiler tracking inference, RTT, thermal, and battery.
  - SwarmLoadBalancer:  Greedy earliest-completion-time scheduler.
  - estimate_local_time / enforce_deadline:
                        Shared helpers the Lyapunov and round-robin schedulers
                        import, so all three agree on what a frame costs locally
                        and on when a frame is beyond saving.

Design rationale (from the architecture guide):
  For every frame, estimate the completion time for *every* candidate and pick
  the lowest.  If none can beat the frame's remaining useful lifetime, drop it
  rather than spend compute on a result nobody will see in time.  EWMA smooths
  against jitter; a switching margin prevents path-flapping when estimates are
  close.

Extended beyond the guide with:
  - Thermal penalty:     multiplier >= 1.0 reflecting GPU throttle.
  - Battery penalty:     additive latency, or outright exclusion, for a node
                         whose battery is too low to keep taking work.
  - Volatility penalty:  a node whose mean looks fine but whose numbers swing
                         wildly is still a deadline risk the mean cannot see.
  - Reliability:         recent dispatch outcomes, which is the only signal here
                         derived from what actually happened rather than from
                         what a node reports about itself.
"""

import math
import time

# ---------------------------------------------------------------------------
# Thermal constants
# ---------------------------------------------------------------------------
# Set from the Jetson Nano's real behaviour, not from a guess.  The DVFS
# governor starts pulling clocks back around 70 °C and throttles hard by 85 °C.
#
# These were previously 45/60, which was well below where the hardware actually
# throttles: a Nano under sustained inference sits above 60 °C almost all the
# time, so every node pinned at the maximum penalty and the thermal term became
# a constant multiplier that cancelled out of every comparison.  Anything that
# is the same for all candidates cannot influence a choice between them.
#
# Re-derive these from a real swarm/thermals_*.csv if your cooling differs.
THERMAL_SAFE_C = 70.0              # Below this: no throttle penalty
THERMAL_CRITICAL_C = 85.0          # At or above: maximum throttle penalty
THERMAL_MAX_PENALTY = 1.5          # Weight at critical temperature
#
# NOT a multiplier on estimated time any more, and that was a real defect.
# infer_time is *measured*: when a Nano throttles, the measurement gets slower
# on its own, because that is what throttling does. Multiplying that measurement
# by thermal_penalty charged for the same slowdown a second time, so a genuine
# 1.5x throttle inflated the estimate by 2.2x — and it compounded over a run as
# the board heated, until the worker failed its feasibility check and offloading
# decayed to nothing.
#
# It was also asymmetric. A worker's temperature arrives in its heartbeat, while
# the master reads its own from _last_table and falls back to 0.0 when it is not
# listed there — meaning no penalty. So the double charge landed on the worker
# and usually spared the master, which is the direction the decay actually ran.
#
# Thermal now appears only where a time-independent cost belongs: the penalty
# term of the Lyapunov objective, weighted by V. Greedy and round robin
# therefore stop pricing temperature at all, which is correct — for them it was
# never anything but a second charge for a slowdown already in the numbers.

# ---------------------------------------------------------------------------
# Battery constants
# ---------------------------------------------------------------------------
# Only active when a node actually reports a battery — see
# node_common.read_battery_pct.  A bench Jetson on a wall adapter reports None
# and is correctly treated as unpenalised rather than as flat.
BATTERY_WARN_PCT = 20.0            # Below this: additive latency penalty begins
BATTERY_CRITICAL_PCT = 10.0        # At or below this: node excluded entirely
BATTERY_MAX_PENALTY_SEC = 0.5      # Maximum additive penalty at the warn threshold

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
# Memory pressure is a cliff, not a slope: a node at 70% behaves exactly like one
# at 40%, and then past this line it starts swapping or the kernel kills the
# inference process — with nothing visible in the timing measurements first.
#
# This is why RAM gets a hard exclusion and CPU gets none.  A busy CPU already
# shows up as slower measured inference, and adding a separate CPU penalty on top
# would count the same congestion twice.  Nothing else in the formula can see a
# node that is about to run out of memory.
RAM_CRITICAL_PCT = 95.0            # At or above this: node excluded entirely

# ---------------------------------------------------------------------------
# Volatility / trend / reliability constants
# ---------------------------------------------------------------------------
JITTER_SAFETY_FACTOR = 1.0         # Safety margin per second of observed volatility
QUEUE_TREND_PENALTY_SEC = 0.05     # Penalty per unit of queue growth since last tick
RELIABILITY_FLOOR = 0.5            # Dispatch-success rate below which a node is excluded
RELIABILITY_MAX_PENALTY_SEC = 0.3  # Max additive penalty for degraded reliability

# Rate at which reliability climbs back toward 1.0 for a node that is alive but
# receiving no work.  This constant is what stops the reliability floor being a
# one-way trapdoor, and it is worth being explicit about why that mattered.
#
# record_dispatch_outcome is only ever called on the dispatch path.  So once
# reliability fell under RELIABILITY_FLOOR, estimate_time returned inf, the
# scheduler stopped choosing the node, nothing was dispatched to it, and no
# further outcome was ever recorded — the score froze below the floor for the
# rest of the process's life.  Every other exclusion here (battery, RAM) is fed
# by heartbeats that keep arriving regardless of routing, so those recover on
# their own.  Reliability was the only exclusion whose evidence was gated by the
# decision it controlled, which made it permanent by construction.
#
# At 0.05/s a node written off at 0.35 is back above the floor in three seconds
# and fully trusted in thirteen, so a genuinely dead node still stays out (the
# reaper re-fails it faster than this heals it) while a node that merely hit a
# congestion burst comes back.
RELIABILITY_RECOVERY_PER_SEC = 0.05

# Completed inferences ignored for jitter purposes when a worker first connects.
# A Jetson's opening TensorRT inference measured 460 ms against a 190 ms steady
# state on this testbed, and two frames was the whole warmup.
WARMUP_SAMPLES = 2

# Ceiling on one jitter sample, as a multiple of the current mean.  A node that
# is genuinely erratic still earns a wide margin over several samples; no single
# frame can set it.
JITTER_CLAMP_FACTOR = 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  EWMA — Exponentially Weighted Moving Average
# ═══════════════════════════════════════════════════════════════════════════

class EWMA:
    """Exponentially weighted moving average.

    alpha closer to 1 reacts faster to recent samples;
    alpha closer to 0 smooths harder against noise.

    The first sample seeds the average directly — there is no prior to blend
    with — so the tracker is useful immediately after one observation instead of
    spending a dozen frames crawling away from its seed value.
    """

    __slots__ = ("alpha", "value", "_initialized")

    def __init__(self, alpha=0.25, initial=0.0):
        self.alpha = alpha
        self.value = initial
        self._initialized = False

    def update(self, sample):
        """Blend *sample* into the running average and return the new value."""
        if not self._initialized:
            self.value = sample
            self._initialized = True
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helpers — used by all three schedulers
# ═══════════════════════════════════════════════════════════════════════════

def thermal_penalty(temp_celsius):
    """Multiplier >= 1.0 reflecting GPU thermal throttle.

    Linear ramp from 1.0 at THERMAL_SAFE_C to THERMAL_MAX_PENALTY at
    THERMAL_CRITICAL_C, clipped at both ends.  A reading of 0.0 means the
    thermal zone could not be read, which lands below SAFE and so applies no
    penalty — "unknown" must never be treated as "hot".
    """
    if temp_celsius <= THERMAL_SAFE_C:
        return 1.0
    if temp_celsius >= THERMAL_CRITICAL_C:
        return THERMAL_MAX_PENALTY
    ratio = ((temp_celsius - THERMAL_SAFE_C)
             / (THERMAL_CRITICAL_C - THERMAL_SAFE_C))
    return 1.0 + ratio * (THERMAL_MAX_PENALTY - 1.0)


def estimate_local_time(local_pending, local_infer_ewma, local_temp_celsius=0.0):
    """Estimated completion time for handling this frame on the master itself.

    `local_pending + 1` because *this* frame joins the back of the queue: it is
    the (local_pending + 1)-th item to be processed, not the local_pending-th.
    NodeProfiler.estimate_time applies exactly the same +1 for remote nodes.
    Keeping both in named functions is what stops the two halves of the
    comparison drifting apart — which is precisely what had happened before.

    No network term: the master already holds the frame.
    """
    return (local_pending + 1) * local_infer_ewma.value


def enforce_deadline(estimates, preferred_node, remaining_sec):
    """Apply the frame deadline to an already-chosen node.  Returns a node or "drop".

    Every scheduler must run its choice through this, so that a comparison
    between schedulers measures scheduling quality rather than differences in
    how aggressively each one gives up.  Greedy previously dropped infeasible
    frames while Lyapunov and round-robin delivered them late, which made their
    drop rates and tail latencies incomparable by construction.

    Args:
        estimates:      {node_name: estimated_completion_seconds}
        preferred_node: what the scheduler picked on its own terms.
        remaining_sec:  deadline minus the frame's current age.

    Returns:
        *preferred_node* if it fits, otherwise the fastest node that does fit,
        otherwise "drop".
    """
    if remaining_sec <= 0.0:
        return "drop"

    preferred_time = estimates.get(preferred_node, math.inf)
    if preferred_time <= remaining_sec and not math.isinf(preferred_time):
        return preferred_node

    feasible = {
        n: t for n, t in estimates.items()
        if t <= remaining_sec and not math.isinf(t)
    }
    if not feasible:
        return "drop"
    return min(feasible, key=feasible.get)


# ═══════════════════════════════════════════════════════════════════════════
#  StalenessProbe — forced exploration, shared by greedy and Lyapunov
# ═══════════════════════════════════════════════════════════════════════════

# How long a live worker may go unmeasured before a frame is spent on it
# deliberately.  0 disables probing.
PROBE_INTERVAL_SEC = 5.0

# After this long with no measurement at all, probe a worker even if its
# estimate says it cannot meet the deadline.  See StalenessProbe for why the
# ordinary feasibility check has to be overridable.
STALE_FORCE_SEC = 30.0


class StalenessProbe:
    """Keeps a worker's statistics fresh when the scheduler stops choosing it.

    infer_time, network_rtt and both jitter EWMAs only take a sample when a
    result comes back, so a node that falls out of favour keeps whatever
    estimate it held at that moment — including one inflated by the congestion
    burst that pushed it out.  Nothing else in the system can bring it back
    down, which makes the situation self-perpetuating: too expensive to choose,
    and unable to prove otherwise without being chosen.

    This lives in the shared module, and both greedy and Lyapunov use it, for a
    specific reason.  Round robin keeps its estimates fresh structurally — it
    visits every node in rotation whatever the numbers say.  If only Lyapunov
    probed, a three-way comparison would partly be measuring which schedulers
    happen to keep fresh estimates rather than how well they schedule, and the
    conclusion would not be about scheduling at all.  Equal footing here means
    the remaining differences are attributable to the decision rule.

    Ordinarily a probe spends a routing choice and never a frame: a candidate
    whose estimate exceeds the remaining deadline is skipped.  That check has to
    be overridable, though, and the reason is the crux of the whole failure.

    A worker's opening frames are TensorRT warmup — measured at 460 ms against a
    190 ms steady state.  Fed through the jitter and RTT terms those two frames
    drove the node's estimate to 786 ms against a 600 ms deadline.  From that
    point the feasibility check declined to probe it, so no result ever came
    back, so the 786 ms stood for the rest of the run: the worker sat idle while
    the estimate that condemned it could not be revised.  Refusing to probe an
    infeasible node is right when the estimate is trustworthy, and a trap
    exactly when it is not — and a number that has gone unrefreshed for
    STALE_FORCE_SEC is not evidence of anything.

    So after a long enough silence the probe fires regardless of the estimate.
    It costs at most one frame per STALE_FORCE_SEC per worker, and it is the
    only thing here that can overturn a wrong estimate rather than defer to it.
    """

    __slots__ = ("interval", "force_after", "_clock", "_last", "count", "forced")

    def __init__(self, interval_sec=PROBE_INTERVAL_SEC, clock=time.monotonic,
                 force_after_sec=STALE_FORCE_SEC):
        self.interval = interval_sec
        self.force_after = force_after_sec
        self._clock = clock
        self._last = {}
        self.count = 0
        self.forced = 0

    def due(self, estimates, remaining_sec):
        """The worker most overdue for a measurement, or None.

        Args:
            estimates:     {node_name: estimated_completion_seconds}, "local" included.
            remaining_sec: deadline minus the frame's current age.
        """
        if self.interval <= 0.0:
            return None
        now = self._clock()
        stalest, waited, was_forced = None, self.interval, False
        for node, est in estimates.items():
            # inf is a hard health exclusion (flat battery, RAM, reliability),
            # not an estimate — never overridden.
            if node == "local" or math.isinf(est):
                continue
            # setdefault, not get: a node seen for the first time has to have
            # its clock started, or `idle` is recomputed as exactly `interval`
            # on every call and never grows. A worker whose very first estimate
            # already exceeded the deadline would then sit below the force
            # threshold forever — the same trap, entered one step earlier.
            idle = now - self._last.setdefault(node, now - self.interval)
            forced = idle >= self.force_after
            if est > remaining_sec and not forced:
                continue
            if idle >= waited:
                stalest, waited, was_forced = node, idle, forced
        if stalest is not None:
            self.count += 1
            if was_forced:
                self.forced += 1
        return stalest

    def mark(self, node):
        """Record that *node* has just been given a frame, so it is measured again."""
        if node != "local" and node != "drop":
            self._last[node] = self._clock()

    def force(self, node):
        """Make *node* maximally overdue, so the next due() picks it whatever its estimate.

        Called when a worker joins or rejoins. A node that has just appeared has
        no measurements at all, and every number the scheduler would judge it on
        is a seed rather than an observation — so deferring to those numbers is
        deferring to a guess. Backdating past force_after makes the next routing
        decision spend one frame finding out, which is the only way the guess
        gets replaced by a fact.

        This is the same override due() already applies after STALE_FORCE_SEC of
        silence; joining is simply the other moment where an estimate is known
        to be uninformed, and waiting out the timer to discover that would mean
        a worker sits idle for the first STALE_FORCE_SEC of its life.
        """
        self._last[node] = self._clock() - (self.force_after + 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  NodeProfiler — per-worker statistics + health
# ═══════════════════════════════════════════════════════════════════════════

class NodeProfiler:
    """Tracks network, inference, thermal, and battery stats for ONE remote node.

    The master's own inference time needs only a bare EWMA — it pays no network
    cost, so a full profiler would imply penalties that do not apply to it.
    estimate_local_time above handles the local side.
    """

    def __init__(self):
        # Pessimistic on purpose, and it costs nothing: EWMA.update replaces the
        # seed outright on the first real sample rather than blending into it, so
        # this value survives exactly one measurement.
        #
        # 80 ms was the old seed, chosen to match a typical cold first frame. The
        # failure that argues against it is the failover: the surviving node
        # inherits a queue, a worker rejoins with no measurements, and an
        # unmeasured node claiming 80 ms beats a master carrying any backlog at
        # all. Every frame then goes to a board still loading its TensorRT
        # engine, which cannot take them — and its queue is invisible, capped by
        # OUTGOING_QUEUE_MAX and the socket's SNDHWM, so the estimate keeps
        # under-reporting while frames age out at WORKER_RESULT_TIMEOUT_SEC.
        # Reliability then collapses and the scheduler swings to the opposite
        # failure, refusing to offload at all. Both halves of that oscillation
        # start here.
        #
        # 500 ms loses the routing comparison against any healthy master while
        # staying well inside the frame deadline, so StalenessProbe can still
        # spend one frame on the node — which is how it earns the measurement
        # that clears this seed. Bulk traffic follows only once that arrives.
        #
        # Seeding the estimate rather than padding it is what makes this reach
        # every scheduler: greedy reads infer_time through estimate_time,
        # Lyapunov through service_time, and both get the same caution.
        self.infer_time = EWMA(alpha=0.2, initial=0.500)
        self.network_rtt = EWMA(alpha=0.3, initial=0.010)   # 10 ms — measured WiFi RTT

        # Health telemetry, updated from heartbeat messages.  None consistently
        # means "not reported", which every consumer treats as "apply no
        # penalty" rather than guessing a value.
        self.temperature = 0.0          # °C; 0.0 means "not reported"
        self.battery_pct = None
        self.cpu_pct = None             # Diagnostic only — see estimate_time
        self.ram_pct = None             # Hard exclusion above RAM_CRITICAL_PCT

        # Volatility.  Mean-absolute-deviation EWMAs: cheap, robust, and unlike
        # true variance they need no buffered sample window.
        self.infer_jitter = EWMA(alpha=0.2, initial=0.010)
        self.rtt_jitter = EWMA(alpha=0.3, initial=0.003)

        # Completed inferences seen from this node.  Used only to skip the
        # warmup transient — see record_inference.
        self.samples = 0

        # Queue depth, sampled every scheduler tick rather than only on dispatch,
        # so congestion is visible even on frames routed elsewhere.
        self.queue_depth = EWMA(alpha=0.4, initial=0.0)
        self.queue_trend = 0.0          # Signed change since the last tick

        # Recent dispatch-success rate.  1.0 = everything recent worked.
        # Driven by both send failures and, via the master's in-flight reaper,
        # by results that never came back at all — and healed by
        # record_liveness_tick, without which the floor is an absorbing state.
        self.reliability = EWMA(alpha=0.1, initial=1.0)

        # Diagnostic only: how often this node's pipe was full.  Deliberately
        # not wired into any penalty — see record_backpressure.
        self.backpressure_events = 0

    # --- Stat recorders ---

    def record_inference(self, duration_sec):
        """Update smoothed inference time from a completed remote task.

        The first WARMUP_SAMPLES results move the mean but are kept out of the
        jitter estimate, and every later jitter sample is winsorised.  Both
        guard the same thing: a volatility measure that a single outlier can
        dominate.

        A Jetson's first TensorRT inference after connecting runs ~460 ms
        against a ~190 ms steady state, because the engine is still warming.
        Fed in raw, that one frame produced a deviation of |460 - 50| = 410 ms
        and drove infer_jitter to ~90 ms in one step.  estimate_time then adds
        that 90 ms as deadline-safety padding to *every* subsequent estimate of
        the node — and the only thing that can bring it back down is more
        results, which requires the node to be chosen, which the padding is
        busy preventing.  Exactly the shape of the RTT problem: a transient
        writing itself permanently into a stat that only updates on use.
        """
        self.samples += 1
        if self.samples > WARMUP_SAMPLES:
            deviation = abs(duration_sec - self.infer_time.value)
            # Winsorised: a genuine outlier should widen the margin, not define
            # it.  Standard practice for MAD estimators, and it bounds the
            # damage from any single pathological frame, not just warmup ones.
            self.infer_jitter.update(
                min(deviation, JITTER_CLAMP_FACTOR * self.infer_time.value))
        self.infer_time.update(duration_sec)

    def record_rtt(self, rtt_sec):
        """Update smoothed network round-trip time.

        Winsorised on the same reasoning as record_inference: rtt_jitter is
        added to every estimate as safety margin, so one bad sample must not be
        able to set it.
        """
        if self.network_rtt._initialized:
            deviation = abs(rtt_sec - self.network_rtt.value)
            self.rtt_jitter.update(
                min(deviation, JITTER_CLAMP_FACTOR * self.network_rtt.value))
        self.network_rtt.update(rtt_sec)

    def record_queue_observation(self, pending_tasks):
        """Feed a fresh queue-depth sample every scheduling tick.

        Called for every candidate on every tick — including workers the
        scheduler does not choose this frame.  That is what lets the trend term
        notice a queue climbing one tick before the next dispatch to that worker
        would have revealed the same thing.
        """
        previous = self.queue_depth.value
        self.queue_depth.update(pending_tasks)
        self.queue_trend = self.queue_depth.value - previous

    def record_dispatch_outcome(self, success):
        """Track whether work sent to this node actually worked out.

        Called on send failure, and — importantly — from the master's in-flight
        reaper when a dispatched frame never produces a result.  That second
        caller is what makes this signal mean anything: a node can go silent
        while its last heartbeat still claims perfect health, and this is the
        only input that notices.

        Reserved for evidence that the node is *not working*.  Do not call this
        for backpressure — see record_backpressure.
        """
        self.reliability.update(1.0 if success else 0.0)

    def record_backpressure(self):
        """Note that this node's pipe was full, WITHOUT blaming its reliability.

        A worker whose socket high-water mark is reached is telling us it is
        busy, not that it is broken, and the two need entirely different
        responses: busy means "route this frame elsewhere for now", broken means
        "stop considering this node".  Feeding backpressure into the reliability
        EWMA conflated them, and because the master's SNDHWM is 2 against a
        worker that needs ~160 ms per frame, a burst of perfectly healthy
        "please wait" signals arrived in well under a second.  At alpha=0.1 it
        takes seven of them to cross RELIABILITY_FLOOR (0.9^7 = 0.478), so a
        moment of congestion permanently retired a working node.

        The queue-depth signal already tells the scheduler this node is loaded,
        which is the correct and sufficient response.
        """
        self.backpressure_events += 1

    def record_liveness_tick(self, dt_sec):
        """Heal reliability for a node that is alive but idle.

        See RELIABILITY_RECOVERY_PER_SEC.  Called every scheduler tick for every
        candidate that is still heartbeating, which is what makes the floor a
        temporary quarantine rather than a life sentence.
        """
        if dt_sec <= 0.0 or self.reliability.value >= 1.0:
            return
        self.reliability.value = min(
            1.0, self.reliability.value + RELIABILITY_RECOVERY_PER_SEC * dt_sec)
        self.reliability._initialized = True

    def update_telemetry(self, temp_celsius=0.0, battery_pct=None,
                         cpu_pct=None, ram_pct=None):
        """Ingest health data from a heartbeat message.

        Every field is optional; a node that reports nothing stays at the
        defaults and takes no penalty.  Only what a node actually sends is
        recorded, so a missing reading never overwrites a good one with a guess.
        """
        self.temperature = temp_celsius
        if battery_pct is not None:
            self.battery_pct = battery_pct
        if cpu_pct is not None:
            self.cpu_pct = cpu_pct
        if ram_pct is not None:
            self.ram_pct = ram_pct

    # --- Penalty calculators ---

    def _thermal_penalty(self):
        """This node's thermal multiplier.  See module-level thermal_penalty()."""
        return thermal_penalty(self.temperature)

    def _battery_penalty(self):
        """Additive penalty in seconds based on battery state.

        - None or > BATTERY_WARN_PCT              → 0.0 (no penalty)
        - BATTERY_CRITICAL_PCT < pct <= WARN_PCT  → linear ramp to 0.5 s
        - <= BATTERY_CRITICAL_PCT                 → inf (exclude entirely)

        Additive rather than multiplicative on purpose: a dying node must be
        avoided even when its raw completion estimate is the lowest on offer.
        Scaling a small number by a large factor still leaves a small number.
        """
        if self.battery_pct is None or self.battery_pct > BATTERY_WARN_PCT:
            return 0.0
        if self.battery_pct <= BATTERY_CRITICAL_PCT:
            return math.inf
        ratio = ((BATTERY_WARN_PCT - self.battery_pct)
                 / (BATTERY_WARN_PCT - BATTERY_CRITICAL_PCT))
        return ratio * BATTERY_MAX_PENALTY_SEC

    # --- Cost estimators ---

    def service_time(self):
        """Marginal cost of ONE frame on this node, in seconds — no queueing.

        Split out for the Lyapunov scheduler, which needs the per-frame work
        term and the backlog term as separate quantities: drift-plus-penalty
        weights them against each other as a product, where estimate_time only
        ever needs their sum.  Keeping this deliberately free of the jitter,
        reliability and trend margins that estimate_time adds — those are
        deadline-safety padding, and padding does not belong inside a queue
        weight.
        """
        return self.infer_time.value

    def transfer_time(self):
        """Seconds of network cost to hand this node one frame.

        Serialisation is already inside the measured RTT — see the note in
        estimate_time about why adding it again biases every decision local.
        """
        return self.network_rtt.value

    def is_excluded(self):
        """True when this node must not be given work at all.

        Same three hard cuts estimate_time applies, exposed on their own so a
        scheduler can tell "excluded" apart from "merely expensive" without
        having to test an estimate for inf.
        """
        return (math.isinf(self._battery_penalty())
                or (self.ram_pct is not None and self.ram_pct >= RAM_CRITICAL_PCT)
                or self.reliability.value < RELIABILITY_FLOOR)

    def estimate_serialization_time(self, frame_bytes, link_mbps=54.0):
        """Time to put *frame_bytes* on the wire at *link_mbps*.

        Not used by estimate_time — see the note there about double-counting.
        Kept because it is the right tool if you ever schedule across links with
        very different bandwidths, where transit stops being a constant that the
        measured RTT already absorbs.  Replace the default with your measured
        iperf3 figure if you do.
        """
        bytes_per_sec = link_mbps * 1_000_000 / 8
        return frame_bytes / bytes_per_sec

    def estimate_time(self, pending_tasks, frame_bytes=0):
        """Full estimated completion time for sending this frame to this node.

        Formula::

            T = ((pending + 1) × infer_time × thermal_penalty)
              + RTT
              + battery_penalty
              + reliability_penalty
              + jitter_safety_margin
              + queue_trend_penalty

        The `+ 1` is this frame.  It used to be missing, and the consequence was
        not subtle: with an idle worker, `pending` is 0, so the whole compute
        term multiplied out to zero and the estimate collapsed to bare network
        RTT.  An idle worker looked like it could return a result in 12 ms when
        the true figure was nearer 42 ms, so essentially every frame offloaded
        regardless of load.  estimate_local_time applies the same +1 for the
        master, so the two sides of the comparison now count the same frame.

        *frame_bytes* is accepted for interface symmetry with the other
        schedulers and is deliberately not added as a transit term: serialisation
        time is already inside the measured network_rtt EWMA, and adding it again
        double-counts roughly 9 ms of phantom cost, biasing every decision toward
        local.  Use estimate_serialization_time explicitly if you need it.

        CPU load is tracked but deliberately absent from this formula for the
        same reason.  A node whose CPU is saturated decodes frames more slowly
        and runs inference more slowly, and both already land in the measured
        infer_time EWMA — charging for them again would penalise the same
        congestion twice.  CPU is reported for the logs and the report, not for
        routing.

        Returns math.inf if this node is excluded: flat battery, memory about to
        run out, or a reliability score through the floor.
        """
        battery = self._battery_penalty()
        if math.isinf(battery):
            return math.inf

        # Memory is the one health signal the timing measurements cannot see
        # coming, so it gets a hard cut rather than a penalty.
        if self.ram_pct is not None and self.ram_pct >= RAM_CRITICAL_PCT:
            return math.inf

        # A node that has been quietly failing is excluded the same way a flat
        # battery is.  Its self-reported telemetry can look perfect right up
        # until you notice nothing it was sent ever came back.
        if self.reliability.value < RELIABILITY_FLOOR:
            return math.inf
        reliability_penalty = (1.0 - self.reliability.value) * RELIABILITY_MAX_PENALTY_SEC

        # Volatility margin: two nodes with identical means are not equally safe
        # if one of them swings wildly.
        #
        # sqrt(queued), not queued.  The margin covers the *combined* deviation
        # of `queued` frame times, and independent deviations add in quadrature,
        # so the spread of their sum grows as sqrt(n) — scaling linearly charges
        # a deep queue for a worst case in which every frame runs long together.
        #
        # On this testbed that was not a small distinction.  The Nano's measured
        # RTT ranges 19-441 ms, which settles infer_jitter near 30 ms; at linear
        # scaling a queue depth of 2 padded the estimate by 90 ms and pushed the
        # node past the 600 ms deadline, so enforce_deadline vetoed it and the
        # frame was dropped rather than offloaded.  The worker was ruled out for
        # variance it does not actually have, precisely when it was needed.
        queued = pending_tasks + 1
        jitter_penalty = JITTER_SAFETY_FACTOR * (
            self.infer_jitter.value * math.sqrt(queued) + self.rtt_jitter.value
        )

        # Only penalise a queue that is actively growing.  A shrinking queue
        # earns no bonus — this term is meant to be cautious, not optimistic.
        trend_penalty = max(0.0, self.queue_trend) * QUEUE_TREND_PENALTY_SEC

        return (
            queued * self.infer_time.value
            + self.network_rtt.value
            + battery
            + reliability_penalty
            + jitter_penalty
            + trend_penalty
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SwarmLoadBalancer — greedy earliest completion time, with hysteresis
# ═══════════════════════════════════════════════════════════════════════════

class SwarmLoadBalancer:
    """Compares every candidate node per frame and returns the routing decision.

    Instantiated once on the master and called from the scheduler's hot loop.
    Not thread-safe by design — it runs only on the scheduler thread, which
    saves a lock acquisition on every single frame.

    Args:
        local_infer_ewma:   EWMA tracking the master's own inference time.
        switch_margin_sec:  Minimum advantage needed to switch away from the
                            current path.  Raise it if the logs show flapping,
                            lower it if the system reacts sluggishly.
        frame_deadline_sec: Maximum useful frame age.  Lower means dropping more
                            aggressively under load instead of delivering results
                            too late to act on.
    """

    def __init__(self, local_infer_ewma, switch_margin_sec=0.015,
                 frame_deadline_sec=0.5, probe_interval_sec=PROBE_INTERVAL_SEC,
                 clock=time.monotonic):
        self._local_ewma = local_infer_ewma
        self._margin = switch_margin_sec
        self._deadline = frame_deadline_sec
        self._last_node = "local"
        # See StalenessProbe: without this, greedy can price a worker out on a
        # stale estimate and then never gather the evidence that would correct
        # it. clock is injectable so a test can drive it from simulated time
        # rather than waiting out PROBE_INTERVAL_SEC in real seconds.
        self._probe = StalenessProbe(probe_interval_sec, clock)

    def pick(self, local_pending, remote_candidates, frame_age, frame_bytes,
             local_temp_celsius=0.0):
        """Select the node for the next frame.

        Args:
            local_pending:      Frames already queued on the master.
            remote_candidates:  {worker_id: {"pending": int, "profiler": NodeProfiler}},
                                snapshotted inside the worker-pool lock.
            frame_age:          Seconds since this frame was captured on the Pi.
            frame_bytes:        Compressed frame size.
            local_temp_celsius: Master's GPU temperature.

        Returns:
            "local", a worker_id, or "drop".
        """
        remaining = self._deadline - frame_age
        if remaining <= 0.0:
            return "drop"

        estimates = {
            "local": estimate_local_time(local_pending, self._local_ewma, local_temp_celsius)
        }
        for wid, info in remote_candidates.items():
            estimates[wid] = info["profiler"].estimate_time(info["pending"], frame_bytes)

        # Forced exploration first: a worker that has gone unmeasured too long
        # gets this frame outright, bypassing both the comparison and the
        # hysteresis. Deliberately ahead of them — the whole point is to gather
        # evidence the comparison cannot ask for on its own.
        probe = self._probe.due(estimates, remaining)
        if probe is not None:
            self._probe.mark(probe)
            self._last_node = probe
            return probe

        best_node = min(estimates, key=estimates.get)

        # Hysteresis: stay put unless the alternative is meaningfully better.
        # Without this, two nodes within a millisecond of each other trade the
        # stream back and forth every frame, and the switching itself costs more
        # than the difference being chased.
        if best_node != self._last_node and self._last_node in estimates:
            incumbent = estimates[self._last_node]
            if incumbent - estimates[best_node] < self._margin:
                best_node = self._last_node

        best_node = enforce_deadline(estimates, best_node, remaining)
        if best_node != "drop":
            self._last_node = best_node
            self._probe.mark(best_node)
        return best_node
