"""
scheduler_lyapunov.py

Lyapunov optimisation load balancer — true drift-plus-penalty.

Minimises, per frame, an upper bound on the Lyapunov drift plus a weighted
penalty::

    choose  i* = argmin_i [ Q_i(t) · w_i(t)  +  (V + Z_i(t)) · e_i(t) ]

    Q_i(t)  backlog at node i, in SECONDS of outstanding work
    w_i(t)  marginal work this frame adds at node i (service + transfer)
    e_i(t)  instantaneous penalty for using node i (thermal excess, battery)
    Z_i(t)  virtual queue enforcing a TIME-AVERAGE thermal budget on node i
    V       tradeoff weight, in s²

Why this is not the previous version
------------------------------------
The scheduler that used to live here computed ``ECT + V × thermal_penalty``.
That is the greedy earliest-completion-time rule with a rounding error attached:
thermal_penalty spans [1.0, 1.5], so at V=0.05 the whole penalty term moved the
decision by at most 25 ms, against a local-versus-remote gap measured at 230 ms
on this testbed.  The two schedulers were the same scheduler, and the three-way
comparison had nothing to compare.

The structural difference is that drift-plus-penalty weights backlog against
work as a PRODUCT, where completion time takes their SUM:

    ECT   ranks by   (Q_i + 1) · s_i        ← sum
    DPP   ranks by    Q_i · w_i + V · e_i   ← product

Under a sum, a node that is idle but slow still costs its full service time, so
a worker 4× slower than the master only wins once the master is ~5 frames deep —
by which point the frame is nearly dead against the deadline and gets dropped
instead of offloaded.  Under a product, an idle node costs Q·w = 0·w ≈ 0 and is
chosen immediately: being slow only matters once you also have a backlog.  That
is the entire reason this scheduler offloads under load where the previous one
collapsed to local and shed the overflow as drops.

Why Q is virtual and not just the queue depth
---------------------------------------------
Q_i is integrated state, not an instantaneous reading.  The physical local queue
is bounded (maxsize 30) and is *drained by dropping*, so under sustained
overload it pins at its ceiling and stops carrying information — the pressure to
offload vents as dropped frames instead of accumulating.  A virtual queue has no
ceiling: it grows without bound while arrivals exceed service, so offload
pressure keeps rising for as long as the overload lasts.  That unbounded growth
is exactly what makes the standard O(1/V) penalty / O(V) backlog tradeoff hold,
and it is what you sweep V against to produce the tradeoff curve.

It is anchored back to the observed depth each tick (Q_ANCHOR_BETA) so that
errors in w_i cannot let the virtual queue drift away from physical reality.

Deadline handling is unchanged and still routes through the shared
enforce_deadline(), so drop policy stays identical across all three schedulers
and a comparison between them measures scheduling rather than how aggressively
each one gives up.
"""

import math
import time

# NodeProfiler and EWMA are re-exported, not used here: mec_node.py imports all
# three names from whichever scheduler module is active, so each one has to offer
# the same surface.
from completion_time_scheduler import (
    NodeProfiler, EWMA, StalenessProbe,
    estimate_local_time, enforce_deadline, thermal_penalty,
    PROBE_INTERVAL_SEC,
)

# ---------------------------------------------------------------------------
# Drift-plus-penalty tunables
# ---------------------------------------------------------------------------
# V is the knob the whole method is organised around: it buys penalty reduction
# at the cost of backlog, with the standard guarantee that time-average penalty
# lands within O(1/V) of optimal while time-average backlog grows as O(V).
# Sweeping it is the experiment worth running — V=0 is pure queue balancing,
# large V is "avoid hot nodes almost regardless of latency".
#
# Units are s², because Q·w is s² and e is dimensionless.  0.02 keeps the
# penalty term comparable to a backlog of ~0.4 s on a 50 ms node, which is the
# range where the tradeoff is actually visible in the results.
DEFAULT_V = 0.02

# Time-average thermal budget.  Z_i grows while node i runs above this and
# shrinks below it, so a brief spike costs nothing and sustained heat
# accumulates real avoidance pressure.  1.05 sits just above the no-throttle
# floor of 1.0, i.e. "nodes should mostly be running cool".
THERMAL_TARGET_PENALTY = 1.05

# Ceiling on Z, and the rate it integrates at.  Both are set against the scale
# of the term Z competes with, which is what makes V mean anything:
#
#   Q·w    ranges over roughly [0, 0.33] s²  (Q up to ~1.5 s, w 0.05-0.22 s)
#   (V+Z)·(p-1)   needs a comparable range, and (p-1) tops out at 0.5
#
# so V + Z has to live in [0, ~0.6].  At the Z_MAX of 10.0 this started at, Z
# dominated the objective within seconds of any sustained heat, the sum was
# ~10 regardless of V, and sweeping V changed nothing — the tradeoff curve came
# out flat for a reason that had nothing to do with the tradeoff.
#
# Z_RATE keeps Z from slamming into its ceiling: at an excess of 0.15 it takes
# roughly a minute of sustained heat to accumulate half the available weight,
# which is the timescale a *time-average* constraint is supposed to act on.
Z_MAX = 0.5
Z_RATE = 0.05

# How hard each tick pulls the virtual queue toward the measured backlog.  Small
# enough to preserve memory across ticks, large enough that a wrong w_i cannot
# accumulate into a fiction.
Q_ANCHOR_BETA = 0.15

# Exploration (PROBE_INTERVAL_SEC, StalenessProbe) is imported from the shared
# module rather than defined here, so this scheduler and greedy run exactly the
# same one.  Round robin needs none — it visits every node in rotation whatever
# the estimates say.  If only this scheduler probed, the three-way comparison
# would partly be measuring which schedulers keep fresh statistics instead of
# how well they schedule.

# Guard against a stalled scheduler thread integrating one enormous timestep.
MAX_TICK_SEC = 0.5


class SwarmLoadBalancer:
    """Lyapunov drift-plus-penalty scheduler.

    Args:
        local_infer_ewma:   EWMA tracking the master's own inference time.
        switch_margin_sec:  Accepted for interface compatibility.  This
                            scheduler deliberately has no hysteresis — the
                            backlog term already damps oscillation (choosing a
                            node raises its own Q and so its own next cost), and
                            a second damping mechanism on top would make the
                            behaviour impossible to attribute to either.
        frame_deadline_sec: Maximum useful frame age.
        V:                  Drift/penalty tradeoff weight in s².  See DEFAULT_V.
        thermal_target:     Time-average thermal penalty each node is held to.
        probe_interval_sec: Forced-exploration interval.  0 disables probing.
    """

    def __init__(self, local_infer_ewma=None, switch_margin_sec=0.005,
                 frame_deadline_sec=0.6, V=DEFAULT_V,
                 thermal_target=THERMAL_TARGET_PENALTY,
                 probe_interval_sec=PROBE_INTERVAL_SEC,
                 clock=time.monotonic):
        # clock is injectable so a test can drive the virtual queues from
        # simulated time. Q and Z integrate over dt, so anything that could not
        # control the clock would integrate real elapsed wall time while
        # advancing simulated time separately, and measure nothing. See the
        # backlog cases in test_scheduler_fixes.py, which depend on this.
        self._clock = clock
        self._local_ewma = local_infer_ewma
        self._deadline = frame_deadline_sec
        self.V = V
        self._thermal_target = thermal_target
        self._probe = StalenessProbe(probe_interval_sec, clock)

        # Virtual queues, keyed by node name ("local" or a worker_id).
        self._Q = {}
        self._Z = {}

        self._t_last = None             # previous tick, for the drain term

        # Exposed for the report: how each decision was reached.
        self.stats = {"probe": 0, "dpp": 0, "deadline_override": 0, "drop": 0}

    # -- virtual queue bookkeeping -------------------------------------

    def _tick(self, now):
        """Seconds since the previous call, clamped."""
        if self._t_last is None:
            self._t_last = now
            return 0.0
        dt = now - self._t_last
        self._t_last = now
        return max(0.0, min(dt, MAX_TICK_SEC))

    def _update_queues(self, dt, observed_backlog, penalties):
        """Advance Q and Z one timestep.

        Q_i(t+1) = max(0, Q_i(t) - dt) anchored toward the observed backlog.
        The drain is wall-clock because Q is denominated in seconds of work: a
        busy node retires one second of backlog per second of real time.

        Z_i(t+1) = max(0, Z_i(t) + (penalty_i - target) * dt), the standard
        virtual queue for a time-average constraint.
        """
        for node, observed in observed_backlog.items():
            drained = max(0.0, self._Q.get(node, 0.0) - dt)
            self._Q[node] = ((1.0 - Q_ANCHOR_BETA) * drained
                             + Q_ANCHOR_BETA * observed)

            excess = penalties.get(node, 1.0) - self._thermal_target
            self._Z[node] = max(0.0, min(
                Z_MAX, self._Z.get(node, 0.0) + excess * dt * Z_RATE))

        # A node that has gone away stops accumulating anything.
        for stale in [n for n in self._Q if n not in observed_backlog]:
            del self._Q[stale]
            self._Z.pop(stale, None)

    def _charge(self, node, work):
        """Add this frame's work to the chosen node's backlog.

        The self-damping that replaces hysteresis: picking a node immediately
        raises its own cost for the next frame, so the stream spreads across
        nodes instead of pinning to whichever was cheapest at one instant.
        """
        self._Q[node] = self._Q.get(node, 0.0) + work

    # -- the decision --------------------------------------------------

    def pick(self, local_pending, remote_candidates, frame_age, frame_bytes,
             local_temp_celsius=0.0):
        """Select the node for the next frame.  Returns "local", a worker_id, or "drop"."""
        now = self._clock()
        dt = self._tick(now)

        remaining = self._deadline - frame_age
        if remaining <= 0.0:
            self.stats["drop"] += 1
            return "drop"

        # --- per-node work, penalty, and observed backlog ---
        local_thermal = thermal_penalty(local_temp_celsius)
        local_service = self._local_ewma.value * local_thermal

        work = {"local": local_service}
        penalty = {"local": local_thermal}
        observed = {"local": local_pending * local_service}

        for wid, info in remote_candidates.items():
            prof = info["profiler"]
            # Reliability healing (record_liveness_tick) happens in the master's
            # scheduler_loop, not here: it is bookkeeping about node health, not
            # scheduling policy, and all three schedulers need it equally.
            if prof.is_excluded():
                continue
            svc = prof.service_time()
            work[wid] = svc + prof.transfer_time()
            penalty[wid] = prof._thermal_penalty()
            observed[wid] = info["pending"] * svc

        self._update_queues(dt, observed, penalty)

        # --- completion times, for feasibility only ---
        # Kept on exactly the basis the greedy scheduler uses, so enforce_deadline
        # sees the same numbers in both and the comparison stays honest.  The DPP
        # metric decides *preference*; these decide *feasibility*.  Conflating
        # the two is what broke the original version of this file.
        estimates = {
            "local": estimate_local_time(local_pending, self._local_ewma, local_temp_celsius)
        }
        for wid, info in remote_candidates.items():
            estimates[wid] = info["profiler"].estimate_time(info["pending"], frame_bytes)

        # --- forced exploration ---
        # Shared with greedy, so the two schedulers explore identically and the
        # comparison between them isolates the decision rule. Only ever returns
        # a node that can still make the deadline: a probe spends a routing
        # choice, never a frame.
        probe = self._probe.due(estimates, remaining)
        if probe is not None:
            self.stats["probe"] += 1
            self._probe.mark(probe)
            self._charge(probe, work[probe])
            return probe

        # --- drift-plus-penalty ---
        metrics = {
            node: self._Q.get(node, 0.0) * w
                  + (self.V + self._Z.get(node, 0.0)) * max(0.0, penalty[node] - 1.0)
            for node, w in work.items()
        }
        best = min(metrics, key=metrics.get)

        decision = enforce_deadline(estimates, best, remaining)
        if decision == "drop":
            self.stats["drop"] += 1
            return decision
        if decision != best:
            self.stats["deadline_override"] += 1
        else:
            self.stats["dpp"] += 1

        self._probe.mark(decision)
        self._charge(decision, work.get(decision, local_service))
        return decision
