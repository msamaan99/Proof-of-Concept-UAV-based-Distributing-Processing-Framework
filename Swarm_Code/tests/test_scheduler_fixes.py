"""
test_scheduler_fixes.py

Regression tests for the defects behind "it offloads at first, then goes 100%
local and never recovers".

Each test states the defect as a property that must hold, so a later change that
reintroduces one fails here rather than in a flight test.  Run:

    python3 test_scheduler_fixes.py
"""

import math
import unittest

import completion_time_scheduler as cts
from completion_time_scheduler import NodeProfiler, EWMA
import scheduler_lyapunov as lyap


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def nano_profiler(infer=0.158, rtt=0.065):
    """A profiler matching the measured Jetson: 158 ms inference, 65 ms link."""
    p = NodeProfiler()
    p.infer_time = EWMA(alpha=0.2, initial=infer)
    p.network_rtt = EWMA(alpha=0.3, initial=rtt)
    p.infer_jitter = EWMA(alpha=0.2, initial=0.030)
    p.rtt_jitter = EWMA(alpha=0.3, initial=0.010)
    return p


class ReliabilityFloorIsNotAbsorbing(unittest.TestCase):
    """The floor must be a quarantine, not a life sentence.

    record_dispatch_outcome is only reachable on the dispatch path, so once
    reliability crossed RELIABILITY_FLOOR the node was excluded, excluded meant
    never dispatched to, and never dispatched to meant no further outcome could
    ever be recorded.  The score froze under the floor for the life of the
    process, which is why restarting the master was the only thing that brought
    a worker back.
    """

    def test_seven_consecutive_failures_exclude_the_node(self):
        p = nano_profiler()
        for _ in range(7):
            p.record_dispatch_outcome(False)
        self.assertLess(p.reliability.value, cts.RELIABILITY_FLOOR)
        self.assertTrue(p.is_excluded())
        self.assertTrue(math.isinf(p.estimate_time(0, 60_000)))

    def test_an_excluded_node_recovers_while_it_keeps_heartbeating(self):
        p = nano_profiler()
        for _ in range(10):
            p.record_dispatch_outcome(False)
        self.assertTrue(p.is_excluded())

        # Ten seconds of the master's scheduler ticks, no work sent.
        for _ in range(1000):
            p.record_liveness_tick(0.01)

        self.assertFalse(p.is_excluded())
        self.assertFalse(math.isinf(p.estimate_time(0, 60_000)))

    def test_recovery_does_not_outrun_a_genuinely_dead_node(self):
        """A node still failing must stay out, or the fix trades one bug for another."""
        p = nano_profiler()
        for _ in range(10):
            p.record_dispatch_outcome(False)
        # The reaper writes off one frame a second while healing runs each tick.
        for _ in range(30):
            for _ in range(100):
                p.record_liveness_tick(0.01)
            p.record_dispatch_outcome(False)
        self.assertTrue(p.is_excluded())


class BackpressureIsNotFailure(unittest.TestCase):
    """A full pipe means the worker is busy, not broken.

    SNDHWM is 2 against a worker needing ~160 ms a frame, so zmq.Again fires
    during ordinary congestion.  At alpha=0.1 seven of them cross the floor.
    """

    def test_backpressure_leaves_reliability_untouched(self):
        p = nano_profiler()
        for _ in range(50):
            p.record_backpressure()
        self.assertEqual(p.reliability.value, 1.0)
        self.assertFalse(p.is_excluded())
        self.assertEqual(p.backpressure_events, 50)


class JitterMarginScalesAsSqrt(unittest.TestCase):
    """Independent per-frame deviations add in quadrature, not linearly.

    Linear scaling charged a queue of n frames for the case where all n run long
    together, which on this testbed pushed the worker past the deadline at a
    depth of 2 and turned an offload into a dropped frame.
    """

    def test_margin_grows_sublinearly_with_depth(self):
        p = nano_profiler()
        base = p.estimate_time(0, 60_000)
        deep = p.estimate_time(8, 60_000)
        compute_growth = 9 * p.infer_time.value - p.infer_time.value
        margin_growth = (deep - base) - compute_growth
        # sqrt(9) - sqrt(1) = 2 jitter units, not 8.
        self.assertAlmostEqual(margin_growth, 2 * p.infer_jitter.value, places=6)


class DriftPlusPenaltyOffloadsUnderBacklog(unittest.TestCase):
    """The property the whole rewrite exists for.

    Under ECT a node 4x slower than the master only wins once the master is ~5
    frames deep, by which point the frame is nearly dead against the deadline and
    is dropped instead of offloaded.  Drift-plus-penalty weights backlog against
    work as a product, so an idle worker costs ~0 and is used before the local
    queue ever gets deep.
    """

    def setUp(self):
        self.clock = FakeClock()
        self.local = EWMA(alpha=0.3, initial=0.051)
        self.local._initialized = True
        self.sched = lyap.SwarmLoadBalancer(
            self.local, frame_deadline_sec=0.6, clock=self.clock,
            probe_interval_sec=0.0)      # off, so this tests the metric alone
        self.prof = nano_profiler()

    def _candidates(self, pending):
        return {"nano2": {"pending": pending, "profiler": self.prof}}

    def test_idle_worker_is_used_before_the_local_queue_gets_deep(self):
        seen = set()
        for _ in range(40):
            self.clock.advance(0.03)
            seen.add(self.sched.pick(
                local_pending=2, remote_candidates=self._candidates(0),
                frame_age=0.02, frame_bytes=60_000))
        self.assertIn("nano2", seen,
                      "an idle worker was never used at a local depth of 2")

    def test_sustained_backlog_forces_offload(self):
        picks = []
        for _ in range(60):
            self.clock.advance(0.03)
            picks.append(self.sched.pick(
                local_pending=6, remote_candidates=self._candidates(0),
                frame_age=0.02, frame_bytes=60_000))
        offloaded = sum(1 for p in picks if p == "nano2")
        self.assertGreater(offloaded, 0,
                           "deep local backlog produced no offloading at all")

    def test_backlog_pressure_is_unbounded_in_the_virtual_queue(self):
        """Q must keep growing under sustained overload.

        The physical local queue is capped at 30 and is drained by dropping, so
        it stops carrying information exactly when overload gets bad. A virtual
        queue has no ceiling, which is what keeps offload pressure rising.
        """
        for _ in range(200):
            self.clock.advance(0.005)      # arrivals faster than service
            self.sched.pick(local_pending=25, remote_candidates=self._candidates(3),
                            frame_age=0.02, frame_bytes=60_000)
        self.assertGreater(self.sched._Q.get("local", 0.0), 0.5)

    def test_choosing_a_node_raises_its_own_next_cost(self):
        """Self-damping is what replaces hysteresis; without it the stream pins."""
        self.clock.advance(0.03)
        self.sched.pick(local_pending=0, remote_candidates=self._candidates(0),
                        frame_age=0.02, frame_bytes=60_000)
        before = dict(self.sched._Q)
        self.sched.pick(local_pending=0, remote_candidates=self._candidates(0),
                        frame_age=0.02, frame_bytes=60_000)
        self.assertTrue(any(self.sched._Q[n] > before.get(n, 0.0) for n in self.sched._Q))


class ExplorationKeepsEstimatesFresh(unittest.TestCase):
    """Stale estimates are self-perpetuating without a probe.

    infer_time, network_rtt and both jitter EWMAs only take a sample when a
    result comes back. A node that stops being chosen keeps whatever estimate it
    held at the moment it fell out of favour — including one inflated by the
    congestion burst that pushed it out — and nothing in the system can ever
    bring it back down.
    """

    def test_a_long_unused_worker_is_probed(self):
        clock = FakeClock()
        local = EWMA(alpha=0.3, initial=0.051)
        local._initialized = True
        sched = lyap.SwarmLoadBalancer(local, frame_deadline_sec=0.6, clock=clock,
                                       probe_interval_sec=5.0)
        prof = nano_profiler()
        cands = {"nano2": {"pending": 0, "profiler": prof}}

        # A local-only stretch: cheap local, idle worker nobody asks about.
        picks = []
        for _ in range(400):
            clock.advance(0.05)
            picks.append(sched.pick(local_pending=0, remote_candidates=cands,
                                    frame_age=0.02, frame_bytes=60_000))
        self.assertGreater(sched.stats["probe"], 0,
                           "worker went unmeasured for 20 s with no probe")

    def test_probing_never_costs_a_dropped_frame(self):
        """A probe may spend a routing choice, never a frame."""
        clock = FakeClock()
        local = EWMA(alpha=0.3, initial=0.051)
        local._initialized = True
        sched = lyap.SwarmLoadBalancer(local, frame_deadline_sec=0.6, clock=clock,
                                       probe_interval_sec=0.001)
        prof = nano_profiler(infer=2.0)          # far too slow to make the deadline
        cands = {"nano2": {"pending": 0, "profiler": prof}}
        for _ in range(50):
            clock.advance(0.05)
            decision = sched.pick(local_pending=0, remote_candidates=cands,
                                  frame_age=0.02, frame_bytes=60_000)
            self.assertNotEqual(decision, "nano2",
                                "probed a worker that cannot meet the deadline")


class ExcludedNodesAreNeverSelected(unittest.TestCase):
    """Whatever the metric says, an inf estimate must not be routed to."""

    def test_dpp_skips_an_excluded_worker(self):
        clock = FakeClock()
        local = EWMA(alpha=0.3, initial=0.051)
        local._initialized = True
        sched = lyap.SwarmLoadBalancer(local, frame_deadline_sec=0.6, clock=clock)
        prof = nano_profiler()
        prof.battery_pct = 5.0                   # below BATTERY_CRITICAL_PCT
        cands = {"nano2": {"pending": 0, "profiler": prof}}
        for _ in range(30):
            clock.advance(0.03)
            self.assertNotEqual(
                sched.pick(local_pending=10, remote_candidates=cands,
                           frame_age=0.02, frame_bytes=60_000),
                "nano2")


class ColdStartDoesNotFreezeAWorkerOut(unittest.TestCase):
    """The failure seen in the field, as a property.

    A Jetson's first TensorRT inferences ran 460 ms against a 190 ms steady
    state. Fed raw into the jitter and RTT terms, those two frames drove the
    worker's estimate to 786 ms against a 600 ms deadline — and the probe's
    feasibility check then declined to probe it, so nothing ever corrected the
    number. The worker received five frames in three minutes and then nothing.
    """

    COLD, WARM = 0.460, 0.190

    def test_two_cold_frames_do_not_exceed_the_deadline(self):
        p = NodeProfiler()
        p.infer_time.update(0.050)                 # heartbeat seed, as the master does
        for _ in range(2):
            p.record_inference(self.COLD)
            p.record_rtt(0.040)
        self.assertLess(p.estimate_time(0, 60_000), 0.6,
                        "cold start alone made the worker unschedulable")

    def test_estimate_recovers_once_steady_state_arrives(self):
        p = NodeProfiler()
        p.infer_time.update(0.050)
        for _ in range(2):
            p.record_inference(self.COLD)
            p.record_rtt(0.040)
        for _ in range(10):
            p.record_inference(self.WARM)
            p.record_rtt(0.040)
        est = p.estimate_time(0, 60_000)
        self.assertLess(est, 0.35, f"estimate stayed inflated at {est*1000:.0f} ms")

    def test_a_worker_priced_out_by_a_bad_estimate_is_still_probed(self):
        """A stale number is not evidence, so feasibility must be overridable."""
        clock = FakeClock()
        probe = cts.StalenessProbe(interval_sec=5.0, clock=clock,
                                   force_after_sec=30.0)
        healthy = {"local": 0.05, "nano2": 0.30}      # as it looks on connect
        estimates = {"local": 0.05, "nano2": 0.786}   # after cold-start poisoning
        remaining = 0.58

        self.assertEqual(probe.due(healthy, remaining), "nano2")     # first look
        probe.mark("nano2")
        clock.advance(10.0)
        self.assertIsNone(probe.due(estimates, remaining),
                          "probed an infeasible node before the force window")
        clock.advance(25.0)                                          # 35 s total
        self.assertEqual(probe.due(estimates, remaining), "nano2",
                         "infeasible worker never re-probed — this is the trap")
        self.assertEqual(probe.forced, 1)

    def test_a_never_measured_worker_eventually_forces_a_probe(self):
        """The trap entered one step earlier: infeasible from the very first look."""
        clock = FakeClock()
        probe = cts.StalenessProbe(interval_sec=5.0, clock=clock,
                                   force_after_sec=30.0)
        estimates = {"local": 0.05, "nano2": 0.786}
        self.assertIsNone(probe.due(estimates, 0.58))
        clock.advance(40.0)
        self.assertEqual(probe.due(estimates, 0.58), "nano2",
                         "a worker infeasible on first sight was never probed")

    def test_a_health_excluded_node_is_never_forced(self):
        """inf means unhealthy, not mismeasured — that exclusion stands."""
        clock = FakeClock()
        probe = cts.StalenessProbe(interval_sec=5.0, clock=clock,
                                   force_after_sec=30.0)
        estimates = {"local": 0.05, "nano2": math.inf}
        probe.mark("nano2")
        clock.advance(120.0)
        self.assertIsNone(probe.due(estimates, 0.58))

if __name__ == "__main__":
    unittest.main(verbosity=2)
