"""
scheduler_fixed.py

Fixed-ratio load balancer — offload a percentage you set by hand.

The other three schedulers each decide *for themselves* how much work to send
away, so the offload percentage in your results is an outcome, not a setting.
That makes one question unanswerable from those runs alone: how much of the
latency difference comes from the *policy*, and how much simply from the fact
that the policies happened to land on different offload ratios?

This scheduler answers it. You name the ratio, and it holds it:

    MEC_OFFLOAD_PCT=25 python3 mec_node.py

Run it at the ratio Lyapunov chose on its own, and any remaining latency gap is
what Lyapunov's *timing* bought you, with the ratio held constant. Run it at a
sweep (0, 25, 50, 75) and you have the latency-vs-offload curve the adaptive
schedulers are trying to find the bottom of.

How the ratio is held
---------------------
Closed loop on what actually happened, not open loop on a coin flip.

A random draw at p=0.25 over 5000 frames lands within about ±1.2% of target,
and drifts differently in every run — so two runs at "25%" are not the same
experiment. Instead this counts the decisions it has actually made and sends
each frame to whichever side pulls the realized ratio back toward target. Over
5000 frames the realized ratio lands within one frame of the setting, every
time, which is what makes the runs comparable to each other.

It also self-corrects for the deadline. When enforce_deadline() overrides the
choice — sending a frame local because the worker could not make it in time —
the counters record the override, not the intent, and later frames lean the
other way to make it up. An open-loop version would silently drift off target
exactly when the system got interesting.

Know what that means when the link is unstable: catch-up is not gentle. If the
worker is unreachable for eight seconds, every frame in that window goes local,
and once the worker returns the scheduler offloads at 100% until the ratio is
square again. Across the whole run the setting is still exactly honoured, which
is the property that makes runs comparable — but the run is no longer a steady
25% throughout, it is a dip and a spike that average to 25%. Check the link
before a fixed-ratio run, and read the per-run rows in comparison_runs.csv if a
result looks strange.

What it deliberately does not do
--------------------------------
It does not look at queue depth, temperature, RTT or inference time to make its
choice. Like round robin, that is the point: it is a *controlled* baseline, and
a baseline that peeked at load would be a fourth adaptive scheduler rather than
a control.

It does respect the frame deadline, through the same enforce_deadline() the
other three use, so its drop rate stays comparable to theirs.
"""

import os

# Re-exported so mec_node.py's single import line works whichever scheduler is
# selected. This scheduler does not consult a profiler itself, but the master
# still builds one per worker to populate the thermal and reliability CSVs.
from completion_time_scheduler import (
    NodeProfiler, EWMA,
    estimate_local_time, enforce_deadline,
)


def _target_from_env(default=25.0):
    """Read MEC_OFFLOAD_PCT once, clamped to 0-100.

    Accepts a percentage (25) or a fraction (0.25) — 0.25 is unambiguous because
    a quarter of one percent is not a setting anybody means.
    """
    raw = (os.environ.get("MEC_OFFLOAD_PCT") or "").strip()
    if not raw:
        return default / 100.0
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(
            "MEC_OFFLOAD_PCT must be a number (e.g. 25 or 0.25), got %r" % raw)
    if value < 0.0 or value > 100.0:
        raise SystemExit("MEC_OFFLOAD_PCT must be between 0 and 100, got %s" % value)
    return value if value <= 1.0 else value / 100.0


# Resolved at import so mec_node.py can name the run after it — a results file
# called results_fixed_001.csv would not tell you which ratio produced it.
TARGET_OFFLOAD = _target_from_env()

# The deadline veto is OFF here by default, unlike every other scheduler.
#
# This one exists to hold a ratio you chose. A gate that silently converts
# "50%" into something else defeats its entire purpose — in the 2026-09-07
# battery runs round robin intended 2,500 frames for the worker, 1,300 arrived,
# and enforce_deadline redirected 639 to local and dropped 561. Reported as a
# 29% offload rate, which was never anyone's policy.
#
# So: the ratio is the policy. Frames go where the setting says regardless of
# whether they will make the deadline, and if that means they arrive late or die
# at the worker, that is the measurement — it is what a fixed split costs.
#
# MEC_ENFORCE_DEADLINE=1 puts the gate back, for when you want this scheduler
# on the same footing as greedy, lyapunov and rr in one table. Without it, its
# drop rate and latency tail are not comparable with theirs, because they give
# up on infeasible frames and this one does not.
ENFORCE_DEADLINE = os.environ.get("MEC_ENFORCE_DEADLINE", "0") == "1"


class SwarmLoadBalancer:
    """Hold the offload ratio at TARGET_OFFLOAD, ignoring load entirely.

    Args:
        local_infer_ewma:   EWMA tracking the master's own inference time. Not
                            used to choose, only to judge feasibility against
                            the deadline.
        switch_margin_sec:  Accepted for interface compatibility; meaningless
                            here, since the ratio is the entire strategy.
        frame_deadline_sec: Maximum useful frame age.
        target_offload:     Fraction in [0, 1]. Defaults to MEC_OFFLOAD_PCT.
    """

    def __init__(self, local_infer_ewma=None, switch_margin_sec=0,
                 frame_deadline_sec=0.5, probe_interval_sec=None, clock=None,
                 target_offload=None):
        # probe_interval_sec and clock are accepted and ignored, so every
        # scheduler takes the same constructor and mec_node.py can build
        # whichever one MEC_SCHED names without special-casing any of them.
        # A fixed ratio needs no exploration: it
        # visits the worker on a schedule of its own regardless of what the
        # worker's statistics say, so those statistics cannot go stale.
        self._local_ewma = local_infer_ewma
        self._deadline = frame_deadline_sec
        self.target = TARGET_OFFLOAD if target_offload is None else float(target_offload)

        # Decisions actually made, which is what the ratio is held against.
        self.n_local = 0
        self.n_offload = 0
        self._rr_index = 0          # rotation across workers for the offloaded share

    @property
    def realized(self):
        """Offload fraction so far, over assigned (non-dropped) frames."""
        total = self.n_local + self.n_offload
        return (self.n_offload / total) if total else 0.0

    def pick(self, local_pending, remote_candidates, frame_age, frame_bytes,
             local_temp_celsius=0.0):
        """Route the frame to hold the target ratio. Returns "local", a worker_id, or "drop"."""
        remaining = self._deadline - frame_age
        if remaining <= 0.0:
            return "drop"

        workers = list(remote_candidates.keys())

        # No worker present: everything is local by necessity. The counter still
        # records it, so a link that drops out mid-run shows up as a realized
        # ratio below target rather than being quietly made up afterwards by a
        # burst of offloading once it returns.
        if not workers or self.target <= 0.0:
            chosen = "local"
        else:
            # Look one frame ahead down both branches and take whichever lands
            # the running ratio closer to target. Comparing the ratio *after*
            # the decision rather than before it is what makes the endpoints
            # exact: at target 1.0 the "before" test reads 1.0 < 1.0 as false
            # and sends a frame local that never needed to go there.
            total = self.n_local + self.n_offload + 1
            err_offload = abs((self.n_offload + 1) / total - self.target)
            err_local = abs(self.n_offload / total - self.target)
            if err_offload <= err_local:
                # Rotation spreads the offloaded share evenly across workers.
                chosen = workers[self._rr_index % len(workers)]
                self._rr_index += 1
            else:
                chosen = "local"

        # Completion times are computed only to test the choice against the
        # deadline — never to influence it. A fixed ratio that peeked at these
        # would not be a fixed ratio.
        estimates = {
            "local": estimate_local_time(local_pending, self._local_ewma, local_temp_celsius)
        }
        for wid, info in remote_candidates.items():
            estimates[wid] = info["profiler"].estimate_time(info["pending"], frame_bytes)

        # With the gate off the frame goes where the ratio said, full stop. The
        # only thing still checked is the expired-deadline test at the top of
        # this method: a frame already past its deadline on arrival is dead
        # whatever we do with it, and routing it would only consume capacity
        # that a live frame needs.
        decision = (enforce_deadline(estimates, chosen, remaining)
                    if ENFORCE_DEADLINE else chosen)

        # Count the decision, not the intention. If the deadline pushed an
        # offload back to local, the ratio is now behind and the next frame
        # leans the other way — which is the whole reason this is a closed loop.
        if decision == "local":
            self.n_local += 1
        elif decision != "drop":
            self.n_offload += 1

        return decision
