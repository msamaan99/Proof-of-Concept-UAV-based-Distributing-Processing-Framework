"""
scheduler_rr.py

Round-robin load balancer — the deliberately naive baseline.

Cycles through the available nodes evenly (local → nano2 → nano3 → local …),
ignoring load, network cost, temperature, and every other signal the other two
schedulers weigh.  That is the point: it is the control case that shows what the
measurement-driven schedulers are actually buying you.

Drop-in replacement for the greedy scheduler — same constructor, same pick()
signature, swapped by changing one import line in mec_master.py.

One thing it does NOT ignore: the frame deadline.  It routes its choice through
the same enforce_deadline() the other two use, so all three give up on an
unreachable frame at the same point.  Without that, this baseline would deliver
late frames that greedy dropped, and its drop rate and latency tail would not be
comparable to anything — which would defeat the purpose of having a baseline.

Nor does it need the StalenessProbe the other two carry.  Exploration exists to
stop a worker's statistics going stale once the scheduler stops choosing it, and
this scheduler cannot stop choosing anything: rotation visits every live node
regardless of what its numbers say.  Round robin is accidentally immune to the
failure that forced the probe into the other two.
"""

# Re-exported so mec_master.py's single import line works whichever scheduler is
# selected.  Round robin does not consult a profiler itself, but the master still
# builds one per worker to populate the thermal and reliability CSVs.
import os

from completion_time_scheduler import (
    NodeProfiler, EWMA,
    estimate_local_time, enforce_deadline,
)

# MEC_NO_DEADLINE=1 removes the deadline veto, making the rotation absolute.
#
# What it is for: the rotation intends an even split, but enforce_deadline can
# overrule it, and under node asymmetry it overrules it constantly. In the
# 2026-09-07 battery runs round robin intended 2,500 frames for the worker;
# 1,300 arrived, 639 were redirected to local and 561 dropped, so the "50/50"
# baseline actually ran at 29.3%. That is the capacity-matched ratio — the
# safety net had quietly done the load balancing round robin refuses to do,
# which flatters the baseline and hides what a naive static split really costs.
#
# With this set the split is exactly 1/(1+workers) and the cost of that policy
# is visible in the drop count instead of being absorbed.
#
# What it costs: comparability. enforce_deadline exists so every scheduler gives
# up at the same point. Without it this one delivers frames late that greedy and
# lyapunov would have dropped, so its drop rate and latency tail are not on the
# same footing as theirs. A run with this set belongs beside round robin *with*
# the gate on, not in a table with the adaptive schedulers.
ENFORCE_DEADLINE = os.environ.get("MEC_NO_DEADLINE", "0") != "1"


class SwarmLoadBalancer:
    """Even rotation across every live node.

    Args:
        local_infer_ewma:   EWMA tracking the master's own inference time.  Not
                            used to choose, only to judge feasibility against the
                            deadline.
        switch_margin_sec:  Accepted for interface compatibility; meaningless
                            here, since rotating is the entire strategy.
        frame_deadline_sec: Maximum useful frame age.
    """

    def __init__(self, local_infer_ewma=None, switch_margin_sec=0,
                 frame_deadline_sec=0.5, probe_interval_sec=None, clock=None):
        # probe_interval_sec and clock are accepted and ignored, so every
        # scheduler takes the same constructor and mec_node.py can build
        # whichever one MEC_SCHED names without special-casing any of them.
        # Rotation needs no exploration and keeps no time-dependent state, so
        # there is nothing here for a clock to drive.
        self._local_ewma = local_infer_ewma
        self._deadline = frame_deadline_sec
        self._rr_index = 0

    def pick(self, local_pending, remote_candidates, frame_age, frame_bytes,
             local_temp_celsius=0.0):
        """Take the next node in the rotation.  Returns "local", a worker_id, or "drop"."""
        remaining = self._deadline - frame_age
        if remaining <= 0.0:
            return "drop"

        # Built fresh each frame so a worker joining or dying enters and leaves
        # the rotation immediately, with no bookkeeping to keep in sync.
        available = ["local"] + list(remote_candidates.keys())
        chosen = available[self._rr_index % len(available)]
        self._rr_index += 1

        # With the gate off the rotation is the whole policy and the frame goes
        # where it says. The expired-deadline test above still applies: a frame
        # already past its deadline on arrival is dead whatever we do with it,
        # and routing it would only consume capacity a live frame needs.
        if not ENFORCE_DEADLINE:
            return chosen

        # Completion times are computed only to test the choice against the
        # deadline — never to influence it.  Round robin that peeked at these
        # would not be round robin.
        estimates = {
            "local": estimate_local_time(local_pending, self._local_ewma, local_temp_celsius)
        }
        for wid, info in remote_candidates.items():
            estimates[wid] = info["profiler"].estimate_time(info["pending"], frame_bytes)

        return enforce_deadline(estimates, chosen, remaining)
