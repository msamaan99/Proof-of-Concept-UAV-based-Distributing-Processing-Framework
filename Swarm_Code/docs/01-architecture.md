# Architecture

How the system actually works, at the level you need to modify it. For the
build procedure see [`02-build-from-scratch.md`](02-build-from-scratch.md).

---

## 1. Three tiers, and why the split is where it is

| Tier | Hardware | Job | Deliberately does **not** |
|---|---|---|---|
| 1 — sensing | Raspberry Pi | capture, JPEG-compress, stream | run inference |
| 2 — edge compute | Jetson Nano ×2 | form the network, schedule, infer | render anything |
| 3 — display | laptop | draw the feeds, log events, be the clock of record | decide anything |

**Tier 1 runs no inference on purpose.** A Pi has no GPU worth using for
real-time object detection; running YOLOv5n there would be slower than the
network round-trip to a Jetson, which would make the entire offloading question
vacuous. The Pi's value is being cheap, low-power, and physically close to what
is being watched.

**Tier 3 decides nothing.** The ground station answers "is the feed healthy?",
not "how are the Nanos doing?" — different questions with different readers. Node
health lives in the master's log line and `telemetry_NNN.csv`. What tier 3 *is*
uniquely good for is being the one machine that stays up across a failover and
keeps one continuous clock, which makes its event log the authoritative timeline
for recovery time.

---

## 2. Two layers per Jetson

```
  ┌─────────────────────────── one Jetson Nano ───────────────────────────┐
  │                                                                       │
  │  swarm_net.py                          mec_node.py                    │
  │  ────────────                          ───────────                    │
  │  systemd, root, at boot                run by hand (or systemd)       │
  │                                                                       │
  │  · scan → claim or join                · reads its role from the file │
  │  · rebuild the AP if it dies           · MASTER half or WORKER half   │
  │  · UDP telemetry, 2 Hz                 · loads TensorRT once, keeps it│
  │                                                                       │
  │         │                                        ▲                    │
  │         │ writes  runtime/neighbors.json         │ polls 1 Hz         │
  │         └────────────────────────────────────────┘                    │
  │                  role, master_address, and per node:                  │
  │                  ip · cpu_pct · ram_pct · gpu_temp_c · battery_pct     │
  │                  infer_ms · queue · online · mec_active · mec_role     │
  │                                                                       │
  │         ┌────────────────────────────────────────┐                    │
  │         │ reads   runtime/local_perf.json        │ writes 1 Hz        │
  │         ▼                                        │                    │
  │  folds infer_ms + queue into the broadcast ──────┘                    │
  └───────────────────────────────────────────────────────────────────────┘
```

### Why files and not a socket

Either program can be restarted, killed, or started first without the other
noticing. Both files are written to a temporary name and `os.replace`d, which is
atomic on POSIX, so a reader never catches a half-write. Both are **stale-checked
on read** — `neighbors.json` older than 10 s means the network layer is not
running, and a stale file still names a role and a peer list that may be hours out
of date. Treating that as authoritative would be worse than having nothing.

### Why the layers are separate at all

- The network must come up on power-on and heal itself whether or not anyone is
  logged in. A network that only recovers while someone has an SSH session open
  is not fault tolerance.
- The application *could not* do this job even if you wanted it to: reconfiguring
  `wlan0` from inside an SSH session drops that session on the spot.
- `swarm_net.py` runs as **root** because `nmcli connection up` on an AP profile
  needs privileges. That is also why it does nothing else — no model, no ZMQ, no
  application sockets. The privileged process is kept as small as it can be.
- The processing layer stays manual for measurement, because that is where
  `MEC_SCHED`, `MEC_RUN_FRAMES` and `MEC_TRACE_N` go.

### What each layer deliberately omits

`mec_node.py` has **no registration handshake and no heartbeat sockets.** It used
to. Both were rediscovering, more slowly, what the network layer already knows:
who is present, at which address, and how healthy they are. A worker becomes
usable to the master when its entry in `neighbors.json` has `online`,
`mec_active`, and `mec_role: worker` all true.

`swarm_net.py` never tears the network down on exit. A service restart, or
someone stopping it to look at something, must not disconnect the sensors — and
if this board holds the access point, dropping it would take every other device
in the swarm offline.

---

## 3. The role state machine

```
        power on
           │
           ▼
   ┌───────────────┐   SSID found
   │  scan for     ├──────────────────►  activate NanoNet-client  ──►  WORKER
   │  "NanoNet"    │                     (static .51 / .55)
   └───────┬───────┘
           │ nothing after claim_delay
           ▼
     activate NanoNet-ap
     (shared, 192.168.50.1)  ─────────────────────────────────────►  MASTER
```

**Holding `192.168.50.1` *is* being the master.** There is no election, no
separate virtual IP, and no config flag. This has three consequences worth being
explicit about:

1. A board that reboots after a failure runs the same rule, finds the network its
   partner built, and joins it. **It cannot resume a role it no longer holds**,
   because the role lives in the network rather than in the code.
2. The sensors never learn about a failover. They target `192.168.50.1`, which is
   the AP's gateway address, and whichever board holds it answers. A failover
   shows up on a Pi as a burst of backpressure drops and then normal service.
3. Master can differ between runs with nothing in the logs saying so. Check it
   before every run.

### The claim delay, and why it is asymmetric

| Board | Cold-boot delay | Reclaim delay |
|---|---:|---:|
| Nano 2 | 5 s | 5 s |
| Nano 1 | 30 s | 5 s |

If both boards boot together with the same delay, both scan, both find nothing,
and both claim. Two access points answer one SSID, the Pis split between them,
and both boards believe they are master — with no way to reconcile. **The race is
symmetric, so the tie-break has to break the symmetry**; no amount of rescanning
helps.

The minimum that works is about 8 s: the AP takes 3-5 s to start beaconing, and
the loser needs a full scan cycle (`SCAN_SETTLE_SEC` + `POLL_INTERVAL_SEC`) on top
to see it. The 25 s of extra margin on Nano 1 is a demo choice — Nano 2 is the
intended master, and the cost of Nano 1 waiting a few extra seconds when it is
genuinely alone is nothing against the cost of the wrong board holding `.1` in
front of an audience.

**The reclaim delay is short on purpose.** The long delay exists to lose a race at
boot. Coming back after *holding* a role is not that situation: the network
demonstrably existed a moment ago and has just gone, so there is no race to lose.
Without the distinction, failover would take as long as the boot delay.

Two subtleties in `RoleManager.acquire()`:

- **The AP profile is deactivated before the scan, not after.** A node that has
  just yielded the master role still has its AP up, so a scan taken now finds its
  *own* SSID: it "discovers" the network, calls `_join()`, and `_join()`'s first
  act is to tear down the very access point it is trying to join. It then cannot
  associate with a network that no longer exists, returns, and repeats — a
  self-sustaining loop with a period of about 22 s.
- **`EMPTY_NETWORK_GRACE_SEC` is 0, i.e. disabled.** The idea was that an AP
  nobody joined had probably claimed in error and should stand down. In practice
  it does the opposite of what is wanted: a board that comes up before the Pis
  are powered has no stations through no fault of its own, and at 60 s it tears
  down the only network on the air. It is also now redundant, because the
  duplicate-claim case it guarded against is what the stagger prevents. Set it to
  60 to restore the old behaviour if a split brain is ever observed despite the
  stagger.

A master does **not** step down because a worker went quiet. That is a worker
problem, and tearing down a working access point over it would take the sensors
offline too.

---

## 4. The master's frame path

Nine threads. The shape of it is the design.

```
   Pi ──JPEG──► [ingest_loop]            no decode here
                     │
                     │ FrameStore.put(key, jpeg_bytes, capture_ts)
                     │ _admit(key)  →  task_queue   (maxsize 6)
                     ▼
                [scheduler_loop]  ── balancer.pick() ──┐
                     │                                │
          "local"    │                    worker_id    │        "drop"
                     ▼                                ▼            │
         local_decode_queue (30)          peer["outgoing"] (10)     │
                     │                                │            │
                     ▼                                ▼            │
           [local_decode_loop]              [sender_loop per peer] │
             cv2.imdecode                     ZMQ PUSH :5001       │
                     │                                │            │
                     ▼                          (worker infers)    │
          local_task_queue (3)                        │            │
                     │                                ▼            │
                     ▼                        [results_loop] :5003 │
         [local_inference_loop]                       │            │
              GPU, YOLOv5n                            │            │
                     │                                │            │
                     └──────────► results_queue ◄─────┘            │
                                       │  ◄─────────────────────────┘
                                       ▼
                                [writer_loop]  ──► ZMQ PUSH :6000 ──► GCS
                                       │
                                       └──► csv_queue ──► [csv_loop] ──► disk

   also: [state_loop]  polls neighbors.json, writes telemetry CSV, publishes perf
         [reaper_loop] writes off dispatches that never came back
```

### Decode happens after the routing decision, and only where the pixels are used

This was the single biggest pipeline fix. `ingest_loop` used to decode **every**
frame the moment it arrived — before the scheduler had decided anything — and
store both the decoded array and the original JPEG. The scheduler then either
used the array, or threw it away and sent the raw JPEG to the worker, which
decoded it again.

The cost, from this rig's own trace data: master decode median **6.30 ms**, and at
a 22% offload rate over a 5000-frame run that is ~7 seconds of decode work thrown
away per run, plus every frame that was dropped at the deadline after being
decoded first.

It also matters most exactly when things are worst. When the link returns after an
outage a few hundred frames arrive at once, and the ingest thread can now drain
them at socket speed instead of 6.3 ms apiece.

**Decode is its own stage rather than folded into inference** because decode is
CPU and inference is GPU. `cv2.imdecode` releases the GIL, so on the Nano's four
cores frame N+1 genuinely decodes while frame N is on the GPU. Folding it into
`local_inference_loop` would make each local frame 6.3 + 49.4 = 55.7 ms serial
instead of 49.4 ms pipelined, dropping local throughput from ~20 to ~18 fps.

One consequence: a corrupt JPEG is no longer caught at ingest. It surfaces in
`local_decode_loop` or on the worker, both of which drop it with reason
`CORRUPT_FRAME` rather than skipping it silently — because a sensor sending
garbage used to look exactly like a sensor sending nothing.

### Every queue depth is a deliberate choice

| Queue | Max | Why that number |
|---|---:|---|
| `task_queue` | **6** | This is a *latency budget*, not a buffer. At the ~5 fps a Jetson master sustains under load, 30 slots is six seconds of backlog, so a frame reached the scheduler already older than any useful deadline and was dropped without being looked at. Six slots is ~1.2 s, which the deadline can absorb. |
| `local_decode_queue` | 30 | Still JPEGs here — ~38 kB a frame, so depth is cheap. |
| `local_task_queue` | **3** | Decoded frames. Every slot is a ~921 kB BGR array. Two is already enough that the GPU never waits on the decoder; backlog belongs in the queue *ahead* of this one, where it costs 38 kB instead of 921 kB. |
| `peer["outgoing"]` | 10 | Bounds how far behind a worker can fall. This is a *second* admission control, independent of the deadline check. |
| `results_queue` | 100 | — |
| `csv_queue` | 5000 | Deep and lossy-when-full by design: a stalled SD card should cost log rows, never frames. The loss is counted and reported. |

### `_admit` sheds from the *front*

The queue used to drop the frame that had just arrived and keep the thirty already
waiting. For a live feed that is backwards: the newest frame is the only one
anyone wants, and the old ones ahead of it are precisely why it could not be
admitted. Worse, the scheduler then spent its time on frames that had aged past
the deadline while queued, so they were dropped *after* being decoded and never
reached the display — **a full queue turned into a total outage rather than a
reduced frame rate.** Shedding from the front keeps the waiting room current, and
load now shows up as a lower frame rate rather than as nothing.

### Disk is not on the frame path

`writer_loop` is the only consumer of `results_queue` **and** the only caller of
`_forward`, so whatever it spends per frame sets the drain rate for everything
behind it. An `fsync` per row to a Jetson's SD card used to sit there, and the
cost landed exactly where it did most harm: offloading roughly doubles the frame
rate through that thread, so it fell behind precisely when the worker was being
used.

The cascade from one stalled write: `results_queue` fills → `_emit` blocks up to a
second inside `results_loop` → `peer["busy"]` stays high because its decrement is
stuck behind that → the scheduler reads an idle worker as too loaded to meet the
deadline → the display goes quiet at the same moment and for the same reason.
**One stalled write, both symptoms.** Rows now go to `csv_loop` through a deep
queue, so disk latency cannot reach the frame path at all.

### Threads are wrapped so a crash is loud

A thread dying quietly is the worst failure mode here: if the CSV writer goes
down, its queue fills and every producer blocks forever on `put()` with no log
line and no exit — the whole master frozen while still apparently running.
`_guarded()` logs a traceback instead.

---

## 5. Time, and the two different things it is used for

This distinction is the most subtle part of the system, and getting it wrong once
dropped 100% of frames for minutes at a time on a deadline that had not elapsed.

| | Measured as | Clock | If the clock is wrong |
|---|---|---|---|
| **The deadline** | `time.monotonic() - arrival_mono` | **one** board's monotonic clock | nothing happens — a monotonic clock cannot be stepped |
| **Reported latency** | `recv_ts - capture_ts - offset` | master's wall clock − Pi's | a wrong number in a CSV |

**The deadline is measured from arrival here, on a monotonic clock.** It used to
be `now - capture_ts`, a subtraction across two machines' clocks, which made the
drop decision depend on those clocks agreeing. They did not: a board came up with
a clock years off, was promoted to master while still wrong, and was then stepped
by chrony mid-run. Every one of those made the age enormous and dropped
everything.

The cost of the fix is that **the deadline no longer covers the sensor hop** — a
frame delayed in transit arrives looking fresh. That hop is not measurable without
cross-clock arithmetic, which is the thing being removed. Reported latency still
spans it.

### `SensorClockOffset` — reporting only

It estimates each sensor's skew from traffic already flowing. Per frame:

```
sample     = recv_ts - capture_ts = network_delay - skew
min_sample ≈ min_network_delay - skew          (over a 30 s window)
corrected  = (true_age - skew) - (min_network_delay - skew)
           =  true_age - min_network_delay
```

Network delay is strictly positive and varies with queueing; skew is constant over
the window. Taking the minimum drives the queueing term to its floor, and
subtracting it cancels the skew exactly, leaving an underestimate bounded by the
minimum one-way delay — a few milliseconds on this LAN. **An unbounded, silent
error traded for a small, bounded, known one.**

It is per sensor (each Pi has its own clock) and time-bounded rather than
count-bounded, so a chrony step ages out of the window instead of pinning the
minimum to a stale value for the rest of the run. A sample more than 1 s from the
current floor is not jitter on this link — it is one of the two clocks being
stepped, so the history is discarded and the estimate re-converges from that
sample. That costs one frame instead of thirty seconds.

**Nothing is dropped, scheduled, or routed on this estimate.** It gates only the
reported figure, which genuinely does span two clocks and has no other way to be
computed.

### The GCS composes latency instead of subtracting across machines

```
display_latency = processing (master, already skew-corrected)
                + transport jitter above the observed floor
                + handoff / buffer wait (measured here)
```

The transport *floor* cannot be separated from clock skew by one-way observation,
so it is excluded and the display says so. The old cross-clock figure is still
computed alongside it purely so the two can be compared: agreement means the
composition is sound, divergence names the clocks as the reason, and the gap is
logged as a `clock disagreement` event.

There is also a hard sanity ceiling. A master with a wrong clock produced
`149294645155 ms` — four and a half years — rendered verbatim on the dashboard.
Anything past 60 s is now rejected as a clock artefact and shows as a dash.

---

## 6. The scheduler

Every policy exposes the same three names so `mec_node.py` can import whichever
`MEC_SCHED` selects with no special-casing: `NodeProfiler`, `EWMA`,
`SwarmLoadBalancer`.

```python
decision = balancer.pick(
    local_pending      = decode_q + infer_q + (1 if gpu_busy else 0),
    remote_candidates  = {worker_id: {"pending": int, "profiler": NodeProfiler}},
    frame_age          = time.monotonic() - entry["arrival_mono"],
    frame_bytes        = len(jpeg),
    local_temp_celsius = this board's GPU temperature,
)   # → "local" | worker_id | "drop"
```

### The shared cost model

```
estimate_local_time  = (local_pending + 1) × local_infer_ewma

NodeProfiler.estimate_time(pending, frame_bytes):
      (pending + 1) × infer_time            ← compute, including this frame
    + network_rtt                           ← measured, serialisation included
    + battery_penalty                       ← additive, or inf
    + reliability_penalty                   ← (1 - reliability) × 0.3 s
    + jitter_safety                         ← infer_jitter × √queued + rtt_jitter
    + queue_trend_penalty                   ← only if the queue is growing
```

Five decisions inside that formula, each of which was a bug in an earlier version:

**The `+ 1` is this frame.** It used to be missing. With an idle worker `pending`
is 0, so the whole compute term multiplied out to zero and the estimate collapsed
to bare network RTT — an idle worker looked like it could return a result in 12 ms
when the real figure was nearer 42 ms, so essentially **everything offloaded
regardless of load**. `estimate_local_time` applies the same `+1`, so both sides
of the comparison count the same frame. Keeping both in named functions is what
stops the two halves drifting apart, which is exactly what had happened.

**Jitter scales as √queued, not queued.** The margin covers the *combined*
deviation of `queued` frame times, and independent deviations add in quadrature.
Linear scaling charges a deep queue for a worst case in which every frame runs
long together. On this rig that was not academic: measured RTT ranged 19-441 ms,
which settles `infer_jitter` near 30 ms, and at linear scaling a queue depth of 2
padded the estimate by 90 ms and pushed the node past the deadline — so the worker
was vetoed and the frame dropped rather than offloaded. **Ruled out for variance
it does not actually have, precisely when it was needed.**

**Thermal is not a multiplier on estimated time.** `infer_time` is *measured*;
when a Nano throttles the measurement gets slower on its own, because that is what
throttling does. Multiplying it by `thermal_penalty` charged for the same slowdown
twice, so a genuine 1.5× throttle inflated the estimate by 2.2×, compounding over
a run until the worker failed its feasibility check and offloading decayed to
nothing. It was also *asymmetric*: a worker's temperature arrives in its
heartbeat, while the master reads its own from `_last_table` and falls back to 0.0
when absent — so the double charge landed on the worker and usually spared the
master, which is the direction the decay actually ran. Thermal now appears only in
the Lyapunov penalty term, where a time-independent cost belongs.

**CPU load is tracked and deliberately absent.** A node whose CPU is saturated
decodes and infers more slowly, and both already land in the measured
`infer_time`. Charging for them again penalises the same congestion twice. CPU is
reported for the logs, not for routing.

**RAM gets a hard cut and CPU gets none**, because memory pressure is a cliff, not
a slope: a node at 70% behaves exactly like one at 40%, and then past ~95% it
starts swapping or the kernel kills the inference process — with nothing visible
in the timing measurements first. It is the one health signal the timing cannot
see coming.

### `network_rtt` is measured rarely and correctly

Only frames dispatched to a worker whose queue was **empty** (`depth_at_dispatch
== 0`) update the RTT estimate. For those, `(recv - dispatch) - infer_time` *is*
the network cost exactly, with nothing to estimate.

Why not every frame: the worker starts its timer after `recv_multipart` returns,
so its reported `infer_time` covers decode and inference but **not** the time the
frame spent waiting in its `RCVHWM=2` buffer behind other frames. That wait was
landing in the RTT EWMA and being read as link latency — wrong twice over, because
`estimate_time` already charges for the worker's queue in its `(pending + 1) ×
infer_time` term. The same congestion was counted once as compute and again as
network. And it does not decay: RTT only takes a sample when a result comes back,
so once a burst inflated it, a node that stopped being chosen kept the inflated
value with nothing able to bring it down.

An earlier attempt subtracted an *estimated* `depth × infer_time`, and estimating
was the flaw: on a cold worker the EWMA still read 132 ms while the frame had
really waited 460 ms behind a warming engine, so 330 ms of queueing landed in RTT
regardless. Combined with jitter it drove the node's estimate to 786 ms against a
600 ms deadline — permanently unschedulable, and unable to correct itself because
correction needs the results that being unschedulable prevents.

### `StalenessProbe` — forced exploration

`infer_time`, `network_rtt` and both jitter EWMAs only take a sample when a result
comes back. A node that falls out of favour keeps whatever estimate it held at
that moment, **including one inflated by the congestion burst that pushed it out**.
Nothing else can bring it down: too expensive to choose, unable to prove otherwise
without being chosen.

So a worker unmeasured for `PROBE_INTERVAL_SEC` (5 s) is given a frame outright,
ahead of both the comparison and the hysteresis. Ordinarily a probe spends a
routing choice and never a frame — a candidate whose estimate exceeds the
remaining deadline is skipped.

**That check has to be overridable, and it is the crux of the whole failure.** A
worker's opening frames are TensorRT warm-up, measured at 460 ms against a 190 ms
steady state. Fed through the jitter and RTT terms those two frames drove the
estimate to 786 ms against a 600 ms deadline. From there the feasibility check
declined to probe it, so no result ever came back, so the 786 ms stood for the
rest of the run: **the worker sat idle while the estimate that condemned it could
not be revised.** After `STALE_FORCE_SEC` (30 s) of silence the probe now fires
regardless of the estimate. It costs at most one frame per 30 s per worker, and it
is the only thing here that can overturn a wrong estimate rather than defer to it.

Health exclusions (`inf`) are never overridden — a flat battery is not a stale
estimate.

Greedy and Lyapunov share this probe deliberately. Round robin needs none: it
visits every live node in rotation whatever the numbers say, so it is accidentally
immune to the failure that forced the probe into the other two. **If only one
scheduler probed, a three-way comparison would partly be measuring which
schedulers keep fresh statistics rather than how well they schedule.**

### Reliability is a quarantine, not a life sentence

`record_dispatch_outcome` is only reachable on the dispatch path. So once
reliability fell below `RELIABILITY_FLOOR` (0.5), `estimate_time` returned `inf`,
the scheduler stopped choosing the node, nothing was dispatched, and no further
outcome was ever recorded — **the score froze below the floor for the life of the
process.** Restarting the master was the only thing that brought a worker back.
Every other exclusion (battery, RAM) is fed by heartbeats that keep arriving
regardless of routing; reliability was the only one whose evidence was gated by
the decision it controlled, which made it permanent by construction.

`record_liveness_tick` now heals it at 0.05/s for any node still heartbeating. A
node written off at 0.35 is back above the floor in three seconds and fully
trusted in thirteen — while a genuinely dead node stays out, because the reaper
re-fails it faster than this heals it.

**Backpressure is not failure.** A worker whose socket high-water mark is reached
is telling you it is busy, not broken, and the two need opposite responses. The
master's `SNDHWM` is 2 against a worker needing ~160 ms a frame, so a burst of
perfectly healthy "please wait" signals arrives in well under a second. At
α=0.1 it takes seven to cross the floor (0.9⁷ = 0.478), so **a moment of
congestion permanently retired a working node.** `record_backpressure` now counts
it as a diagnostic and nothing else; the queue-depth signal already tells the
scheduler the node is loaded, which is the correct and sufficient response.

### The reaper is what makes reliability mean anything

An asynchronous ZMQ PUSH to a dead peer does not raise. Without
`reaper_loop`, the only failures ever recorded are send errors, so **a worker
could stop answering entirely while its score sat at a perfect 1.0.** Any
dispatch with no result after 2 s is written off, the peer's in-flight count is
decremented, and the failure is recorded. Without it a worker that dies mid-frame
leaves its in-flight count permanently high and the scheduler keeps avoiding it
for work it is not doing.

### `enforce_deadline` — the shared gate

Every policy runs its choice through the same function:

```python
enforce_deadline(estimates, preferred_node, remaining_sec)
  → preferred_node if it fits
  → else the fastest node that does fit
  → else "drop"
```

**This exists so that a comparison between schedulers measures scheduling quality
rather than differences in how aggressively each one gives up.** Greedy used to
drop infeasible frames while Lyapunov and round robin delivered them late, which
made their drop rates and latency tails incomparable by construction.

It is also the single most important thing to understand when reading the results,
because under load **it makes more of the routing decisions than the algorithms
do.** In the battery runs round robin's rotation intended 2,500 frames for the
worker; 1,300 arrived, the gate redirected 639 to local and dropped 561. The
"50/50" baseline actually ran at 29.3% — which happens to be near the
capacity-matched optimum. The safety net had quietly done the load balancing round
robin refuses to do, flattering the baseline and hiding what a naive static split
really costs.

**The gate's default differs by policy, and the asymmetry is deliberate:**

| Policy | Gate default | Flag to change it |
|---|---|---|
| `lyapunov`, `greedy` | **on**, always | — |
| `rr` | **on** | `MEC_NO_DEADLINE=1` removes it |
| `fixed` | **off** | `MEC_ENFORCE_DEADLINE=1` puts it back |

`fixed` exists to hold a ratio you chose, and a gate that silently converts "50%"
into something else defeats its entire purpose — so there the ratio *is* the
policy: frames go where the setting says regardless of whether they will make the
deadline, and if they arrive late or die at the worker, that is the measurement.

Either way, **a run with a non-default gate setting belongs beside the same
scheduler with the default, never in a table with the adaptive ones** — because
they give up on infeasible frames and it does not, so its drop rate and latency
tail are not on the same footing.

### The four policies

**Greedy ECT** — compute every candidate's estimated completion time, take the
minimum, with hysteresis (`SWITCH_MARGIN_SEC`, 5 ms) so two nodes within a
millisecond of each other do not trade the stream back and forth every frame.

**Lyapunov drift-plus-penalty** —

```
i* = argmin_i [ Q_i(t)·w_i(t) + (V + Z_i(t))·max(0, e_i(t) - 1) ]

Q_i  virtual backlog at node i, in SECONDS of outstanding work
w_i  marginal work this frame adds  (service + transfer)
e_i  instantaneous thermal penalty, 1.0 … 1.5
Z_i  virtual queue enforcing a TIME-AVERAGE thermal budget
V    tradeoff weight, s²   (DEFAULT_V = 0.02)
```

The structural difference from greedy is that drift-plus-penalty weights backlog
against work as a **product** where completion time takes their **sum**:

```
ECT ranks by  (Q_i + 1) · s_i      ← sum
DPP ranks by   Q_i · w_i + V · e_i ← product
```

Under a sum, a node that is idle but slow still costs its full service time, so a
worker 4× slower than the master only wins once the master is ~5 frames deep — by
which point the frame is nearly dead against the deadline and gets dropped instead
of offloaded. Under a product, an idle node costs `Q·w = 0·w ≈ 0` and is chosen
immediately: **being slow only matters once you also have a backlog.** That is the
entire reason this scheduler offloads under load where an earlier version
collapsed to local and shed the overflow as drops.

`Q_i` is *virtual*, not the physical queue depth, because the physical local queue
is bounded and is drained *by dropping* — under sustained overload it pins at its
ceiling and stops carrying information, venting the pressure to offload as dropped
frames. A virtual queue has no ceiling: it grows while arrivals exceed service, so
offload pressure keeps rising for as long as the overload lasts. That unbounded
growth is what makes the standard O(1/V) penalty / O(V) backlog tradeoff hold. It
is anchored back toward the observed depth each tick (`Q_ANCHOR_BETA = 0.15`) so
an error in `w_i` cannot let it drift into fiction.

Choosing a node charges its own `Q`, which is the self-damping that replaces
hysteresis. A second damping mechanism on top would make the behaviour impossible
to attribute to either.

> **Important caveat for interpreting results.** `thermal_penalty()` returns
> exactly 1.0 below `THERMAL_SAFE_C = 70 °C`, and these boards ran at **33-42 °C
> in every run**. So `max(0, 1.0 - 1.0) = 0`, the entire penalty term multiplies
> out to zero on every frame, and **Lyapunov collapses to `min(Q·w)`** — pure
> backlog balancing. Sweeping `V` changes nothing, because `V` only ever appears
> multiplied by that zero. To make the thermal machinery engage on this hardware
> you would have to lower `THERMAL_SAFE_C` to a temperature the rig actually
> reaches, which is a defensible and documentable change: *"the thermal threshold
> was calibrated to the observed operating range of the testbed, 33-42 °C, rather
> than the 70 °C throttle point."*

**Round robin** — `available[i % len(available)]`, rebuilt fresh each frame so a
worker joining or dying enters and leaves the rotation immediately. Completion
times are computed *only* to test the choice against the deadline, never to
influence it; round robin that peeked at them would not be round robin.

**Fixed ratio** — closed loop on what actually happened, not open loop on a coin
flip. A random draw at p=0.25 over 5000 frames lands within ±1.2% of target and
drifts differently every run, so two runs at "25%" are not the same experiment.
This counts the decisions it has made and sends each frame to whichever side pulls
the realised ratio back toward target, landing within one frame of the setting
every time. It also counts **the decision, not the intention**: when the deadline
pushes an offload back to local, the counters record the override and later frames
lean the other way.

Know what that means when the link is unstable: catch-up is not gentle. If the
worker is unreachable for eight seconds, every frame in that window goes local,
and once it returns the scheduler offloads at 100% until the ratio is square. The
setting is still exactly honoured across the run — which is the property that makes
runs comparable — but the run is a dip and a spike that average to target, not a
steady ratio.

Its purpose: the other three each decide *for themselves* how much to send away,
so the offload percentage in those results is an outcome, not a setting. That
makes one question unanswerable from them alone — how much of the latency
difference comes from the *policy*, and how much simply from the fact that the
policies landed on different ratios? Run this at the ratio Lyapunov chose on its
own and any remaining gap is what Lyapunov's *timing* bought you.

### `STRICT_OFFLOAD`

Two things downstream of `pick()` can turn an offload into a local run without the
scheduler hearing about it: a full outgoing queue, and a send that fails because
the worker's socket is blocked. Both are correct for operating a live system — a
delivered frame beats a lost one — and both are wrong for a scheduler whose whole
job is to hold a ratio, because the ratio it reports is then not the ratio that
ran.

On by default for `fixed`, off for the others, and forced on by
`MEC_NO_DEADLINE=1`.

---

## 7. The worker

Holds no state the swarm depends on and makes no routing decisions, which is what
makes workers interchangeable: the master chooses between them purely on measured
behaviour, and adding a fourth Nano needs no change anywhere.

Its inbound socket has **`RCVHWM = 2`** on purpose. A deep buffer would let a
backlog build that the master cannot see, so the master's estimate of this node
would stay optimistic while the real wait grew. Shallow means congestion shows up
as rising RTT almost immediately — a signal the scheduler acts on.

It **always replies, even on failure.** Silence would leave the master counting
the frame as in flight until the reaper times it out, and would cost this node
reliability for a frame it did answer for.

---

## 8. The ground station

Two builds, identical layout and wire format, differing only in playback policy so
a recording of one can be held against a recording of the other with no allowance
for presentation.

| | `gcs.py` (DIRECT) | `gcs_buffered.py` (BUFFERED) |
|---|---|---|
| Playback | every frame on arrival, arrival order | per-sensor buffer, capture order |
| Reordering | **counted, never fixed** | fixed |
| Handoff queue | 12, **oldest discarded** when full | 500 |
| Unique metric | `Gaps` — ids that never arrived at all | `Skipped` — ids playback stepped over |
| Cost | none | the buffer wait, measured and included in the displayed latency |

The direct build's handoff queue is shallow because it exists only because tkinter
is not thread-safe, **not** to smooth the feed. When it fills the *oldest* frame is
discarded: a backlog on a display with no jitter buffer is a contradiction, since
it would mean showing old frames while newer ones sit behind them. Blocking instead
would back-pressure all the way up to the master.

### A trap in reading its event log

`ORDER_QUIET_SEC = 10` rate-limits `sequence gap` and `out of order` rows to **one
per sensor per kind per 10 seconds.** Three 168-second runs each logged exactly
**17** of each, 10.0 s apart to the second — 168 ÷ 10 = 16.8 → 17. That is the
limiter saturated for the whole run, and it reads as "all three schedulers behaved
identically". Recomputed from the master's own completion order, the real figures
were **69.1% / 61.9% / 57.2%** of delivered frames out of sequence.

**Never quote those counts without checking whether they equal `run_seconds / 10`.**
`frame dropped` rows are not rate-limited and do match the master's counts exactly.

### Nine event types, all mirrored to CSV

`session started` · `sensor appeared` · `worker joined` / `rejoined` / `left` ·
`frame dropped` (with the master's reason and queue depth) · `sequence gap` ·
`out of order` · `feed lost` · `feed resumed` · `clock disagreement` ·
`corrupt frame` · `render failed` · `session ended`

`feed lost` → `feed resumed` **is** the failover recovery measurement, on the one
clock that never moved.

---

## 9. The sensor

`[sensor_id, frame_id, capture_ts, jpeg_bytes, stamps_json]` — five parts, the
fifth optional so an older four-part sensor still works.

**`capture_ts` is stamped at the moment the frame is handed to ZMQ**, and it keeps
that exact meaning permanently, because every latency figure in the system and
every earlier baseline measures from it. The trace stamps ride in the fifth part
where the master may ignore them.

**The frame id advances only on frames actually handed to ZMQ.** That is what lets
`frame_trace.py coverage()` account for loss: within a traced span, any id absent
from the file is a frame that was sent and never reached the GCS. Counting the rows
that arrived tells you nothing about that.

**Strict send is on by default**, and this is a measurement decision. The sender
used to be `NOBLOCK`: when the master fell behind, the frame was discarded and the
id was *not* advanced, so the loss left no gap in the sequence and nothing
downstream could see it had happened. That is right for a live feed and wrong for a
measurement, because it silently converts "the master is saturated" into "the Pi
sent fewer frames". Strict mode blocks up to `SEND_TIMEOUT_MS` (1 s), so back
pressure shows up honestly as a lower achieved rate.

`ZMQ_IMMEDIATE` is set, so frames captured during an outage are not queued for a
master that is not there — they would arrive seconds stale, be dropped by the
deadline check anyway, and displace fresh frames on the way.

### JPEG q80, not q95

| | payload | at 30 fps |
|---|---:|---:|
| q75 | 25 kB | 6.2 Mbps |
| **q80** | **31 kB** | **7.7 Mbps** |
| q95 | 107 kB | 26.4 Mbps |

The last fifteen points of quality cost 3.4× the bytes. YOLOv5n letterboxes to
640×640 and runs at conf 0.60, well above the threshold where q80 artefacts change
a detection, so those bytes buy nothing the model can use — they only compete with
the master's offload traffic and the GCS feed on the same radio.

---

## 10. Failover, end to end

```
 t=0      master (Nano 2) loses power
 t≈1-2 s  Nano 1's wlan0 disassociates. swarm_net notices within
          ASSOCIATION_FAIL_THRESHOLD (3 s) of polling.
          The Pis' ZMTP heartbeats (1 s ivl, 3 s timeout) fail; their PUSH
          sockets begin reconnecting at a fixed 200 ms interval.
 t≈5 s    Nano 1 rescans. RECLAIM_DELAY_SEC, not the 30 s boot delay.
 t≈8 s    Nano 1 activates NanoNet-ap, takes 192.168.50.1.
 t≈10 s   mec_node.py on Nano 1 polls neighbors.json, sees role=master,
          tears down its WorkerRole and starts a MasterRole —
          WITH THE MODEL STILL LOADED, so this costs a second, not 30.
 t≈10-15s The Pis reassociate (autoconnect, priority 100, infinite retries)
          and their sockets connect to the same address as before.
          nanonet-link.py forces it if NetworkManager does not.
 t≈15 s   GCS logs "feed resumed".
```

Four design choices make that work, and each replaces something slower:

- **ZMTP heartbeats instead of TCP timeouts.** At ZMQ's defaults heartbeats are
  off, so a dead peer is only noticed when TCP says so — and Linux retries an
  established connection 15 times with doubling backoff (0.2 s … 25.6 s). If the
  new AP comes up at t=10 s, the next retransmit might not fire until t=25 s.
- **Fixed reconnect interval, no backoff.** An outage lasts as long as the other
  Nano needs to rebuild the network, and a backed-off socket would be waiting
  seconds between attempts by the end of it — adding its own delay on top of the
  real recovery.
- **The model is loaded once, before any role is taken.** The single biggest lever
  on failover time: a cold TensorRT load is 10-30 s on a Nano and would otherwise
  dominate every role change.
- **A worker that moved has its bookkeeping reset.** Its in-flight count, its
  reliability EWMA and its warm-up sample count all clear, because reliability is
  about a node's recent conduct and a node that just restarted has no recent
  conduct. Carrying the old score across meant a worker could come back healthy
  and still be excluded on the strength of failures its previous process
  committed.
