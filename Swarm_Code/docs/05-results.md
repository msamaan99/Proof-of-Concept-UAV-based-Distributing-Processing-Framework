# Results

What this rig actually measured. Every number here is recomputed from the raw
per-frame CSVs the master writes, with drops excluded from averages and reported
separately.

**Status of these numbers:** each configuration is **one run of 5000 frames**, not
a repeated trial. Every figure is a point estimate, and differences smaller than a
few percentage points cannot be distinguished from run-to-run noise by this
dataset. That is stated here rather than buried, because it bounds every claim
below.

---

## The testbed

| | |
|---|---|
| Compute | 2 × Jetson Nano 4 GB, YOLOv5n FP16 TensorRT, 640×640 |
| Sensing | 2 × Raspberry Pi, 640×480, JPEG q80 (31 kB) |
| Network | self-formed single-hop 2.4 GHz 802.11g star |
| Time | chrony, Nano 2 as reference, ~15 µs residual |
| Deadline | **D = 1200 ms** |
| Combined capacity | **37.2 fps** measured (1/s_l + 1/s_w) |
| Frames per run | 5000 per policy |

**This is a ground-based hardware-in-the-loop testbed, not a flight test.** The
Pi/Jetson pairing emulates the sensor-and-compute split of an aerial platform
without requiring the platform. Nothing here should be read as an airborne result.

**D = 1200 ms is a hardware ceiling on this PoC, not a mission requirement.** It
was 600 ms, calibrated for a laptop master; with a Jetson master the worker fails
its feasibility check on every comparison at 600 ms. 1200 ms is the smallest
deadline both boards can serve. It suits the surveillance and mapping missions this
targets, where a roughly one-second-old detection is still useful. **It is not a
number to reuse for collision avoidance or closed-loop tracking.**

### The four load points

ρ is the offered rate over the combined service capacity.

| | Sensors | Offered | ρ | Character |
|---|---|---|---|---|
| **LP1** | 1 | 15.0-15.9 fps | 0.43 | headroom |
| **LP2** | 1 | 29.9 fps | 0.80 | master saturated |
| **LP3** | 1 | 29.8 fps | — | heterogeneous (worker battery-derated, k ≈ 1.95) |
| **LP4** | 2 | 39.7-40.0 fps | 1.07 | overload |

---

## The headline finding: drop rate misranks schedulers under overload

This is the most transferable result in the project, and it is methodological.

The admission gate drops a frame whose **predicted** completion exceeds the
remaining budget. It does not guarantee that an *admitted* frame completes within
it: the prediction can be wrong, and the queue can grow after the decision. **A
frame delivered at 2112 ms against a 1200 ms deadline is counted as delivered by
the drop metric and is useless by the mission definition that set the deadline.**

So define **useful yield** as the fraction of *offered* frames that are both
delivered *and* delivered within D. It cannot be computed exactly from summary
statistics, but it can be bounded distribution-free: where P95 exceeds D, at most
95% of delivered frames met it; where the *mean* exceeds D, Cantelli's inequality
gives the sharper `Pr[X ≤ X̄ − t] ≤ σ²/(σ² + t²)` with `t = X̄ − D`.

### LP4 — the two metrics disagree completely

| Policy | Drop % | ≤ within-D of delivered | **≤ useful yield** |
|---|---:|---:|---:|
| Lyapunov | 8.20 | 95.0 % | **87.2 %** |
| Greedy (ECT) | **7.50** | 95.0 % | **87.9 %** |
| Round robin | 8.20 | **50.8 %** | **46.6 %** |
| Fixed 35 % | 15.80 | 95.0 % | 80.0 % |
| Fixed 65 % | 21.22 | 95.0 % | 74.8 % |

**By drop rate, round robin (8.20%) is indistinguishable from Lyapunov (8.20%) and
close to greedy (7.50%). By useful yield it delivers at most 46.6% of offered
frames on time, against at least 87% for the adaptive pair.** The bound is driven
by round robin's mean *delivered* latency of 1618 ms, which exceeds the deadline
itself.

**And mean latency misranks them too, in the opposite direction.** Fixed 65% has
the *lowest* mean latency of any policy at LP4 — 848 ms — and the second-worst
useful yield. Its standard deviation is 710 ms against Lyapunov's 114 ms: the low
mean is a bimodal artefact of discarding a fifth of the frames.

> **Neither mean latency nor drop rate can rank schedulers under loss.** Report
> both, plus the spread, plus a yield bound.

---

## LP4 — two sensors, system in overload (ρ = 1.07)

39.7 fps aggregate against 37.2 fps of capacity. The fixed ratios were chosen to
bracket the adaptive operating point of ~47%.

| Metric | Lyapunov | Greedy (ECT) | Round Robin | Fixed 35 % | Fixed 65 % |
|---|---:|---:|---:|---:|---:|
| Delivered / Dropped | 4590 / 410 | **4625 / 375** | 4590 / 410 | 4210 / 790 | 3939 / 1061 |
| Drop % | 8.20 | **7.50** | 8.20 | 15.80 | 21.22 |
| **≤ Useful yield %** | **87.2** | **87.9** | **46.6** | 80.0 | 74.8 |
| Main drop reason | deadline | deadline | link/timeout | local queue full | worker link blocked |
| Mean latency (ms) | 1161.48 | 1141.31 | 1617.73 | 1045.77 | *848.24* |
| P95 (ms) | 1248.90 | **1236.70** | 2112.40 | 1804.56 | 1527.91 |
| Std deviation (ms) | **114.35** | 138.28 | 424.60 | 835.54 | 709.59 |
| Offload % | 47.30 | 47.14 | 50.68 | 35.24 | 64.84 |
| Master / worker duty % | 99 / 99 | 100 / 99 | 99 / 99 | 99 / 80 | 71 / 99 |

Note the drop *reasons* differ, and they say where each policy broke:

- adaptive pair → `DEADLINE_EXCEEDED`: the gate rejecting frames up front
- round robin → **link/timeout**: frames sent to a worker that could not hold them
- fixed 35% → `LOCAL_QUEUE_FULL`: not offloading enough, master saturated
- fixed 65% → `WORKER_LINK_BLOCKED`: offloading more than the worker can absorb

---

## LP2 — one sensor, master saturated (ρ = 0.80)

**All three policies deliver every frame with zero drops**, so the comparison turns
entirely on the latency *distribution* — and it is a genuine trade, not a win.

| Metric | Lyapunov | Greedy | Round Robin |
|---|---:|---:|---:|
| Mean latency (ms) | 76.83 | 84.83 | **72.52** |
| P95 (ms) | 93.30 | 111.30 | **85.70** |
| P99 (ms) | 198.21 | 158.92 | **150.91** |
| **Max (ms)** | **245.7** | 635.2 | 1050.3 |
| **Frames > 250 ms** | **0** | 21 | 14 |
| Std deviation (ms) | **20.39** | 27.24 | 25.56 |
| Offload % | 42.20 | 37.52 | 50.06 |
| Master GPU duty % | 92 | 99 | **77** |
| Worker GPU duty % | 75 | 67 | 88 |
| Decision overhead (ms) | 0.183 | 0.098 | **0.076** |

Round robin is 4.3 ms faster on the mean and best at P95 and P99. But **not one
Lyapunov frame in 5000 exceeds 250 ms**, against 21 for greedy and 14 for round
robin — whose worst frame reached 1050 ms, 87% of the deadline.

Lyapunov carries a fatter shoulder of mildly-late frames and **eliminates the
extremes**. Which of those you want is a mission question, not a scheduling one.

Lyapunov also ran both boards coolest. Round robin's blind 50/50 split drove the
worker to 88% duty and 42 °C, the hottest reading in the whole dataset.

---

## LP1 — master has headroom (ρ = 0.43)

Every policy delivers every frame.

| Metric | Lyapunov | Greedy | Round Robin |
|---|---:|---:|---:|
| Offered rate (fps) | 15.0 | 15.9 | 15.9 |
| Mean latency (ms) | **73.06** | 78.81 | 88.78 |
| P95 (ms) | 101.60 | **95.62** | 127.62 |
| P99 (ms) | **199.20** | 326.05 | 247.61 |
| Max (ms) | **745.4** | 1124.6 | 763.1 |
| Std deviation (ms) | **33.39** | 59.01 | 48.67 |
| Offload % | 12.94 | 7.86 | 50.00 |
| Decision overhead (ms) | 0.168 | 0.093 | **0.070** |

> **These runs were not rate-matched** — Lyapunov was offered 15.0 fps against
> 15.9 for the other two — so latency differences between the columns are **not**
> attributable to the decision rule alone. LP1 is reported as evidence that the
> policies behave sensibly under headroom, not as a ranking.

**Greedy offloading almost nothing (7.86%) is the correct decision**, not a
failure. Sending a frame the master could finish in 50 ms to a node 58 ms away
plus a link is a loss. Lyapunov's own fraction is low for the same reason: with no
queue pressure, `Q_i·w_i` rarely favours the worker.

---

## LP3 — heterogeneous nodes (worker battery-derated, k ≈ 1.95)

| Metric | Lyapunov | Greedy | RR |
|---|---:|---:|---:|
| Drop % | **1.60** | 2.98 | 26.36 |
| Main drop reason | deadline | deadline | link/timeout |
| Mean latency (ms) | 1109.80 | 1135.87 | *1006.89* |
| P95 (ms) | 1248.50 | **1236.10** | 2638.60 |
| Max (ms) | 1436.9 | **1351.4** | 2925.3 |
| Avg network RTT (ms) | **4.11** | 4.81 | 852.82 |
| Worker/master ratio k | 1.95 | 1.93 | 1.94 |
| Offload % (delivered) | 34.23 | 34.57 | 55.90 |
| Out-of-order % | 65.30 | 55.89 | 40.06 |

> **Two qualifications, both material.** Battery derating changes the worker's
> clock, voltage and thermal behaviour together, so k is not a clean single
> variable. And **the round-robin run here is contaminated** by a logged network
> fault — an 852 ms average RTT is not a scheduling result — so it is excluded
> from the analysis. A draining battery is also a moving target: across three runs
> on this rig the worker drifted **37%**, meaning whichever policy ran last faced
> the fastest worker. Bench the worker before *and* after such a set.

### Out-of-order delivery is a first-class finding, not a footnote

**Between 40% and 65% of delivered frames arrive at the GCS out of capture
order at this load point.** For surveillance and mapping, a detection stream that
is majority-reordered requires either a reordering buffer — which adds latency to
the very budget the scheduler is defending — or a downstream consumer that
tolerates reordering.

Per-frame routing across two nodes with different service times makes reordering
**structural, not incidental**: the fast path overtakes the slow one whenever both
are busy.

---

## Per-frame stage decomposition

One continuous 2488-frame capture (1341 local, 1147 offloaded), median with IQR in
ms. A dash marks a stage that does not exist on that path.

| Stage | Local path | Offload path |
|---|---:|---:|
| Capture (shutter → frame in hand) | 3.61 (3.28–4.00) | 3.59 (3.29–3.93) |
| Conversion (JPEG) | 7.41 (6.60–8.45) | 7.43 (6.51–8.48) |
| Pi handoff to ZMQ | 0.00 | 0.00 |
| Network Pi → master | 7.21 (6.69–8.62) | 7.52 (6.93–8.77) |
| Wait for scheduler | 0.49 (0.46–0.56) | 1.02 (0.79–1.22) |
| **Decision** | **0.17 (0.17–0.18)** | **0.32 (0.21–0.40)** |
| Decode queue + decode (master) | 5.15 (5.08–5.35) | — |
| GPU queue + inference (master) | 97.99 (94.51–110.41) | — |
| Dispatch queue | — | 0.26 (0.21–0.32) |
| Network master → worker | — | 6.10 (4.82–7.85) |
| Worker decode + inference | — | 56.90 (56.72–57.64) |
| **Network worker → master** | — | **1.70 (1.36–3.90)** |
| Forward to GCS | 0.44 (0.40–0.48) | 0.51 (0.44–0.54) |
| Network master → GCS | 6.81 (5.43–8.53) | 5.74 (5.09–7.12) |
| **Total, capture → GCS** | **132.92 (126.26–144.83)** | **93.96 (89.10–103.02)** |

Three things this shows:

**1. The return hop is an order of magnitude cheaper than the outbound.** 6.10 ms
out against 1.70 ms back, because the return carries a few hundred bytes of JSON
detections rather than a ~31 kB JPEG. **This is the proof-of-concept insight of
the whole architecture** — offloading inference is cheap precisely because the
answer is small — measured over a thousand offloaded frames.

**2. Scheduling is free at this scale.** The decision costs a median of 0.17 ms
against a median total of 132.9 ms — about **0.13%** — and 0.07-0.19 ms across
every policy and load point, against roughly 50 ms of inference. Three orders of
magnitude cheaper than the work it routes, which is a budget a trained policy or a
per-epoch solve cannot meet on this hardware.

**3. Board asymmetry dominates path choice.** The master's inference median here
is 97.99 ms and the worker's 56.90 ms — two nominally identical Jetson Nanos
differing by **1.72×** under identical configuration — so the offloaded path is
*faster* end to end.

> **Board roles in this capture are reversed relative to the LP tables**, so its
> absolute numbers are not comparable with them. It is included because it
> characterises the pipeline shape and the hardware asymmetry any deployment will
> meet.
>
> Note also that **`capture` here is 3.6 ms, not the 40-47 ms seen in some
> single-frame traces.** When the sensor loop outruns the camera, the capture stamp
> is mostly the loop *blocking for the next frame* — time during which the frame
> does not yet exist — so charging it as latency overstates the figure. Quote
> **`frame ready → GCS`**, which the reader prints separately.

---

## The capacity-matching operating point

At LP4 two policies with entirely different decision rules converge on the same
offload fraction: **Lyapunov 47.30%, greedy 47.14%.** That is not a coincidence.

**Greedy.** Taking `argmin_i T_i` every frame drives the system toward
`T_local = T_worker`. Queues stay bounded only when each node's arrival rate equals
its service rate, `λ(1−p) = 1/s_l` and `λp = 1/s_w`, whence

```
p* = s_l / (s_l + s_w) = 1 / (1 + k),        k = s_w / s_l
```

**Lyapunov.** Selecting node i adds `w_i` to `Q_i`, which drains by `Δt` per tick,
so `dQ_i/dt = λ_i·w_i − 1`, steady state `λ_i* = 1/w_i`, giving
`p* = w_l / (w_l + w_w)`. Since `w ≈ s`, **the same fixed point is reached by a
different route: greedy equalises cost, Lyapunov equalises queue growth.**

### The precondition, which is routinely omitted when this fraction is quoted

`p*` requires **both** flow-balance equations to hold simultaneously — that is
`λ = 1/s_l + 1/s_w`, exactly the critically loaded condition **ρ = 1**. Below
saturation the system has slack, neither equation binds, and an adaptive policy has
no reason to seek `p*`; it concentrates work on whichever node is cheapest instead.

Every Lyapunov and greedy run in the study, not only the saturated ones:

| Run | ρ | s_l (ms) | s_w (ms) | k | p* | measured offload |
|---|---:|---:|---:|---:|---:|---:|
| LP4 Lyapunov | 1.07 | 50.70 | 57.20 | 1.13 | 46.99 % | **47.30 %** |
| LP4 Greedy | 1.07 | 50.70 | 57.00 | 1.12 | 47.08 % | **47.14 %** |
| LP3 Lyapunov | ≈1 | 50.70 | 98.70 | 1.95 | 33.94 % | **34.23 %** |
| LP3 Greedy | ≈1 | 51.25 | 98.90 | 1.93 | 34.13 % | **34.57 %** |
| LP2 Lyapunov | 0.80 | 50.70 | 57.20 | 1.13 | 46.99 % | 42.20 % |
| LP2 Greedy | 0.80 | 50.70 | 57.00 | 1.12 | 47.08 % | 37.52 % |
| LP1 Lyapunov | 0.43 | 50.70 | 57.20 | 1.13 | 46.99 % | 12.94 % |
| LP1 Greedy | 0.43 | 50.70 | 57.00 | 1.12 | 47.08 % | 7.86 % |

At ρ ≈ 1 both policies land within **0.4 percentage points** of `p*`, at two
different values of k. Below saturation they diverge from it by up to 39 points,
exactly as the derivation predicts. **The precondition is not a technicality.**

### A trap that runs in both directions

`p*` is zero only when k = 1. On the near-symmetric pair at LP4 (k ≈ 1.13) the
error from assuming 50/50 is small, which makes it easy to believe a static split
has converged when it has not:

- At **LP2**, round robin's structural 50.06% sits near p* = 46.99% by
  coincidence of the hardware, not by adaptation.
- At **LP3** (k ≈ 1.95, p* ≈ 34%), the same static 50% is 16 points wrong and its
  drop rate goes to 26%.

---

## Threats to validity

Stated explicitly rather than left for a reader to find.

**Not shown here, and closing any of them requires new runs on the rig:**

- **No statistical replication.** One 5000-frame run per (policy, load point), no
  confidence intervals.
- **No detection-accuracy (mAP) evaluation.** JPEG q80 was chosen over q95 on
  bandwidth grounds; the claim that it does not change detections at conf 0.60 is
  reasoned, not measured.
- **No direct energy measurement.** Temperature and duty cycle are proxies.
- **No join-the-shortest-queue baseline.**
- **One failover drill**, not a distribution of recovery times.
- **Not rate-matched at LP1**, as noted above.
- **LP3 round robin is contaminated** by a logged network fault.
- **Vibration, flight dynamics and the multi-worker generalisation of p* are
  untested.** This is a bench rig.
- **Security is WPA2 only**, with no join authentication: any device with the PSK
  can become a worker.

**Known artefacts of the implementation, documented rather than hidden:**

- **Lyapunov's thermal term is inert on this hardware.** `thermal_penalty()`
  returns exactly 1.0 below 70 °C and the boards ran at 33-42 °C, so the entire
  `(V + Z)·(e − 1)` term is zero on every frame and Lyapunov reduces to
  `min(Q·w)`. **Sweeping V therefore changes nothing** — V only ever appears
  multiplied by that zero. Engaging it would mean lowering `THERMAL_SAFE_C` to a
  temperature the rig reaches, which is a defensible change but a change.
- **The deadline gate makes more routing decisions than the algorithms do under
  overload**, which is why the three policies converge to within 18 ms of each
  other at the most stressed load point. That is a property of the operating
  point, not of the schedulers.
- **The two Jetson Nanos are not identical.** Measured 50.9 ms vs 95.5 ms under
  identical power mode, pinned clocks, TensorRT version and FP16 bindings; traced
  to cuDNN tactic availability differing between L4T R32.7.1 and R32.7.6. Later
  runs showed both at ~51 ms, which contradicts that and **has not been
  re-benched** — so bench both boards before quoting either figure.

---

## Reproducing any of this

```bash
./scripts/preflight.sh && ./scripts/preflight.sh soak     # the gate
./scripts/preflight.sh roles                              # record which board is master

# one run per policy, same frame count
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && \
  MEC_SCHED=lyapunov MEC_RUN_FRAMES=5000 python3 mec_node.py'
# ... repeat for greedy, rr, fixed, with ~5 min cooldown between

# collect and tabulate
P=LP2; mkdir -p ~/uav_project/$P
scp 'admindesktop1@10.42.0.226:~/yolov5/swarm/runtime/*.csv' ~/uav_project/$P/
python3 tools/algo_report.py ~/uav_project/$P -o ~/uav_project/tables_$P
```

Raw per-frame CSVs for the runs above are not in this repository — they are ~200 kB
each and there are dozens. The derived tables are in
[`../results/tables/`](../results/tables/), and a sample per-frame trace is in
[`../results/traces/`](../results/traces/).

Full write-up, with the derivations and the literature positioning, is in the
project report and research paper (LaTeX sources kept outside this repository).
