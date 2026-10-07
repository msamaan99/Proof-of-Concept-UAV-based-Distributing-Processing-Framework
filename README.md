# UAV Swarm MEC

**A distributed edge-computing framework for UAV swarms, built and measured on real
hardware.**

Two Jetson Nanos form their own Wi-Fi network with no infrastructure, no configured
leader and no single point of failure. Raspberry Pi camera nodes stream frames into
it. For every individual frame, the board holding the network decides whether to run
YOLOv5n object detection itself or ship the frame to its neighbour — and four
different scheduling policies are implemented, instrumented and compared for making
that decision.

Kill the board running the network and the other one rebuilds it, takes over the
master's address, and keeps serving. The cameras never find out.

Final Year Design Project, College of Aeronautical Engineering, PAF Academy Asghar
Khan, Risalpur.

---

## The problem

A conventional UAV is an airborne data collector. It senses, and it streams raw data
to a ground station or a cloud service that does the actual thinking. That
architecture has five consequences, and all of them get worse as the swarm gets
bigger:

- **A single point of failure.** Sever the downlink and the aircraft is useless.
- **Transmission latency** that scales with distance to the processing site, not with
  the difficulty of the task.
- **Network jitter** on a link that was never designed for a real-time control loop.
- **A continuous, high-bandwidth requirement** — raw video, all of it, all the time.
- **An electromagnetic signature**, which in a contested environment is a targeting
  aid.

The fix is conceptually simple: process the data where it is captured. The
difficulty is that a single low-power edge board cannot always keep up, and the
moment you have more than one board the question becomes *which* board — and that
question has to be answered per frame, in less time than the frame is worth.

## The aim

> To develop a distributed edge computing framework for resource-constrained aerial
> platforms that intelligently distributes computational workloads across a
> cooperative swarm network, enabling real-time data processing without reliance on
> centralised cloud or ground infrastructure.

## The gap this addresses

Task offloading for UAV swarms is a well-populated literature. Heuristics (round
robin, greedy) are cheap but queue-agnostic. Lyapunov optimisation offers provable
queue stability at O(1) overhead. Evolutionary methods search well but converge
slowly. Deep reinforcement learning handles complex environments but needs hardware
accelerators the aircraft does not have.

**Almost all of it is evaluated in simulation.** Simulators do not reproduce
TensorRT warm-up transients, Wi-Fi power-save parking the radio between beacons,
clock daemons stepping a board's time mid-run, two nominally identical boards
differing by 2×, or a fixed-ratio scheduler quietly becoming a different ratio
because an admission gate overruled it. Every one of those materially changed a
result on this rig, and each is documented in `docs/06-troubleshooting.md`.

So the contribution here is not a new algorithm. It is **a physical proof of concept
on COTS hardware, and an honest measurement of what happens when textbook policies
meet a real link** — including the finding that the metric the literature most often
reports, drop rate, *misranks* the schedulers under overload.

---

## What was built

```
  TIER 1 — sensing              TIER 2 — edge compute              TIER 3 — display
  ────────────────              ─────────────────────              ────────────────

  Raspberry Pi 1                Jetson Nano  (MASTER)              Laptop GCS
  camera 640×480         ┌────► · owns the Wi-Fi AP at .1    ┌───► · dual-pane feed
  JPEG q80, 15–20 fps    │      · ingests every frame        │     · event log
                    :5000│      · decides: here or there?    │     · recovery timing
  Raspberry Pi 2         │      · runs YOLOv5n on its GPU    │
  same, sensor id pi2 ───┘      · forwards results ──────────┘ :6000
                                        │
                                  :5001 │ offload          :5003
                                        ▼                    ▲
                                Jetson Nano  (WORKER)        │
                                · YOLOv5n, returns ──────────┘
                                  detections as JSON
```

Three tiers, and the split is deliberate at each boundary:

| Tier | Hardware | Does | Deliberately does **not** |
|---|---|---|---|
| 1 | Raspberry Pi | capture, compress, stream | run inference — a Pi is slower than the round trip to a Jetson, which would make offloading vacuous by construction |
| 2 | Jetson Nano ×2 | form the network, schedule, infer | render anything |
| 3 | Laptop | display, log, be the clock of record | decide anything |

### Two ideas carry the design

**1. The role is claimed, not configured.** Whoever holds `192.168.50.1` *is* the
master. There is no election, no virtual IP, no config flag. A board that reboots
after a failure runs the same rule — scan, join if you find a network, build one if
you do not — finds the network its partner built, and joins as a worker. **It cannot
resume a role it no longer holds, because the role lives in the network rather than
in the code.** The cameras target the gateway address and never learn that anything
happened.

**2. Scheduling is a per-frame decision priced in time.** For each frame the master
estimates completion time for itself and for every worker, including the network
round trip, the worker's current backlog, its measured volatility and its recent
reliability. If nothing can finish the frame before its deadline, the frame is
dropped rather than spending compute on a result nobody will see in time.

### The four policies under comparison

Selected by one environment variable, never by editing a file:

| `MEC_SCHED` | Policy | Decision rule | Role |
|---|---|---|---|
| `lyapunov` | Drift-plus-penalty | `argmin Q_i·w_i + (V+Z_i)·(e_i−1)` | queue stability traded against thermal cost |
| `greedy` | Earliest completion time | `argmin` estimated completion, with hysteresis | latency-minimising, myopic |
| `rr` | Round robin | `i mod n` | the naive baseline |
| `fixed` | Fixed ratio | closed loop on a ratio you set | the **controlled** baseline |

All four share one cost model, one deadline gate and one exploration probe, so a
comparison between them isolates the decision rule rather than measuring which
implementation happens to keep fresher statistics. The fixed-ratio policy exists
because the other three each choose their own offload percentage, which makes one
question unanswerable from those runs alone: *how much of the latency difference
comes from the policy, and how much simply from the fact that the policies landed on
different ratios?*

---

## Headline results

2 × Jetson Nano + 2 × Raspberry Pi, 5000 frames per policy, deadline 1200 ms, four
load points. Full tables and caveats in `docs/05-results.md`.

### Drop rate misranks schedulers under overload

The most transferable finding, and it is methodological. The admission gate drops a
frame whose *predicted* completion exceeds the remaining budget; it does not
guarantee an *admitted* frame completes within it. **A frame delivered at 2112 ms
against a 1200 ms deadline counts as delivered and is useless.**

At ρ = 1.07 (two sensors, 40 fps):

| Policy | Drop % | ≤ useful yield |
|---|---:|---:|
| Lyapunov | 8.20 | **87.2 %** |
| Greedy (ECT) | **7.50** | **87.9 %** |
| Round robin | 8.20 | **46.6 %** |
| Fixed 35 % | 15.80 | 80.0 % |
| Fixed 65 % | 21.22 | 74.8 % |

By drop rate, round robin is indistinguishable from Lyapunov. By useful yield —
delivered *and* within deadline — it manages at most 46.6% against 87% for the
adaptive pair, because its mean delivered latency (1618 ms) exceeds the deadline
itself. **Mean latency misranks them too, in the opposite direction:** Fixed 65% has
the *lowest* mean of any policy (848 ms) and the second-worst yield, because that
low mean is a bimodal artefact of discarding a fifth of the frames.

### Two different rules converge on the same operating point

At saturation, Lyapunov offloads **47.30%** and greedy **47.14%** — within 0.4
percentage points of the capacity-matched fixed point `p* = 1/(1+k)`, reached by two
different routes: greedy equalises cost, Lyapunov equalises queue growth.

**Below saturation that fixed point does not bind**, and they diverge from it by up
to 39 points. That precondition — `ρ = 1` — is routinely omitted when `p*` is quoted
in the offloading literature, and the measurements show it is not a technicality.

### The trade the latency mean hides

At ρ = 0.80 all three policies deliver every frame, so the comparison is entirely
about the distribution. Round robin wins the mean by 4.3 ms. But **not one Lyapunov
frame in 5000 exceeds 250 ms**, against 21 for greedy and 14 for round robin — whose
worst frame reached 1050 ms, 87% of the deadline. Lyapunov accepts more moderate
lateness to eliminate the extremes. Which you want is a mission question.

### Scheduling is effectively free

The decision costs a **median of 0.17 ms** against ~50 ms of inference — 0.13% of a
frame, three orders of magnitude cheaper than the work it routes. That is the budget
a trained policy or a per-epoch solve cannot meet on this hardware, and it is the
practical argument for the O(1) family over DRL here.

### Offloading is cheap because the answer is small

Median master→worker transfer **6.10 ms**; worker→master return **1.70 ms**. The
outbound carries a ~31 kB JPEG, the return a few hundred bytes of JSON detections.
This is the proof-of-concept insight of the whole architecture, measured over a
thousand offloaded frames.

---

## Repository layout

Directories mirror where the code runs. `deploy.sh` copies each to the right place.

```
nano/          → ~/yolov5/swarm/ on both Jetsons
  mec_node.py                    processing layer: master and worker halves
  swarm_net.py                   network layer: forms/heals NanoNet, telemetry
  completion_time_scheduler.py   shared cost model + greedy ECT + exploration probe
  scheduler_lyapunov.py          drift-plus-penalty
  scheduler_rr.py                round robin baseline
  scheduler_fixed.py             fixed-ratio controlled baseline
  bench_infer.py                 measure one board off-pipeline
  engine_info.py                 what is actually inside a TensorRT engine
  systemd/                       swarm_net.service, mec_node.service

pi/            → ~/pi_sensor/ on both Raspberry Pis
  pi_sensor.py                   capture, JPEG, stream, progress watchdog
  nanonet_link.py                root link supervisor — keeps wlan0 on the swarm
  systemd/                       pi-sensor.service, nanonet-link.service

gcs/           → stays on the laptop
  gcs.py                         ground station, DIRECT playback
  gcs_buffered.py                ground station, BUFFERED playback
  gcs_common.py                  everything the two builds must agree on

shared/
  frame_trace.py                 per-frame stage timing, across all four machines

scripts/
  hosts.env                      THE rig inventory — the only file you edit
  deploy.sh                      md5-verified deploy to any subset of the rig
  preflight.sh                   every pre-run gate, one command
  chrony_setup.sh                time sync, one argument per device role
  pi1_setup.sh / pi2_setup.sh    Raspberry Pi provisioning, staged

tools/algo_report.py             raw per-frame CSVs → comparison tables
tests/test_scheduler_fixes.py    17 regression tests, one per historical defect
docs/                            see below
results/                         derived comparison tables + a sample trace
```

`shared/frame_trace.py` is imported by both the Jetsons and the ground station, so
`nano/frame_trace.py` and `gcs/frame_trace.py` are **symlinks** to it — one source of
truth, and `scp` follows them so each device receives a plain file.

---

## Documentation

| | |
|---|---|
| **[02 — Build From Scratch](docs/02-build-from-scratch.md)** | **Start here.** Bare boards to a measured result in 14 parts. Every command written out, each part ending in a check. ~5 hours. |
| [01 — Architecture](docs/01-architecture.md) | How it works at the level you need to modify it: thread layout, the scheduler maths, why every queue depth is the number it is. |
| [03 — Operations](docs/03-operations.md) | Power-on order, daily checks, running a measurement set, the failover drill, deploying to a live rig. |
| [04 — Configuration](docs/04-configuration.md) | Every environment variable and tunable constant, what it does, and what breaks at the wrong value. |
| [05 — Results](docs/05-results.md) | The measured numbers, the `p*` derivation, and the threats to validity stated rather than buried. |
| [06 — Troubleshooting](docs/06-troubleshooting.md) | Every symptom this rig actually produced, with its cause and its fix — starting with the five that fail *silently*. |

---

## Quick start

On a rig that is already built — `docs/02-build-from-scratch.md` if not:

```bash
git clone <this-repo> uav-swarm-mec && cd uav-swarm-mec
$EDITOR scripts/hosts.env          # your addresses, logins and sensor ids
WITH_UNITS=1 ./scripts/deploy.sh all
./scripts/preflight.sh
```

Power on **Nano 2 first, wait 60 seconds, then everything else** — the boards claim
the access point on a staggered delay, and the head start makes the intended master
win deterministically. Then:

```bash
cd gcs && MEC_DEADLINE_MS=1200 python3 gcs.py                                    # laptop
ssh -t admindesktop@10.42.0.43   'cd ~/yolov5/swarm && python3 mec_node.py'      # worker
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && python3 mec_node.py'      # master
```

The Pis stream as systemd services and need no command. To run one measurement:

```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && \
  MEC_SCHED=lyapunov MEC_RUN_FRAMES=5000 python3 mec_node.py'
```

---

## Requirements

| Device | Stack |
|---|---|
| Jetson Nano ×2 | JetPack 4.6.x (L4T R32.7.x), Python 3.6.9, torch 1.8.0, torchvision 0.9.0a0, TensorRT 8.2.1.8, OpenCV, pyzmq, yolov5 v6.2 |
| Raspberry Pi ×2 | Raspberry Pi OS 64-bit (Bookworm+), python3-picamera2, python3-opencv, pyzmq, chrony 4.x, NetworkManager |
| Laptop | Python 3, pyzmq, OpenCV, Pillow, tkinter, chrony |

**`OPENBLAS_CORETYPE=ARMV8` is mandatory on the Jetsons.** Without it `import numpy`
aborts with SIGILL — exit code 132 — and prints **nothing at all**: no traceback, no
message. Every Python file here sets it before importing numpy and every systemd unit
carries it, but any new script you write over plain SSH needs it too.

---

## Why the comments are so dense

Nearly every constant in this codebase is a value that was wrong once, and the
comment beside it says what went wrong. A sample, all fixed and all regression-tested
in `tests/test_scheduler_fixes.py`:

- **The `+1` for the current frame was missing from the remote cost estimate.** With
  an idle worker `pending` is 0, so the whole compute term multiplied out to zero and
  the estimate collapsed to bare network RTT — 12 ms against a true ~42 ms.
  Essentially everything offloaded regardless of load.
- **The reliability floor was an absorbing state.** Dispatch outcomes are only
  recorded on the dispatch path, so once a node fell below the floor it was never
  chosen, never measured, and never recovered — for the life of the process.
  Restarting the master was the only thing that brought a worker back.
- **Backpressure was charged as failure.** The master's socket high-water mark is 2
  against a worker needing ~160 ms a frame, so a burst of perfectly healthy "please
  wait" signals crossed the reliability floor in under a second and permanently
  retired a working node.
- **TensorRT warm-up wrote itself permanently into the cost estimate.** Two 460 ms
  opening frames drove a worker's estimate to 786 ms against a 600 ms deadline; the
  feasibility check then declined to probe it, so no result came back and the
  estimate could not be revised. The worker sat idle for the rest of the run — too
  expensive to choose, unable to prove otherwise without being chosen.
- **The thermal penalty multiplied a measured inference time**, charging twice for a
  slowdown already in the number — and asymmetrically, because the master's own
  temperature often read as 0.0. Offloading decayed to nothing over a long run.
- **The deadline was measured across two machines' clocks.** A board that came up
  with a wrong clock, was promoted to master, and was then stepped by chrony dropped
  100% of frames for minutes on a deadline that had not elapsed.
- **`fsync` per row sat on the frame path.** One stalled SD-card write filled the
  results queue, blocked the results thread, left the worker's in-flight count high,
  and made the scheduler read an idle worker as overloaded. The display went quiet
  and offloading stopped at the same moment, for the same reason.
- **The jitter margin scaled linearly with queue depth instead of √depth**, charging
  a deep queue for a worst case where every frame runs long together, and vetoing
  the worker exactly when it was needed.

```bash
PYTHONPATH=nano python3 tests/test_scheduler_fixes.py -v
```

Each test states a defect as a property that must hold, so a change that reintroduces
one fails here rather than in a flight test.

---

## Status and honest limits

Working, and measured. Read `docs/05-results.md` before quoting anything. In short:

- **One 5000-frame run per configuration.** No statistical replication, no confidence
  intervals. Differences of a few percentage points are not distinguishable from
  run-to-run noise by this dataset.
- **No mAP evaluation.** JPEG q80 was chosen over q95 on bandwidth grounds; the claim
  that it does not change detections at conf 0.60 is reasoned, not measured.
- **No direct energy measurement** — temperature and GPU duty cycle are proxies.
- **No join-the-shortest-queue baseline**, and one failover drill rather than a
  distribution of recovery times.
- **Lyapunov's thermal term is inert on this hardware.** It returns exactly 1.0 below
  70 °C and the boards run at 33–42 °C, so the penalty term is zero on every frame,
  Lyapunov reduces to `min(Q·w)`, and sweeping `V` changes nothing.
- **The two Jetson Nanos are not identical** — 50.9 ms vs 95.5 ms under identical
  power mode, pinned clocks, TensorRT version and FP16 bindings, traced to cuDNN
  tactic availability across L4T releases. Bench both boards before quoting either.
- **The deadline gate makes more routing decisions than the algorithms do under
  overload**, which is why the three policies converge to within 18 ms at the most
  stressed load point. That is a property of the operating point, not of the
  schedulers.
- **Ground-based hardware-in-the-loop, not a flight test.** The Pi/Jetson pairing
  emulates the sensor-and-compute split of an aerial platform without requiring the
  platform. Nothing here should be read as an airborne result.
- **WPA2-PSK only, no join authentication** — any device with the key can associate
  and, if it runs `mec_node.py`, be handed frames as a worker.

---

## Academic context

**Project:** Proof-of-Concept Development of a UAV-Based Distributed Processing
Framework
**Author:** Muhammad Samaan · CMS 432625 · 99th (B) EC
**Institution:** College of Aeronautical Engineering, PAF Academy Asghar Khan,
Risalpur
**Advisors:** Dr. Ahnaf Lodhi · Engr. Atif Shahzad

Project deliverables, per the Project Definition Document:

1. Design and simulation of a UAV-based distributed processing framework
2. Physical proof-of-concept implementation using COTS UAV hardware
3. Experimental performance evaluation and validation report

This repository is deliverable 2, plus the instrumentation and analysis tooling that
produced deliverable 3. The full report and research paper are maintained separately.

---

## License

MIT — see [LICENSE](LICENSE).
