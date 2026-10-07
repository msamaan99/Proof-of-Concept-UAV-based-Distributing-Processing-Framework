# UAV Swarm MEC — Distributed Inference Across a Self-Forming Edge Network

A working proof of concept for **UAV-based distributed processing**: Raspberry Pi
sensors stream camera frames to a pair of Jetson Nanos that form their own Wi-Fi
network, decide per frame which board should run YOLOv5n object detection, and
forward results to a ground station — with **no centralised infrastructure, no
static master, and no single point of failure.**

Kill the board running the network and the other one rebuilds it, takes over the
master's address, and keeps serving. The sensors never notice.

Built for a Final Year Design Project at the College of Aeronautical Engineering,
PAF Academy Asghar Khan. **Everything here runs on real hardware and every number
in [`docs/05-results.md`](docs/05-results.md) is measured, not simulated.**

---

## What it does

```
  Raspberry Pi 1 ──┐                  Jetson Nano  (MASTER)                  Laptop
  camera, q80 JPEG ├──► :5000 ───────►  owns the Wi-Fi AP at .1        ┌───►  dual-pane feed
  Raspberry Pi 2 ──┘                    schedules every frame          │      event log
                                        runs YOLOv5n locally    ───────┘      recovery timing
                                              │
                                              │ :5001  offload
                                              ▼
                                      Jetson Nano  (WORKER)
                                        YOLOv5n, returns JSON  ──► :5003
```

**The role is claimed, not configured.** Whoever holds `192.168.50.1` is the
master — there is no election, no config flag, no virtual IP. A board that reboots
after a failure runs the same rule, finds the network its partner built, and joins
it as a worker. It cannot resume a role it no longer holds, because the role lives
in the network rather than in the code.

**Four scheduling policies compete on identical hardware**, selected by one
environment variable:

| `MEC_SCHED` | Policy | Decision rule |
|---|---|---|
| `lyapunov` | Drift-plus-penalty | `argmin Q_i·w_i + (V+Z_i)·(e_i−1)` — queue stability vs thermal cost |
| `greedy` | Earliest completion time | `argmin` estimated completion, with hysteresis |
| `rr` | Round robin | `i % n`. The naive baseline. |
| `fixed` | Fixed ratio | Closed-loop on a ratio you set. The *controlled* baseline. |

All four share one cost model, one deadline gate and one exploration probe, so a
comparison between them isolates the decision rule rather than measuring which
implementation happens to keep fresher statistics.

---

## Headline results

Measured on 2 × Jetson Nano + 2 × Raspberry Pi, 5000 frames per policy, deadline
1200 ms. Full tables and caveats in [`docs/05-results.md`](docs/05-results.md).

**Drop rate misranks schedulers under overload — the project's most transferable
finding.** At ρ = 1.07, round robin's drop rate (8.20%) is indistinguishable from
Lyapunov's (8.20%). By **useful yield** — delivered *and* within deadline — round
robin manages at most **46.6%** against **87%** for the adaptive pair, because its
mean delivered latency (1618 ms) exceeds the deadline itself.

| | ρ | Best mean latency | The real story |
|---|---|---|---|
| **LP1** 1 sensor, 15 fps | 0.43 | Lyapunov 73.1 ms | Greedy offloads 7.9% — correctly declining |
| **LP2** 1 sensor, 30 fps | 0.80 | RR 72.5 ms | **Zero Lyapunov frames of 5000 exceed 250 ms**, vs 21 and 14 |
| **LP3** worker derated | ≈1 | — | k=1.95; 40–65% of frames arrive out of capture order |
| **LP4** 2 sensors, 40 fps | 1.07 | Fixed 65% 848 ms | …and it has the *second-worst* useful yield |

Two policies with entirely different rules converge on the same offload fraction at
saturation — Lyapunov 47.30%, greedy 47.14% — within 0.4 points of the
capacity-matched `p* = 1/(1+k)`. **Below saturation that fixed point does not bind
and they diverge from it by up to 39 points**, which is a precondition routinely
omitted when `p*` is quoted.

Scheduling costs **0.17 ms median** against ~50 ms of inference — 0.13% of a
frame, three orders of magnitude cheaper than the work it routes.

---

## Repository layout

Directories mirror where the code runs. `deploy.sh` copies each one to the right
place.

```
nano/          → ~/yolov5/swarm/ on both Jetsons
  mec_node.py                    processing layer: master and worker halves
  swarm_net.py                   network layer: forms/heals NanoNet, telemetry
  completion_time_scheduler.py   shared cost model + greedy ECT + the probe
  scheduler_lyapunov.py          drift-plus-penalty
  scheduler_rr.py                round robin baseline
  scheduler_fixed.py             fixed-ratio controlled baseline
  bench_infer.py                 measure one board off-pipeline
  engine_info.py                 what is actually inside a TensorRT engine
  systemd/                       swarm_net.service, mec_node.service

pi/            → ~/pi_sensor/ on both Raspberry Pis
  pi_sensor.py                   camera capture, JPEG, stream, watchdog
  nanonet_link.py                root link supervisor — keeps wlan0 on the swarm
  systemd/                       pi-sensor.service, nanonet-link.service

gcs/           → stays on the laptop
  gcs.py                         ground station, DIRECT playback
  gcs_buffered.py                ground station, BUFFERED playback
  gcs_common.py                  everything the two builds must agree on

shared/
  frame_trace.py                 per-frame stage timing; symlinked into nano/ and gcs/

scripts/
  hosts.env                      THE rig inventory — the only file you edit
  deploy.sh                      md5-verified deploy to any subset of the rig
  preflight.sh                   every pre-run gate, one command
  chrony_setup.sh                time sync, one argument per device role
  pi1_setup.sh / pi2_setup.sh    Raspberry Pi provisioning, staged

tools/
  algo_report.py                 raw per-frame CSVs → comparison tables

tests/
  test_scheduler_fixes.py        17 regression tests, one per historical defect

docs/            see below
results/         derived comparison tables + a sample per-frame trace
```

`shared/frame_trace.py` is imported by both the Jetsons and the ground station, so
`nano/frame_trace.py` and `gcs/frame_trace.py` are **symlinks** to it — one source
of truth, and `scp` follows them so each device receives a plain file.

---

## Documentation

| | |
|---|---|
| **[02 — Build From Scratch](docs/02-build-from-scratch.md)** | **Start here.** Bare boards to measured result. Every command written out, nothing left to infer. ~5 hours. |
| [01 — Architecture](docs/01-architecture.md) | How it works, at the level you need to modify it. Thread layout, the scheduler maths, why every queue depth is what it is. |
| [03 — Operations](docs/03-operations.md) | Power-on order, daily checks, running a measurement set, the failover drill, deploying to a live rig. |
| [04 — Configuration](docs/04-configuration.md) | Every environment variable and constant, what it does, what breaks at the wrong value. |
| [05 — Results](docs/05-results.md) | The measured numbers, with the threats to validity stated rather than buried. |
| [06 — Troubleshooting](docs/06-troubleshooting.md) | Every symptom that actually occurred on this rig, with its cause and its fix. |

---

## Quick start

Assuming a rig that is already built ([`docs/02`](docs/02-build-from-scratch.md) if
not):

```bash
git clone <this-repo> uav-swarm-mec && cd uav-swarm-mec
$EDITOR scripts/hosts.env          # your addresses, logins and sensor ids
WITH_UNITS=1 ./scripts/deploy.sh all
./scripts/preflight.sh
```

Then power on **Nano 2 first, wait 60 s, then everything else**, and:

```bash
cd gcs && MEC_DEADLINE_MS=1200 python3 gcs.py                                    # laptop
ssh -t admindesktop@10.42.0.43   'cd ~/yolov5/swarm && python3 mec_node.py'      # worker
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && python3 mec_node.py'      # master
```

The Pis stream as systemd services and need no command.

---

## What makes this more than a demo

The comments in these files are unusually dense, and deliberately so: nearly every
constant is a value that was wrong once, and the comment says what went wrong. A
few examples, all fixed and all regression-tested:

- **The `+1` for the current frame was missing from the remote cost estimate**, so
  an idle worker's compute term multiplied out to zero and looked like a 12 ms
  round trip against a true 42 ms. Essentially everything offloaded regardless of
  load.
- **The reliability floor was an absorbing state.** Dispatch outcomes are only
  recorded on the dispatch path, so once a node fell below the floor it was never
  chosen, never measured, and never recovered — for the life of the process.
- **Backpressure was charged as failure.** The master's socket high-water mark is 2
  against a worker needing ~160 ms a frame, so a burst of healthy "please wait"
  signals crossed the reliability floor in under a second and permanently retired a
  working node.
- **TensorRT warm-up wrote itself permanently into the cost estimate.** Two 460 ms
  opening frames drove a worker's estimate to 786 ms against a 600 ms deadline;
  the feasibility check then declined to probe it, so no result ever came back and
  the estimate could not be revised. The worker sat idle for the rest of the run.
- **The thermal penalty multiplied a measured inference time**, charging twice for
  a slowdown already in the number — and asymmetrically, because the master's own
  temperature often read as 0.0. Offloading decayed to nothing over a run.
- **The deadline was measured across two machines' clocks.** A board that came up
  with a wrong clock, was promoted to master, and was then stepped by chrony
  dropped 100% of frames for minutes on a deadline that had not elapsed.
- **`fsync` per row sat on the frame path.** One stalled SD-card write filled the
  results queue, blocked the results thread, left the worker's in-flight count
  high, and made the scheduler read an idle worker as overloaded — the display went
  quiet and offloading stopped at the same moment, for the same reason.
- **The jitter margin scaled linearly with queue depth instead of √depth**,
  charging a deep queue for a worst case where every frame runs long together, and
  vetoing the worker exactly when it was needed.

The tests in [`tests/test_scheduler_fixes.py`](tests/test_scheduler_fixes.py) state
each of those as a property that must hold:

```bash
PYTHONPATH=nano python3 tests/test_scheduler_fixes.py -v
```

---

## Requirements

| Device | Stack |
|---|---|
| Jetson Nano ×2 | JetPack 4.6.x (L4T R32.7.x), Python 3.6.9, torch 1.8.0, torchvision 0.9.0a0, TensorRT 8.2.1.8, OpenCV, pyzmq, yolov5 v6.2 |
| Raspberry Pi ×2 | Raspberry Pi OS 64-bit (Bookworm+), python3-picamera2, python3-opencv, pyzmq, chrony 4.x, NetworkManager |
| Laptop | Python 3, pyzmq, OpenCV, Pillow, tkinter, chrony |

**`OPENBLAS_CORETYPE=ARMV8` is mandatory on the Jetsons.** Without it `import
numpy` aborts with SIGILL (exit 132) and prints **nothing at all** — no traceback,
no message. Every Python file here sets it before importing numpy and every systemd
unit carries it, but any new script you write over plain SSH needs it too.

---

## Status and honest limits

Working and measured. Read [`docs/05-results.md`](docs/05-results.md#threats-to-validity)
before quoting anything. In short:

- **One 5000-frame run per configuration**, no statistical replication.
- **No mAP evaluation**, no direct energy measurement, no join-the-shortest-queue
  baseline, one failover drill.
- **Lyapunov's thermal term is inert on this hardware** — it returns exactly 1.0
  below 70 °C and the boards run at 33-42 °C, so the penalty term is zero on every
  frame and sweeping `V` changes nothing.
- **The two Jetson Nanos are not identical.** 50.9 ms vs 95.5 ms under identical
  configuration, traced to cuDNN tactic availability across L4T releases. Bench
  both boards before quoting either.
- **Ground-based hardware-in-the-loop, not a flight test.**
- **WPA2 only, no join authentication** — any device with the PSK can become a
  worker.

---

## ⚠ Before making this repository public

**The rig's real Wi-Fi PSK is committed** — in `scripts/hosts.env`,
`scripts/pi2_setup.sh` and `docs/02-build-from-scratch.md`. It is written out
rather than left as a placeholder because the build guide is meant to be followed
without guessing, and a rig whose password is `YourPassword123` in the docs and
something else in reality is exactly how the two drift apart.

Either change the rig's PSK, or scrub it from the repository first:

```bash
grep -rln nanonet123 . | xargs sed -i 's/nanonet123/CHANGEME/g'
```

Then set the new value on every device — `NanoNet-ap` and `NanoNet-client` on each
Jetson, `NanoNet` on each Pi and the laptop:

```bash
sudo nmcli connection modify <profile> wifi-sec.psk "<new-psk>"
```

Nothing else in the repository is sensitive: the `10.42.0.x` and `192.168.50.x`
addresses are private, and no keys, tokens or personal data are committed.

---

## License

MIT — see [LICENSE](LICENSE).
