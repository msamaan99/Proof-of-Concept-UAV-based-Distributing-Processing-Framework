# Troubleshooting

Every symptom that has actually occurred on this rig, with its cause and its fix.
Ordered by how much time it cost.

---

## The five silent failures

These produce a run that **completes and looks entirely normal** while every
number in it is wrong. There is no error message for any of them. Check all five
before trusting any result.

### 1. Both Pis streaming under the same sensor id

The master keys `FrameStore` on `"sensor:frame_id"`. Two Pis claiming `pi1`
interleave their sequences in one keyspace and overwrite each other's frames.

```bash
for T in admin@10.42.0.31 pi02@10.42.0.128; do
  echo "== $T"; ssh "$T" 'systemctl show -p Environment pi-sensor | tr " " "\n" | grep SENSOR'
done
```

Must print `pi1` and `pi2`. Fix:

```bash
ssh -t pi02@10.42.0.128 'sudo sed -i "s|^Environment=PI_SENSOR_ID=.*|Environment=PI_SENSOR_ID=pi2|" \
  /etc/systemd/system/pi-sensor.service && sudo systemctl daemon-reload && sudo systemctl restart pi-sensor'
```

The advisory lock on `/tmp/pi_sensor.lock` now prevents the most common route into
this — starting the sensor by hand while the service is already running — but it
cannot prevent two *boards* misconfigured identically.

### 2. `frame_trace.jsonl` blending two runs

The file is appended to, never truncated. Two runs land in one file and the reader
reports aggregate means over both.

**Nothing looks wrong.** The coverage header still reads `100.0% coverage`, because
the Pi restarts its frame ids each session so the header dedupes by id while the
totals do not. The tell is **identical extremes across two supposedly separate
runs** — the same inference max, the same GPU queue max, the same Pi→master max.

```bash
mkdir -p gcs_logs/old && \
mv gcs_logs/frame_trace.jsonl gcs_logs/old/frame_trace_$(date +%H%M%S).jsonl
```

Rotate before **every** capture, then sanity-check that the record count roughly
matches the master's own frame count minus `MEC_TRACE_SKIP`.

### 3. The master board changed between runs

Master is whoever claimed `192.168.50.1` first, which can differ between runs with
nothing in any log saying so. The two boards are **not** interchangeable — their
inference medians have differed by up to 2× under identical configuration — so
swapping the master silently inverts a local-vs-offload comparison. On this rig one
assignment made offloading look like it *saved* 40 ms (−28.9%) and the other made
it *cost* 79 ms.

```bash
./scripts/preflight.sh roles
```

Run it before every single run and write down which board it was.

### 4. A stale `__pycache__`

Python only recompiles when the source mtime is newer, and `scp` preserves nothing
by default. A board kept running the previous scheduler after its source had been
replaced.

`deploy.sh` removes the cache and md5-verifies every file, so use it rather than
bare `scp`. To check by hand:

```bash
for T in admindesktop@10.42.0.43 admindesktop1@10.42.0.226; do
  echo "== $T"; ssh "$T" 'cd ~/yolov5/swarm && md5sum mec_node.py scheduler_*.py completion_time_scheduler.py'
done
```

Both boards must report identical sums, and they must match the local files.

### 5. GCS gap counts that are really a saturated rate limiter

`ORDER_QUIET_SEC = 10` caps `sequence gap` and `out of order` rows at one per
sensor per kind per 10 seconds. Three 168-second runs each logged exactly **17** of
each, 10.0 s apart — 168 ÷ 10 = 16.8 → 17. Read as "all three schedulers behaved
identically"; the real figures, recomputed from the master's completion order, were
**69.1% / 61.9% / 57.2%**.

**Check whether the count equals `run_seconds / 10` before quoting it.** If it
does, recompute from the master's `results_*.csv` instead. `frame dropped` rows are
not rate-limited and do match the master exactly.

---

## Startup and roles

| Symptom | Cause and fix |
|---|---|
| **`Waiting for the network layer`** | `swarm_net` is not running. `systemctl status swarm_net`. The processing layer waits rather than guessing a role. |
| **Both Nanos say MASTER** | The claim delays are equal. Check `CLAIM_DELAY_BY_NODE` in `swarm_net.py` — `nano2: 5.0`, `nano1: 30.0` — and that `NODE_ID_BY_HOST` matches the actual hostnames. |
| **Both say WORKER** | Neither holds `192.168.50.1`. `ip -4 addr show wlan0` on each — the AP profile failed to activate. Try `sudo nmcli connection up NanoNet-ap` by hand and read the error. |
| **Master never finds the worker** | On the master: `cat runtime/neighbors.json \| python3 -m json.tool`. The peer needs `online`, `mec_active` **and** `mec_role: worker` all true. If `mec_active` is false, `mec_node.py` is not running there. |
| **`neighbors.json` missing** | Both layers resolve it to `<dir holding the scripts>/runtime/`, so they always agree. If it is absent, `swarm_net` has not written one yet. |
| **A board yields the AP and loops every ~22 s** | Fixed: `acquire()` now deactivates the AP profile *before* scanning. A board that had just yielded found its own SSID, called `_join()`, and `_join()` tore down the AP it was trying to join. If you see it, you are on old code. |
| **The AP tears itself down when nobody has joined** | `EMPTY_NETWORK_GRACE_SEC` is not 0. At 60 s a board that came up before the Pis were powered destroyed the only network on the air. |
| **No model found** | `mec_node.py` looks for `yolov5n.engine` or `yolov5n.pt` in the **parent** of its own folder. The error names both paths it tried. The code must live in `<yolov5 repo>/swarm/`. |
| **Node pinned with `--role` will not fail over** | Working as intended. Use plain `mec_node.py` for the failover drill. |

---

## Python dies with no output at all

**Exit code 132, nothing printed, no traceback, even with `python3 -u`.**

```bash
ssh admindesktop@10.42.0.43 'python3 -c "import numpy"'; echo "rc=$?"
```

`rc=132` is 128 + SIGILL. OpenBLAS misdetects the Cortex-A57 and the **CPU** kills
the process, so Python never gets to report anything. It reads like a hang, a slow
import, or an OOM kill.

```bash
export OPENBLAS_CORETYPE=ARMV8
```

Every Python file in this repo sets it before importing numpy, and every systemd
unit carries it. The gap it covers: `~/.bashrc` only runs for *interactive* shells,
so `python3 script.py` typed at the board works while
`ssh nano 'python3 script.py'` dies.

---

## Performance that makes no sense

| Symptom | Cause and fix |
|---|---|
| **Inference 2× slower than expected, temperatures fine** | `jetson_clocks` does not survive a reboot. The GPU sits at its 76.8 MHz floor and ramps per frame, worst at low duty cycle — i.e. on the worker. `sudo nvpmodel -m 0 && sudo jetson_clocks`. Verify with `./scripts/preflight.sh perf`. |
| **The two boards differ by ~2× with identical config** | This is real and was traced to cuDNN convolution tactic availability differing between L4T R32.7.1 and R32.7.6. Ruled out by direct measurement: power mode, CPU/GPU/EMC clock, thermal, TensorRT version, engine precision, engine build quality, torch/torchvision, CPU preprocessing, NMS. **Not fixable without reflashing.** Report it as a system property — *heterogeneous edge nodes, 50.9 ms and 95.5 ms, identical hardware and clocks, differing cuDNN tactic sets* — and keep the same board as master across a set. |
| **Worker inference looks slower than the master's for the same board** | They measure different spans. The master times `infer_start → infer_done`, excluding the JPEG decode it does in its own stage; the worker times `worker_recv → worker_done`, which **includes** decode. Use `bench_infer.py` to compare boards, not the pipeline's own numbers. |
| **Offload decayed to nothing over a long run** | Fixed: `thermal_penalty` was multiplying the measured `infer_time`, charging twice for a slowdown already in the number, and the double charge landed on the worker while sparing the master. If you see it, you are on old code. |
| **A worker went idle and never came back** | Three fixed causes, all the same shape — a transient writing itself permanently into a stat that only updates on use. (a) Reliability floor was an absorbing state → `record_liveness_tick` heals it. (b) Backpressure counted as failure → `record_backpressure` no longer touches reliability. (c) A warm-up-inflated estimate could not be revised because the node was never chosen → `StalenessProbe` forces a probe after 30 s regardless. Diagnose with `MEC_VERBOSE=1` and read the `excluded` / `infeasible` / `costlier` counters. |
| **~100% offload regardless of load** | Fixed: the `+1` for this frame was missing from `estimate_time`, so an idle worker's compute term multiplied out to zero and the estimate collapsed to bare RTT — 12 ms against a true ~42 ms. |
| **Everything local, frames dropped instead of offloaded** | Jitter was scaling linearly with queue depth instead of √depth, padding the estimate by ~90 ms at depth 2 and pushing the worker past the deadline. Also check `MEC_DEADLINE_MS` is 1200, not 600. |

### Which term is rejecting the worker

```bash
MEC_VERBOSE=1 python3 mec_node.py
```

```
Routing — local_est=104ms remaining=1093ms | nano1:chosen=38, nano1:costlier=61
  nano1: infer=57ms rtt=6ms ijit=3ms rjit=1ms rel=1.00 busy=0 excluded=False backpressure=0
```

| Bucket | Means | Fix is |
|---|---|---|
| `excluded` | `estimate_time` returned `inf` — battery, RAM, or reliability | a health-signal problem |
| `infeasible` | finishable in principle, but not before the deadline | a deadline or capacity problem |
| `costlier` | feasible and available; the scheduler preferred local | a tuning problem |
| `chosen` | offloaded | — |

A run that is 100% local with `costlier` dominant is very different from the same
run with `infeasible` dominant, and they have different fixes. Aggregate CSV rows
cannot tell them apart: a worker ruled out simply stops appearing, and "no rows
from NANO1" reads the same for all three causes.

---

## Network and link

| Symptom | Cause and fix |
|---|---|
| **The link drops every 20-40 s, in contiguous 120-140 frame blocks** | Wi-Fi power save. `./scripts/preflight.sh radio`, then `sudo iw dev wlan0 set power_save off` on every client **including the laptop** — it is the other end of the GCS hop. Make it permanent with `802-11-wireless.powersave 2` in the profile. |
| **A Pi boots and never joins, profile says `autoconnect yes`** | `connection.autoconnect-retries` defaults to **4**. Nano 2 claims at 5 s and Nano 1 at ~30 s, so a Pi powered on at the same time exhausts all four attempts looking for an SSID that does not exist yet, then **sits idle forever** — NM will not retry without an external event. `sudo nmcli connection modify NanoNet connection.autoconnect-retries 0`. `nanonet-link.service` also repairs this at startup. |
| **A Pi is "connected" but no frames arrive** | Associated is not the same as connected. After a failover a Pi can stay associated to a radio that no longer forwards anything, and NM sees a connected device and does nothing. Only a reachability probe notices — which is what `nanonet-link.service` does. `ping -c3 192.168.50.1` from the Pi. |
| **A board loses internet and wired SSH after joining NanoNet** | The NanoNet profile has `ipv4.gateway` set, which installs a default route over `wlan0`. NanoNet has no route anywhere. `sudo nmcli connection modify NanoNet ipv4.gateway "" ipv4.never-default yes`. This is also why `apt` suddenly cannot reach the archives. |
| **`apt` fails on Pi 2 only** | That upstream 403s any request whose User-Agent looks like apt — verified: the identical URL returned 200 as `curl/8.5.0` and 403 as `Debian APT-HTTP/1.3`. `./scripts/pi2_setup.sh apt` writes a User-Agent override and only falls back to plaintext if that fails. |
| **GCS blank** | It **binds** 6000 and the master connects out. Check `GCS_IP` in `mec_node.py` matches the laptop's NanoNet address, and that the laptop actually holds it: `ip -4 addr show wlp3s0`. |
| **Only one pane appears** | Set `SENSORS` in `gcs/gcs_common.py` to your `PI_SENSOR_ID` values. Not a whitelist — an unlisted sensor still gets a pane on first frame. |
| **`REMOTE HOST IDENTIFICATION HAS CHANGED`** | The board regenerated its SSH host key (Pi 2 did, swapping ECDSA for ED25519). Confirm the MAC is the board you think it is, then `ssh-keygen -f ~/.ssh/known_hosts -R '10.42.0.128'`. |
| **SSH to a board fails after a reboot** | With `autoconnect no` on the Nano profiles, nothing brings `wlan0` up except `swarm_net.py`. If it failed to start, the board is off the Wi-Fi entirely. Use the wired path, or a monitor and keyboard. |

---

## Clocks

| Symptom | Cause and fix |
|---|---|
| **A trace segment comes out NEGATIVE** | Clock skew between the two machines it spans. The reader warns rather than letting you publish it. `./scripts/preflight.sh clocks`, wait for convergence, re-run. |
| **`Reference ID : 00000000` a minute after the network is up** | Nano 2 is dropping UDP 123. `sudo ufw allow from 192.168.50.0/24 to any port 123 proto udp`. Confirm from the other side with `sudo chronyc clients` on Nano 2 — every Pi and Nano 1 should be listed. |
| **Two sources both marked `^x`, syncing to neither** | The stratum deadlock. chrony ranks by stratum only among sources it has already judged truthful; two sources disagreeing past tolerance form no majority. Observed with Nano 1 holding `.1`, polling itself, deadlocked against Nano 2 which was 3.66 s away and fully reachable. **`trust` on the `.55` line is what fixes it**, not stratum. |
| **`^-` against the master's address** | Reachable-but-unused. **This is correct, not a fault.** Healthy state is `^*` on `.55` and `^-` on `.1`. |
| **The GCS shows a latency of billions of ms** | A master with a wrong clock. It happened: `149294645155 ms` — four and a half years — rendered verbatim. Anything past 60 s is now rejected and shows as a dash. |
| **100% of frames dropped with `DEADLINE_EXCEEDED` for minutes** | Fixed: the deadline used to be `now - capture_ts`, a cross-machine subtraction. A board that came up with a wrong clock, was promoted to master, and was then stepped by chrony dropped everything on a deadline that had not elapsed. It is now measured from arrival on a monotonic clock. If you see this, you are on old code. |
| **Local timestamps a day off** | Timezone. Pi 1 was set to `America/Adak`. `sudo timedatectl set-timezone Asia/Karachi`. |
| **A board comes back from reboot with a clock years off** | No RTC hardware — `timedatectl` reports `RTC time: n/a`. `fake-hwclock` is the only thing carrying the clock across a reboot: `sudo apt install -y fake-hwclock && sudo fake-hwclock save`. |

---

## The pipeline

| Symptom | Cause and fix |
|---|---|
| **Results CSV missing rows** | Said out loud rather than left silent: the summary line reports `csv rows lost N`. `csv_queue` is lossy when full by design — a stalled SD card costs log rows, never frames. |
| **`WORKER_RESULT_TIMEOUT` drops** | The worker was sent frames it never answered for. Check it is alive and that its own log shows inference happening. The reaper writes these off after 2 s so the in-flight count does not stay permanently high. |
| **`SHED_STALE` drops** | `task_queue` was full and the **oldest** frame was shed. Correct behaviour for a live feed, and it means the arrival rate exceeds what the pair can serve. Load shows up as a lower frame rate rather than as an outage. |
| **`LOCAL_QUEUE_FULL` drops** | Both local stages are full. The master is genuinely saturated. |
| **`CORRUPT_FRAME` drops** | `cv2.imdecode` returned None. Previously skipped silently at ingest, which made a sensor sending garbage look like a sensor sending nothing. |
| **`WORKER_LINK_BLOCKED`** | The worker's pipe was full — backpressure, not failure. The frame falls back to local. Does **not** count against reliability. |
| **The display goes quiet and offloading stops at the same moment** | The old disk-on-the-frame-path cascade: `results_queue` fills → `_emit` blocks → `peer["busy"]` stays high → the scheduler reads an idle worker as overloaded. **One stalled write, both symptoms.** Fixed by moving CSV writes to their own thread behind a deep queue. |
| **The Pi feed stops but the process is still running** | The watchdog handles this now: no delivery for 10 s rebuilds the socket, no loop iteration for 20 s exits with code 3 for systemd to restart. Check `journalctl -u pi-sensor` for which fired. |
| **`Pi sent fewer frames` when the master is saturated** | Only with `PI_STRICT=0`. The old `NOBLOCK` send discarded the frame *without advancing the id*, so the loss left no gap and nothing downstream could see it. Strict mode (the default) blocks up to 1 s and the shortfall shows in the achieved rate. |

---

## Results that look wrong

| Symptom | What is actually going on |
|---|---|
| **Round robin did not do 50/50** | `enforce_deadline` overrode its rotation. Under heterogeneity it overrides constantly: in the battery runs RR intended 2,500 frames for the worker, 1,300 arrived, 639 were redirected local and 561 dropped — a 29.3% realised rate. **The shared safety net had done the load balancing RR refuses to do.** Use `MEC_NO_DEADLINE=1` to make the rotation absolute and see what the policy actually costs. |
| **Greedy and Lyapunov are nearly identical** | They share the profiler, the feasibility estimates, the deadline gate and the exploration probe by design; only the preference rule differs. And **half of Lyapunov's objective is switched off**: `thermal_penalty()` returns exactly 1.0 below 70 °C and these boards ran at 33-42 °C, so the whole penalty term is zero on every frame and Lyapunov collapses to `min(Q·w)`. Sweeping `V` changes nothing, because `V` only appears multiplied by that zero. |
| **A policy has the lowest mean latency and the worst outcome** | Mean latency over *delivered* frames cannot rank policies under loss. Fixed 65% had the lowest mean (848 ms) at LP4 and the second-worst useful yield — its standard deviation was 710 ms against Lyapunov's 114 ms, and the low mean was a bimodal artefact of discarding a fifth of the frames. See [`05-results.md`](05-results.md). |
| **All three policies converge and the table cannot discriminate** | The load point is too close to the ceiling. At 30 fps with a 3× worker the combined ceiling is ~26 fps, so every policy is above capacity and they can only differ in *how* they fail. Drop the sensor rate until the gate stops firing constantly. |
| **Temperatures identical across policies** | Correct, and so are `Avg GPU inference` and `Avg network RTT` — those are hardware and link properties. No scheduler changes how fast a Nano runs YOLOv5n. If they *differed*, something would be wrong with the experiment. |
| **A battery-powered run is confounded** | A draining battery is a moving target: the worker gets slower as voltage sags, so whichever policy runs last faces the slowest worker, and that confound looks exactly like a scheduling difference. On this rig the worker drifted **37%** across three runs. Bench the worker before *and* after the set; if the medians differ, say by how much. |

---

## Diagnostic commands, all together

```bash
# The full gate before a measurement set
./scripts/preflight.sh && ./scripts/preflight.sh soak

# Which board is master, right now
./scripts/preflight.sh roles

# The network layer's view
ssh admindesktop1@10.42.0.226 'journalctl -u swarm_net -n 40 --no-pager'
ssh admindesktop1@10.42.0.226 'cat ~/yolov5/swarm/runtime/neighbors.json | python3 -m json.tool'

# Who is on the network, and who is drawing time
ssh -t admindesktop1@10.42.0.226 'iw dev wlan0 station dump | grep Station; sudo chronyc clients'

# The Pi side
ssh admin@10.42.0.31 'journalctl -u pi-sensor -n 30 --no-pager; journalctl -u nanonet-link -n 20 --no-pager'

# Board speed, off-pipeline
ssh -t admindesktop@10.42.0.43 'cd ~/yolov5/swarm && python3 bench_infer.py -n 200'
# ... and with an idle gap, to separate the clock ramp from the hardware
ssh -t admindesktop@10.42.0.43 'cd ~/yolov5/swarm && python3 bench_infer.py -n 100 --idle 1.0'

# What is inside each engine
for T in admindesktop@10.42.0.43 admindesktop1@10.42.0.226; do
  ssh -t "$T" 'cd ~/yolov5/swarm && python3 engine_info.py'; done

# Per-frame stage breakdown
python3 shared/frame_trace.py gcs_logs/frame_trace.jsonl

# Regression tests for the scheduler defects above
PYTHONPATH=nano python3 tests/test_scheduler_fixes.py -v
```
