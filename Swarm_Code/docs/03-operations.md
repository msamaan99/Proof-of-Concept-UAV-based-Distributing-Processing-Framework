# Operations

Day-to-day running, once the system is built. For the build itself see
[`02-build-from-scratch.md`](02-build-from-scratch.md).

---

## Power-on sequence

**Nano 2 first, alone. Wait 60 s. Then everything else.**

```
t=0     power Nano 2                    claims 192.168.50.1 at t≈5 s
t=60s   power Nano 1, both Pis          Nano 1 finds NanoNet and joins at .55
t=65s   join the laptop to NanoNet      the Pis autoconnect on their own
```

Nano 2 claims at 5 s, Nano 1 at 30 s — so giving Nano 2 a clear head start makes
it win the race deterministically, and it is the board you want as master because
it is also the chrony reference.

Everything except `mec_node` starts itself:

| Service | Board | Enabled |
|---|---|---|
| `swarm_net` | both Jetsons | yes |
| `mec_node` | both Jetsons | only for unattended demos |
| `nanonet-link` | both Pis | yes |
| `pi-sensor` | both Pis | yes |

### What a reboot resets and you must re-apply

```bash
# Both Jetsons — jetson_clocks does NOT persist
for T in admindesktop@10.42.0.43 admindesktop1@10.42.0.226; do
  ssh -t "$T" 'sudo nvpmodel -m 0 && sudo jetson_clocks'; done
```

```bash
# The laptop's Wi-Fi power save, if NetworkManager reset it
sudo iw dev wlp3s0 set power_save off
```

Then verify:

```bash
./scripts/preflight.sh
```

---

## Daily checks, in order of what they catch

```bash
./scripts/preflight.sh roles      # which board is master — record it
./scripts/preflight.sh clocks     # chrony on all four devices
./scripts/preflight.sh perf       # power mode and pinned GPU clocks
./scripts/preflight.sh radio      # Wi-Fi power save everywhere
./scripts/preflight.sh engines    # engine checksums
./scripts/preflight.sh soak       # 9 min — the gate before a measurement set
```

`./scripts/preflight.sh` with no argument runs everything except the soak.

---

## Running a demo

Three terminals, in this order. **The GCS must be listening before the master
connects out to it** — the master connects, the GCS binds.

```bash
# 1 — laptop
cd gcs && MEC_DEADLINE_MS=1200 python3 gcs.py
```

```bash
# 2 — the worker. No variables; it runs no scheduler.
ssh -t admindesktop@10.42.0.43 'cd ~/yolov5/swarm && python3 mec_node.py'
```

```bash
# 3 — the master
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && python3 mec_node.py'
```

The Pis are already streaming as services. Watch one if you want:

```bash
ssh admin@10.42.0.31 'journalctl -u pi-sensor -f'
```

### Or fully unattended

```bash
for T in admindesktop@10.42.0.43 admindesktop1@10.42.0.226; do
  ssh -t "$T" 'sudo systemctl enable --now mec_node'; done
```

Then the whole rig comes up on power alone and you only start the GCS. Use this
for a demo that must survive a reboot with nobody at the keyboard; use the manual
form when you want to watch the role assignment happen.

### Reading the master's log

```
11:47:41 | INFO | mec | 600 frames | lat avg 78ms p95 121ms | offload 41% | drop 1.4%
11:47:41 | INFO | mec |   nodes: nano2 72C cpu 61% ram 68% · nano1 68C cpu 44% ram 57%
11:47:41 | INFO | mec |   clock: pi1 +0ms · pi2 +0ms
11:47:41 | INFO | mec |   drops: DEADLINE_EXCEEDED×7, WORKER_RESULT_TIMEOUT×1
```

One line every 10 s, not one per frame — at 20 fps per-frame logging is forty
lines a second and buries everything that matters. `MEC_VERBOSE=1` gives per-frame
detail at DEBUG, but **not during a recorded run**: it re-derives every candidate's
estimate and takes the peer lock a second time per frame.

### Reading the network layer's log

```bash
ssh admindesktop1@10.42.0.226 'journalctl -u swarm_net -f'
```

```
[role] scan started — looking for 'NanoNet', will claim after 5s
[role] no network found — nothing after 5s — claiming the access point
[role] role master — access point up, holding 192.168.50.1
master | nano2 44C cpu 18% ram 42% infer 51ms · nano1 41C cpu 9% ram 38% infer 57ms
```

`(mec idle)` on a node means its processing layer is not running, so the master
will not send it work — `mec_active` is how a peer becomes usable rather than
merely present.

---

## A measurement set, start to finish

### Before

1. **Power cycle everything** and re-apply `jetson_clocks`.
2. `./scripts/preflight.sh` — all clear.
3. `./scripts/preflight.sh soak` — **0% loss, max under ~200 ms.** This is the
   gate. Nine minutes now against a wasted forty-five later.
4. `./scripts/preflight.sh roles` — **write down which board is master.**
5. Clear the master's previous results so the report script sees one set:
   ```bash
   ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm/runtime && mkdir -p ../old_runs && \
     mv results_*.csv telemetry_*.csv ../old_runs/ 2>/dev/null; ls'
   ```
6. Rotate the trace file:
   ```bash
   mkdir -p gcs_logs/old && \
   mv gcs_logs/frame_trace.jsonl gcs_logs/old/frame_trace_$(date +%H%M%S).jsonl
   ```

### Set the offered rate on the Pi, not the Nano

```bash
ssh -t admin@10.42.0.31 'sudo systemctl stop pi-sensor && cd ~/pi_sensor && \
  PI_SENSOR_ID=pi1 PI_TARGET_FPS=30 python3 pi_sensor.py'
```

For a two-sensor load point, start both, each with its own id.

### The four runs

Start the GCS and the worker once and **leave both up across all four** so the
worker's engine stays warm.

```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && MEC_SCHED=lyapunov MEC_RUN_FRAMES=5000 python3 mec_node.py'
```
```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && MEC_SCHED=greedy   MEC_RUN_FRAMES=5000 python3 mec_node.py'
```
```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && MEC_SCHED=rr       MEC_RUN_FRAMES=5000 python3 mec_node.py'
```
```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && MEC_SCHED=fixed MEC_OFFLOAD_PCT=50 MEC_RUN_FRAMES=5000 python3 mec_node.py'
```

**~5 minutes of cooldown between runs**, or the last one simply prints the highest
temperature. `MEC_RUN_FRAMES` ends each run by itself — never stop one by hand, or
the four have four different lengths and whichever ran longest looks best for a
reason unrelated to scheduling.

### After each run

```bash
./scripts/preflight.sh roles      # confirm the master did not move mid-set
```

### Collect

```bash
P=LP2; mkdir -p ~/uav_project/$P && \
scp 'admindesktop1@10.42.0.226:~/yolov5/swarm/runtime/results_*.csv' \
    'admindesktop1@10.42.0.226:~/yolov5/swarm/runtime/telemetry_*.csv' \
    'admindesktop1@10.42.0.226:~/yolov5/swarm/runtime/role_events_*.csv' \
    ~/uav_project/$P/ && cp gcs_logs/*.csv ~/uav_project/$P/ && ls ~/uav_project/$P/
```

### Tabulate

```bash
python3 tools/algo_report.py ~/uav_project/LP2 -o ~/uav_project/tables_LP2
```

Read the `── Notes ──` block it prints. It flags anything that would make its own
tables misleading: runs of unequal length, a policy that dropped over 5%, a policy
that offloaded nothing, telemetry that does not overlap the run window.

---

## Capturing a per-frame trace

Start **both** Nanos with tracing on, no `--role`:

```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && MEC_TRACE_N=all python3 mec_node.py'
```
```bash
ssh -t admindesktop@10.42.0.43   'cd ~/yolov5/swarm && MEC_TRACE_N=all python3 mec_node.py'
```

The GCS needs no flag — it reports any frame that arrives carrying stamps, and
appends every one to `gcs_logs/frame_trace.jsonl`.

```bash
python3 shared/frame_trace.py gcs_logs/frame_trace.jsonl
```

**Rotate the file first.** It appends, never truncates, and two runs in one file
blend silently with nothing looking wrong.

**Quote the median with p95 beside it**, not the mean — a single stalled frame
moves the mean and says nothing about the typical frame, while the median-to-p95
gap is the variance a deadline has to survive. For the throughput rows read
**min**, not p95: p95 of a rate is the fast tail, and the worst the link managed is
what matters.

**A negative segment is clock skew, not a slow link.** The reader says so rather
than letting you publish it.

**Leave tracing off for throughput runs.** The Pi's four stamps ride along always
(~120 bytes against a ~31 kB JPEG), but while `MEC_TRACE_N` is set the worker
stamps every frame it is sent and the master keeps a side table.

---

## The failover drill

Measure it **at the laptop**. The GCS stays up throughout and keeps one continuous
clock; the Nanos have no shared time source and one of them is dead for part of the
window.

| Time | Do | Expect |
|---|---|---|
| T+0 | GCS, both Nanos, then the Pis | one MASTER, one WORKER |
| **T+150 s** | **cut power to the master** | GCS logs `feed lost` |
| ~T+165 s | — | the other Nano logs `Network role changed: WORKER → MASTER`; GCS logs `feed resumed` |
| **T+300 s** | **power the dead board back on** | it rejoins as a **worker** |
| T+500 s | end | repeat 5× for a mean and spread |

```bash
cd gcs && python3 -c "
import csv, glob
for path in sorted(glob.glob('gcs_logs/dual_direct_*.csv')):
    lost = None; gaps = []
    for r in csv.DictReader(open(path)):
        if r['Event'] == 'feed lost': lost = float(r['Epoch'])
        elif r['Event'] == 'feed resumed' and lost:
            gaps.append(float(r['Epoch']) - lost); lost = None
    print(f'{path}: ' + (', '.join(f'{g:.1f}s' for g in gaps) if gaps else 'no outage'))
"
```

Cross-check against the Nanos' own `role_events_NNN.csv`, but do not compare
timestamps *between* the two boards — they have no shared clock during the outage.

> **A node pinned with `--role` will not fail over.** Use plain `mec_node.py`.
> `--role master` / `--role worker --master-ip 192.168.50.1` bypasses the network
> layer entirely and is for repeating a run or bench-testing one node.

---

## Changing code on a running rig

```bash
./scripts/deploy.sh all          # verifies every file; restarts nothing
```

Then restart deliberately. **Restart the master last** — restarting it drops the
access point, which takes every other device off the network with it.

```bash
ssh admindesktop@10.42.0.43   'sudo systemctl restart swarm_net mec_node'   # worker first
ssh admin@10.42.0.31          'sudo systemctl restart nanonet-link pi-sensor'
ssh pi02@10.42.0.128          'sudo systemctl restart nanonet-link pi-sensor'
ssh admindesktop1@10.42.0.226 'sudo systemctl restart swarm_net mec_node'   # master last
```

Stopping `swarm_net` does **not** take the network down — it deliberately leaves
the interface as it found it, so a restart will not disconnect the sensors.

Before deploying a scheduler change, run the regression tests locally:

```bash
PYTHONPATH=nano python3 tests/test_scheduler_fixes.py -v
```

Seventeen tests, each stating a defect as a property that must hold. They cover the
reliability floor being an absorbing state, backpressure being charged as failure,
the jitter margin's √depth scaling, drift-plus-penalty offloading under backlog,
forced exploration, and the cold-start trap. **A change that reintroduces one of
those fails here rather than in a flight test.**

---

## Where output lands

| File | Written by | Holds |
|---|---|---|
| `~/yolov5/swarm/runtime/results_<algo>_NNN.csv` | master | per frame: latency, RTT, inference, queue depth, decision cost, drop reason |
| `~/yolov5/swarm/runtime/telemetry_NNN.csv` | master | temperature, CPU, RAM per node, once a second |
| `~/yolov5/swarm/runtime/role_events_NNN.csv` | both Nanos | every network transition |
| `~/yolov5/swarm/runtime/neighbors.json` | `swarm_net` | live role + neighbour table |
| `~/yolov5/swarm/runtime/local_perf.json` | `mec_node` | this node's measured inference time |
| `gcs_logs/dual_direct_NNN.csv` | laptop | events: drops, gaps, feed lost/resumed, worker joined |
| `gcs_logs/dual_buffered_NNN.csv` | laptop | same, from the buffered build |
| `gcs_logs/frame_trace.jsonl` | laptop | per-frame stage timestamps |

All the `NNN` counters auto-increment, so nothing is overwritten. The `fixed`
scheduler puts its ratio in the filename (`results_fixed50_001.csv`) so a sweep at
35/50/65 does not produce three files you cannot tell apart.

---

## Shutdown

Ctrl+C on the processing layers, in any order. It prints the session totals and
closes the CSVs cleanly; rows are flushed as they are written, so nothing is lost
either way.

```bash
for T in admindesktop@10.42.0.43 admindesktop1@10.42.0.226; do
  ssh "$T" 'sudo systemctl stop mec_node 2>/dev/null; pkill -f mec_node.py'; done
```

Leave `swarm_net` running — stopping it does not take the network down, and
starting it again is one less thing to remember.
