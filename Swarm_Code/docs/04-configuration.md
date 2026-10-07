# Configuration Reference

Every setting the system reads, where it is read, what it does, and what breaks
if you get it wrong. Nothing here needs a file edited — all of it is environment
variables or one inventory file.

---

## 1. `scripts/hosts.env` — the rig inventory

The only file you edit when a board is re-imaged, renamed, or handed a different
DHCP lease. `deploy.sh` and `preflight.sh` both source it.

| Key | Example | Notes |
|---|---|---|
| `NANO1_SSH` / `NANO2_SSH` | `admindesktop@10.42.0.43` | Administration path. **DHCP — can move between boards.** |
| `NANO1_MAC` / `NANO2_MAC` | `48:b0:2d:f5:dc:3f` | The reliable identifier. `ip neigh \| grep 10.42.0`. |
| `NANO1_NANONET` / `NANO2_NANONET` | `192.168.50.51` / `.55` | Static, set in the NetworkManager client profile. |
| `PI1_SSH` / `PI2_SSH` | `admin@10.42.0.31` | Note Pi 2's login is `pi02`, with a zero. |
| `PI1_SENSOR_ID` / `PI2_SENSOR_ID` | `pi1` / `pi2` | **MUST differ.** See the warning under `PI_SENSOR_ID`. |
| `NANO_DIR` | `~/yolov5/swarm` | **Fixed by the code** — see below. |
| `PI_DIR` | `~/pi_sensor` | Free choice, but the systemd units reference it. |
| `SSID` / `PSK` | `NanoNet` / `nanonet123` | |
| `MASTER_ADDR` | `192.168.50.1` | Holding this address *is* being master. |
| `GCS_ADDR` | `192.168.50.23` | Must match `GCS_IP` in `nano/mec_node.py`. |

> **`NANO_DIR` is not a free choice.** `mec_node.py`, `bench_infer.py` and
> `engine_info.py` all resolve the model as the **parent** of their own directory:
>
> ```python
> HERE      = os.path.dirname(os.path.abspath(__file__))
> YOLO_ROOT = os.path.dirname(HERE)     # expects ~/yolov5/
> ```
>
> Put the code anywhere other than `<yolov5 repo>/swarm/` and the model is looked
> for in the wrong place. The error names both paths it tried.

---

## 2. Three values still hardcoded in source

Everything else is an environment variable. These three are not, because they
change once per deployment and never again.

| File | Constant | Default | Change it when |
|---|---|---|---|
| `nano/mec_node.py` | `GCS_IP` | `"192.168.50.23"` | The laptop's NanoNet address differs. |
| `nano/swarm_net.py` | `NODE_ID_BY_HOST` | `{admindesktop-desktop: nano1, admindesktop1-desktop: nano2}` | Your Jetsons have different hostnames. |
| `nano/swarm_net.py` | `CLAIM_DELAY_BY_NODE` | `{nano2: 5.0, nano1: 30.0}` | You want the other board as the intended master. |
| `gcs/gcs_common.py` | `SENSORS` | `["pi1", "pi2"]` | Different sensor ids. Not a whitelist — an unlisted sensor still gets a pane. |

`SWARM_NODE_ID` overrides the hostname lookup, for a board that has been renamed
or for testing.

---

## 3. Jetson — `mec_node.py`

### Chosen per run

| Variable | Default | Effect |
|---|---|---|
| **`MEC_SCHED`** | `lyapunov` | `lyapunov` \| `greedy` \| `rr` \| `fixed`. A typo **fails at startup** rather than silently running the default. The value appears in the command, the startup log line, and the output filename. |
| **`MEC_RUN_FRAMES`** | `0` (until Ctrl+C) | End the run once this many frames are *accounted for* (delivered + dropped), then shut down through the same path Ctrl+C takes. **Set this for every measurement run.** |
| **`MEC_OFFLOAD_PCT`** | `25` | `fixed` only. Accepts `50` or `0.50`. Out of range 0-100 exits with an error. |
| **`MEC_DEADLINE_MS`** | `1200` | The frame budget. **Causal, not cosmetic** — `enforce_deadline` compares each node's estimated completion against what is left of it, so a smaller number vetoes more offloads and drops infeasible frames sooner. Set the same value on the GCS. |

> **Why `MEC_RUN_FRAMES` and not stopping by hand.** Stopping by hand gives the
> four runs four different lengths, and length is not neutral: a longer run spends
> a larger share of itself in warmed-up steady state, so whichever ran longest
> looks better for a reason that has nothing to do with scheduling.

> **Why the deadline is 1200 ms.** It was 600 ms, calibrated when the master was a
> 51 ms laptop. With a Jetson master the budget is spent before anything can be
> scheduled: the master needs ~190 ms for its own inference and the worker ~190 ms
> plus 65 ms of link, so at 600 ms the worker cannot hold even two frames and
> fails its feasibility check on **every** comparison. 1200 ms is the smallest
> budget both boards can serve, and it is a hardware ceiling on this PoC rather
> than a mission requirement — it suits surveillance and mapping, where a
> one-second-old detection is still useful, and it is **not** a number to reuse
> for collision avoidance.

### Diagnostics

| Variable | Default | Effect |
|---|---|---|
| `MEC_VERBOSE` | off | DEBUG logging **and** the per-frame routing diagnostic. Re-derives every candidate's estimate and takes the peer lock a second time per frame — real work in the loop the results measure. **Not for a recorded run.** |
| `MEC_TRACE_N` | `0` (off) | Frames traced **per path**. A number, or `all`. Use `all` for anything you intend to report. |
| `MEC_TRACE_SKIP` | `100` | Frames to let past before tracing. **Almost never lower this** — see below. |
| `MEC_TRACE_PRINT` | `2` | Full tables rendered per path in the GCS log. Every traced frame still reaches the JSONL. |
| `MEC_NO_CSV` | off | Run with no results CSV. A diagnostic A/B, not a normal mode — the file it suppresses is the run's output. |

> **`MEC_TRACE_SKIP` catches TensorRT warm-up, which is enormous:**
>
> | | first frame | warm |
> |---|---:|---:|
> | master inference | 364.7 ms | 92.2 ms |
> | worker inference | 549 ms | 118.7 ms |
> | master → GCS | 276.9 ms | 38.0 ms |
>
> And the damage spreads: the frame *behind* a cold one inherits the wait. One
> frame recorded a 317 ms "local queue wait" and began inference 0.2 ms after its
> predecessor finished — it was not queued, it was behind a warm-up. Set it to 0
> only when you are deliberately measuring cold start.

### Admission control — for measuring what admission control buys

| Variable | Default | Effect |
|---|---|---|
| `MEC_NO_DEADLINE` | off | Removes the deadline veto from **`rr`**. Also forces `MEC_STRICT_OFFLOAD=1`. |
| `MEC_ENFORCE_DEADLINE` | off | *Adds* the deadline veto to **`fixed`**, which does not have it by default. |
| `MEC_STRICT_OFFLOAD` | `1` for `fixed`, else `0` | Honour the scheduler's choice even when the dispatch path would rather not. |
| `MEC_OUTGOING_MAX` | `10` | Per-worker dispatch queue depth. Raise it only to measure what happens with no admission control at all. |

The gate defaults differ by policy on purpose:

| Policy | Deadline gate | Reasoning |
|---|---|---|
| `lyapunov`, `greedy` | always on | They are trying to make a good per-frame decision and should take the fallback. |
| `rr` | on | Keeps the baseline comparable to the adaptive policies. |
| `fixed` | **off** | "50%" is a *policy*. A gate that silently makes it something else defeats the purpose. |

> **A run with a non-default gate setting must not share a table with the adaptive
> schedulers.** `enforce_deadline` exists so every policy gives up at the same
> point; without it, one policy delivers late frames the others would have
> dropped, so its drop rate and latency tail are on a different footing. Compare
> it against **the same scheduler with the default**, which is a clean
> single-variable experiment: *what does admission control buy under node
> heterogeneity?*

> **What `MEC_STRICT_OFFLOAD` fixes.** Two things downstream of the scheduler can
> turn an offload into a local run without it hearing about it: a full outgoing
> queue, and a send that fails because the worker's socket is blocked. Both are
> right for operating a live system — a delivered frame beats a lost one — and both
> are wrong for a scheduler whose job is to hold a ratio, because the ratio it
> reports is then not the ratio that ran.

---

## 4. Raspberry Pi — `pi_sensor.py`

Set these in `/etc/systemd/system/pi-sensor.service` so they are visible in
`systemctl cat pi-sensor` rather than buried in a shell history.

| Variable | Default | Demo value | Effect |
|---|---|---|---|
| **`PI_SENSOR_ID`** | `pi1` | `pi1` / `pi2` | **The one setting that MUST differ between the boards.** |
| **`PI_TARGET_FPS`** | `20` | `15` | Offered frame rate. The camera is *asked* for this explicitly; whether it was granted is printed at startup and the achieved rate every 10 s. |
| `PI_JPEG_QUALITY` | `80` | `80` | 0-100. See the payload table in [`01-architecture.md`](01-architecture.md#jpeg-q80-not-q95). |
| `PI_STRICT` | `1` | `1` | `0` restores the old `NOBLOCK` send. |
| `PI_MASTER_IP` | `192.168.50.1` | — | Only for exercising the reconnect path against a local endpoint with the rig off. |
| `PI_INGEST_PORT` | `5000` | — | Must match `PI_INGEST_PORT` in `mec_node.py`. |
| `PI_LOCK_PATH` | `/tmp/pi_sensor.lock` | — | Advisory single-instance lock. |

> ### `PI_SENSOR_ID` is the most destructive setting in the system
>
> The master keys `FrameStore` on `"sensor:frame_id"`. Two Pis streaming under the
> same id interleave their sequences in one keyspace and **silently overwrite each
> other's frames.** The run completes, looks entirely plausible, and every number
> in it is wrong.
>
> Once the sensor is a systemd service that is no longer hypothetical — starting
> it by hand out of habit while the service is already running does exactly that.
> Hence the advisory lock on `PI_LOCK_PATH`: the second instance refuses to start.
>
> Note that Pi 2 has three near-identical strings and they are **not**
> interchangeable: the login is `pi02`, the hostname is `pi2`, and the sensor id is
> `pi2`. Only the login carries the zero.

> **Why `PI_TARGET_FPS` is 20 by default and 15 for the demo.** 20 against a master
> needing ~50 ms a frame is 100% of one board and about half the pair, so
> offloading is necessary but the deadline check is not firing constantly. At 30
> the gate was overriding the schedulers often enough that all three converged and
> the comparison stopped discriminating. The service file's `15` is the
> per-deployment override and always wins, so the source keeps its documented
> default.

### The watchdog

| Variable | Default | Effect |
|---|---|---|
| `PI_WATCHDOG` | `1` | `0` disables both timers. |
| `PI_SEND_STALL_SEC` | `10.0` | Nothing delivered for this long, loop still turning → **rebuild the socket in place.** |
| `PI_LOOP_STALL_SEC` | `20.0` | The capture loop has not come round → **exit with code 3** and let systemd restart clean. |

The feed has stopped in the field with the process still running, and restarting
it by hand brought it straight back. Three faults look identical from outside: a
wedged `Picamera2.capture_array()` (no timeout, can block forever), a stale ZMQ
connection, and a blocking send against a zombie AP. Rather than identify which,
the watchdog detects the symptom they share — no progress — and applies the
escalating version of what a human restart does. At 20 fps the loop turns every
50 ms, so 20 s is four hundred missed iterations; nothing normal comes close,
including a strict-mode send blocking for its full 1 s timeout.

---

## 5. Raspberry Pi — `nanonet_link.py`

Runs as **root**, independently of the sensor, and never exits.

| Variable | Default | Effect |
|---|---|---|
| `PI_SSID` | `NanoNet` | |
| `PI_PSK` | `nanonet123` | Only used if the profile is missing entirely. |
| `PI_MASTER_IP` | `192.168.50.1` | The address whose **reachability** defines "the link works". |
| `PI_WIFI_IFACE` | `wlan0` | |
| `PI_LINK_INTERVAL` | `2.0` | Health-check period. Bounds how long the gap is between the AP appearing and the Pi joining. |
| `PI_LINK_STRIKES` | `8` | Failed pings tolerated while still associated → 16 s. Deliberately longer than any planned failover, so it does not add its own recovery time. |
| `PI_LINK_RESTART_CHRONY` | `0` | `1` restarts chrony on recovery. **Leave it off:** a forced restart steps the clock mid-run and corrupts the latency numbers the rig exists to measure. chrony re-establishes its own polls anyway. |

It repairs only settings that are identical on every Pi —
`autoconnect-retries 0`, `autoconnect-priority 100`, `never-default`,
`powersave 2` — and **never touches `ipv4.addresses`**, because that is the one
setting that legitimately differs between the boards and normalising it would hand
both Pis the same address.

---

## 6. Ground station — `gcs.py` / `gcs_buffered.py`

| Variable | Default | Effect |
|---|---|---|
| `MEC_DEADLINE_MS` | `1200` | **Must match the master.** Sets the latency colour thresholds and the jitter-buffer timeout. |

Colours are *derived* from the budget rather than fixed — green below 25% of it,
amber below 60%, red above. They were once fixed at 150/350 ms, chosen when the
budget was 600 ms; against a 1200 ms budget those thresholds painted **71% of a
healthy run red**, and a recording of that reads as a system in permanent failure.

The master now sends its own `deadline_ms` on the wire and the receiver logs a
warning if they disagree, but set both anyway.

`JITTER_BUFFER_TIMEOUT_SEC` is derived as budget + 0.4 s. It **has to exceed** the
budget: a frame the master still considers live is one that can still turn up, and
abandoning it early throws away a frame that was going to arrive and counts it as
lost. It was 1.0 s against a 1.2 s budget, which was wrong.

---

## 7. Compile-time constants worth knowing

Not environment variables. Change them in source, and know why they are what they
are — every one of them was a bug at a different value. Full reasoning is in the
source comments and in [`01-architecture.md`](01-architecture.md).

### `nano/mec_node.py`

| Constant | Value | |
|---|---:|---|
| `TASK_QUEUE_MAX` | 6 | A latency budget, not a buffer. 30 was six seconds of backlog. |
| `LOCAL_DECODE_QUEUE_MAX` | 30 | Still JPEGs — ~38 kB a frame. |
| `LOCAL_INFER_QUEUE_MAX` | 3 | Decoded — ~921 kB a frame. |
| `FRAME_TTL_SEC` | 5.0 | How long a frame waits in the store for a result. |
| `WORKER_RESULT_TIMEOUT_SEC` | 2.0 | Reaper write-off. |
| `SWITCH_MARGIN_SEC` | 0.005 | Greedy hysteresis. |
| `HEARTBEAT_IVL_MS` / `_TIMEOUT_MS` | 1000 / 3000 | ZMTP, so a dead peer is noticed regardless of TCP. |
| `RECONNECT_IVL_MS` / `_MAX_MS` | 200 / **0** | 0 = constant interval, **no backoff**. |

### `nano/completion_time_scheduler.py`

| Constant | Value | |
|---|---:|---|
| `THERMAL_SAFE_C` / `_CRITICAL_C` | 70 / 85 | Were 45/60, below where the hardware throttles, so every node pinned at max penalty and the term cancelled out of every comparison. **At 70 the term is dead code on a rig running 33-42 °C.** |
| `THERMAL_MAX_PENALTY` | 1.5 | |
| `BATTERY_WARN_PCT` / `_CRITICAL_PCT` | 20 / 10 | Only active when a node actually reports a battery; a bench Jetson reports `None` and is correctly unpenalised. |
| `RAM_CRITICAL_PCT` | 95 | Hard exclusion. The one signal timing cannot see coming. |
| `RELIABILITY_FLOOR` | 0.5 | |
| `RELIABILITY_RECOVERY_PER_SEC` | 0.05 | Stops the floor being an absorbing state. |
| `WARMUP_SAMPLES` | 2 | Results kept out of the jitter estimate. |
| `JITTER_CLAMP_FACTOR` | 1.0 | Winsorises each jitter sample. |
| `PROBE_INTERVAL_SEC` | 5.0 | Forced exploration. 0 disables. |
| `STALE_FORCE_SEC` | 30.0 | Probe fires even against an infeasible estimate. |

### `nano/scheduler_lyapunov.py`

| Constant | Value | |
|---|---:|---|
| `DEFAULT_V` | 0.02 s² | The tradeoff knob. **Sweeping it does nothing while the thermal term is zero.** |
| `THERMAL_TARGET_PENALTY` | 1.05 | Time-average budget the virtual queue enforces. |
| `Z_MAX` / `Z_RATE` | 0.5 / 0.05 | Was `Z_MAX=10`, at which Z dominated within seconds of any heat, the sum was ~10 regardless of V, and the tradeoff curve came out flat for a reason unrelated to the tradeoff. |
| `Q_ANCHOR_BETA` | 0.15 | Pull toward the observed backlog, so a wrong `w_i` cannot accumulate into fiction. |
| `MAX_TICK_SEC` | 0.5 | Guards against a stalled thread integrating one enormous timestep. |

### `nano/swarm_net.py`

| Constant | Value | |
|---|---:|---|
| `CLAIM_STAGGER_SEC` | 8.0 | Minimum that works: AP takes 3-5 s to beacon, plus a scan cycle. |
| `RECLAIM_DELAY_SEC` | 5.0 | Failover, not cold boot. Without it failover takes as long as the boot delay. |
| `EMPTY_NETWORK_GRACE_SEC` | **0.0** | Disabled. At 60 s it tore down the only network on the air when the Pis were simply not powered yet. |
| `ASSOCIATION_FAIL_THRESHOLD` | 3 | Consecutive failed polls before a station declares the AP gone. |
| `SCAN_SETTLE_SEC` | 2.0 | A NetworkManager scan takes 1-3 s and the read must follow it, not race it. |
| `PEER_TIMEOUT_SEC` | 3.0 | No telemetry for this long → peer offline. |
| `TELEMETRY_INTERVAL_SEC` | 0.5 | |

### `gcs/gcs.py`

| Constant | Value | |
|---|---:|---|
| `INBOX_DEPTH` | 12 | Thread handoff only, never smoothing. **Oldest discarded** when full. |
| `DISPLAY_TICK_MS` | 16 | |
| `ORDER_QUIET_SEC` | 10 | Rate limit. **Saturates and makes every run look identical** — see [`01-architecture.md`](01-architecture.md#a-trap-in-reading-its-event-log). |
| `PLAUSIBLE_LATENCY_MS` | 60 000 | Above this it is a broken clock, not a slow frame. A master once reported 149 294 645 155 ms. |

---

## 8. Ports

All of these must agree across files. They are constants in three places and the
names match.

| Link | Port | Transport | Set in |
|---|---:|---|---|
| Pi → master | 5000 | TCP (ZMTP) | `mec_node.PI_INGEST_PORT`, `pi_sensor.PI_INGEST_PORT` |
| master → worker | 5001 | TCP (ZMTP) | `mec_node.WORK_PORT` |
| worker → master | 5003 | TCP (ZMTP) | `mec_node.RESULTS_PORT` |
| Nano ↔ Nano telemetry | 5500 | **UDP broadcast** | `swarm_net.TELEMETRY_PORT` |
| master → GCS | 6000 | TCP (ZMTP) | `mec_node.GCS_PORT`, `gcs_common.GCS_PORT` |
| NTP | 123 | UDP | must be open on Nano 2 |
