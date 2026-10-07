# Build This System From Scratch

Everything, in order, from bare boards to a measured result. Every command is
written out. Nothing is left to infer.

**Read this first:** the whole build is about five hours of work, most of it
waiting on `apt` and on one TensorRT engine build. It is arranged so that each
part ends in a check you can actually perform, and **you should not continue past
a failed check** — every one of them exists because continuing past it once cost
a day.

| Part | What it does | Time |
|---|---|---|
| [0](#part-0--what-you-are-building) | Understand the shape of it | 10 min |
| [1](#part-1--hardware-and-what-each-piece-is-for) | Hardware, and why each piece | 5 min |
| [2](#part-2--flash-the-boards) | Flash 2 Jetsons + 2 Pis | 60 min |
| [3](#part-3--the-administration-network) | Laptop shares ethernet to every board | 20 min |
| [4](#part-4--base-software) | Python deps on all five machines | 40 min |
| [5](#part-5--the-model) | YOLOv5n weights → TensorRT engine | 45 min |
| [6](#part-6--nanonet-the-swarm-wi-fi) | The swarm's own Wi-Fi | 30 min |
| [7](#part-7--time-synchronisation) | chrony, Nano 2 as reference | 20 min |
| [8](#part-8--deploy-the-code) | One command | 5 min |
| [9](#part-9--the-services) | systemd on all four boards | 20 min |
| [10](#part-10--first-bring-up) | Power on in order, verify | 20 min |
| [11](#part-11--run-it) | Watch it work | 10 min |
| [12](#part-12--measurement-runs) | The four load points | 3 h |
| [13](#part-13--the-failover-drill) | Kill a board, measure recovery | 30 min |
| [14](#part-14--build-the-result-tables) | CSVs → tables | 10 min |

---

## Part 0 — What you are building

Three tiers. Frames flow left to right; decisions are made in the middle.

```
  TIER 1 — sensing            TIER 2 — edge compute            TIER 3 — display
  ─────────────────           ──────────────────────           ────────────────

  Raspberry Pi 1              Jetson Nano  (MASTER)            Laptop
  camera, 640x480             · owns the Wi-Fi AP              · draws both feeds
  JPEG q80, 15-20 fps  ──┐    · ingests every frame      ┌──►  · logs every event
                         ├───►· decides: here or there? ─┤     · measures recovery
  Raspberry Pi 2       ──┘    · runs YOLOv5n on its GPU  │
  same, id "pi2"              · forwards results ────────┘
                                        │
                                        │ offload
                                        ▼
                              Jetson Nano  (WORKER)
                              · runs YOLOv5n, returns
                                detections as JSON
```

Two things make this more than a pipeline, and they are the project:

**1. The role is claimed, not configured.** Whoever holds `192.168.50.1` is the
master. There is no election, no config flag, no virtual IP. Kill the master and
the other board rebuilds the network, takes that address, and *becomes* the
master — with the sensors reconnecting to the same address they were already
using, because the address belongs to the network rather than to a board.

**2. Four scheduling policies compete on the same hardware.** For every single
frame the master asks "can I finish this sooner than the worker can, including
the round trip?" Four different answers to that question are implemented, chosen
by one environment variable, and measured against each other:

| `MEC_SCHED` | Policy | What it optimises |
|---|---|---|
| `lyapunov` | Drift-plus-penalty | Queue stability traded against thermal cost |
| `greedy` | Earliest completion time | This frame's latency, myopically |
| `rr` | Round robin | Nothing — the naive baseline |
| `fixed` | A ratio you set | Nothing — the *controlled* baseline |

### The two programs on each Jetson, and why they are two

```
  swarm_net.py   NETWORK layer     systemd, starts at boot, runs as root
       │                           forms/heals the Wi-Fi, broadcasts health
       │  writes runtime/neighbors.json  (role + every node's cpu/ram/temp/ip)
       ▼
  mec_node.py    PROCESSING layer  reads its role from that file
       │                           master half or worker half, switches in place
       └─ writes runtime/local_perf.json  (this node's measured inference time)
                 ▲
                 └─ read back by swarm_net.py and folded into the broadcast
```

They talk through **two JSON files, not a socket**, so either can be restarted,
killed, or started first without the other noticing.

The split is not stylistic. Reconfiguring `wlan0` from inside an SSH session
drops that session on the spot, so the network layer cannot be something you run
by hand — and a network that only heals while someone has a terminal open is not
fault tolerance. Meanwhile the processing layer must stay manual for a
measurement run, because that is where `MEC_SCHED` and `MEC_RUN_FRAMES` go.

### The wire

| Link | Port | Transport | Pattern |
|---|---|---|---|
| Pi → master | 5000 | ZMTP over TCP | PUSH → PULL |
| master → worker | 5001 | ZMTP over TCP | PUSH → PULL |
| worker → master | 5003 | ZMTP over TCP | PUSH → PULL |
| Nano ↔ Nano telemetry | 5500 | **raw UDP broadcast** | — |
| master → GCS | 6000 | ZMTP over TCP | PUSH → PULL |
| network ↔ processing layer | — | JSON file, atomic rename | — |

Telemetry is UDP broadcast on purpose: it needs no peer list, so it works
whichever board is master, and it has **no connection state to go stale during a
failover** — which is precisely the problem that bites the TCP links.

---

## Part 1 — Hardware, and what each piece is for

| Qty | Item | Why this and not something else |
|---:|---|---|
| 2 | **NVIDIA Jetson Nano 4 GB** dev kit | The only piece here with a usable GPU. A Pi 4 running YOLOv5n is slower than the network round-trip to a Nano, which would make offloading pointless by construction. |
| 2 | **Raspberry Pi 3 B+ or 4** + CSI camera | Sensing only, deliberately no inference. Cheap, low power, physically near what is being watched. |
| 1 | **Laptop** with Wi-Fi *and* Ethernet | Two jobs: the ground station on Wi-Fi, and the administration path on Ethernet. You need both at once. |
| 2 | microSD ≥ 32 GB, U3 | The Jetsons write results CSVs while inferring; a slow card shows up as latency. |
| 2 | microSD ≥ 16 GB | For the Pis. |
| 2 | **5 V 4 A barrel-jack PSU** + jumper on J48 | Micro-USB cannot feed a Nano at MAXN. Under-powering it looks exactly like thermal throttling in the results. |
| 1 | USB-Ethernet adapter or a small switch | You will want more than one board wired at a time. See Part 3. |
| — | A monitor + keyboard, once | For the Jetsons' first Wi-Fi bring-up, when SSH is not yet reachable. |

**Both Jetsons must support AP mode.** Verify before anything else — if one is
station-only it can never rebuild the network and the entire failover result is
impossible:

```bash
iw list | grep -A 12 "Supported interface modes"
```

You need `* AP` in that output on **both** boards. No software fixes its absence.

### The rig this was built on

Substitute your own values; every one of them lives in one file
(`scripts/hosts.env`) so you change them once.

| Device | Login | Hostname | Wired (admin) | NanoNet (static) | MAC |
|---|---|---|---|---|---|
| Jetson Nano 1 | `admindesktop` | `admindesktop-desktop` | `10.42.0.43` | `.51` client / `.1` as master | `48:b0:2d:f5:dc:3f` |
| Jetson Nano 2 | `admindesktop1` | `admindesktop1-desktop` | `10.42.0.226` | `.55` client / `.1` as master | `48:b0:2d:c1:71:9f` |
| Raspberry Pi 1 | `admin` | `raspberrypi` | `10.42.0.31` | `192.168.50.11` | — |
| Raspberry Pi 2 | `pi02` | `pi2` | `10.42.0.128` | `192.168.50.12` | `b8:27:eb:d2:82:16` |
| Laptop | — | — | `10.42.0.1` | `192.168.50.23` | — |

Three traps in that table, each of which has cost real time:

- **Nano 2's login ends in `1`.** `admindesktop1` is the board numbered 2. The
  trailing digit is off by one from the board number, permanently.
- **Pi 2 has three near-identical names and they are not interchangeable.** The
  login is `pi02` (with a zero), the hostname is `pi2`, and the sensor id it
  streams under is also `pi2`. Only the login carries the zero. Neither Pi uses
  the Raspberry Pi OS default `pi`, and `pi` is rejected on both.
- **The `10.42.0.x` addresses are DHCP and move between boards.** The MAC is the
  only reliable identifier. `48:b0:2d` is a Jetson; `b8:27:eb`, `dc:a6:32` and
  `e4:5f:01` are Raspberry Pi OUIs. Before trusting an address:

  ```bash
  ip neigh | grep 10.42.0
  ```

---

## Part 2 — Flash the boards

### 2.1 Jetson Nano ×2

Use **JetPack 4.6.x** (L4T R32.7.x). This is not a free choice: it is the last
JetPack for the Nano, it is what ships CUDA 10.2 / cuDNN 8.2 / TensorRT 8.2.1.8,
and it is the only line for which prebuilt PyTorch wheels exist for this board.

1. Download the *Jetson Nano Developer Kit SD Card Image* from
   <https://developer.nvidia.com/embedded/downloads> (JetPack 4.6.1 or 4.6.4).
2. Write it with Balena Etcher or:
   ```bash
   sudo dd if=jetson-nano-jp461-sd-card-image.img of=/dev/sdX bs=4M status=progress conv=fsync
   ```
3. First boot **with a monitor and keyboard attached** — you cannot do this
   headless, and you will need the console again in Part 6.
4. Complete the Ubuntu 18.04 setup wizard. **Choose the username deliberately**
   and write it down; it is baked into the systemd unit paths.
5. Set the hostname now, because `swarm_net.py` derives each board's identity
   from it:

   ```bash
   # On the board you will call Nano 1
   sudo hostnamectl set-hostname admindesktop-desktop
   ```
   ```bash
   # On the board you will call Nano 2
   sudo hostnamectl set-hostname admindesktop1-desktop
   ```

   Any two distinct hostnames work, but they must match the table in
   `nano/swarm_net.py`:

   ```python
   NODE_ID_BY_HOST = {
       "admindesktop-desktop":  "nano1",
       "admindesktop1-desktop": "nano2",
   }
   ```

   > **Why identity comes from the hostname.** It used to be a hand-edited
   > literal, `NODE_ID = "nano1"`. That made the file impossible to deploy: every
   > `scp swarm_net.py` to both boards silently gave Nano 2 Nano 1's identity.
   > It happened twice, and the second time both boards booted believing they
   > were nano1, took the same claim delay, and raced for the access point — the
   > exact split brain the staggered delays exist to prevent, reintroduced by the
   > deploy. Nothing copies over a hostname.

6. Enable SSH (already on in JetPack) and confirm it survives a reboot:
   ```bash
   sudo systemctl enable ssh && systemctl is-enabled ssh
   ```

7. **Set the power mode and pin the clocks.** Do this now and re-do it after
   every power cycle:

   ```bash
   sudo nvpmodel -m 0 && sudo jetson_clocks && sudo jetson_clocks --show
   ```

   Expect `MAXN`, four CPU cores at 1479 MHz, GPU at 921.6 MHz.

   > **`jetson_clocks` does not survive a reboot.** After booting, the GPU sits
   > at its 76.8 MHz floor and ramps per frame. That inflates inference badly at
   > low duty cycle — and worst on the board doing the least work, which is
   > exactly the worker whose cost the scheduler is estimating. Re-run it on both
   > boards after every power-on, before any run you intend to quote.

8. Make the OpenBLAS workaround permanent. Skip this and Python dies with no
   output at all:

   ```bash
   echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc && source ~/.bashrc
   ```

   > **What this prevents.** `import numpy`, `import cv2` and `import torch` all
   > abort with `Illegal instruction (core dumped)` — shell exit code **132**
   > (128 + SIGILL) — because OpenBLAS misdetects the Cortex-A57. The process is
   > killed by the CPU, so there is **no traceback and no output whatsoever**,
   > even with `python3 -u`. It reads like a hang, a slow import, or an OOM kill.
   > Note that `~/.bashrc` only covers interactive shells, so
   > `ssh nano 'python3 script.py'` still dies — which is why every Python file
   > in this repo sets the variable itself before importing numpy, and why every
   > systemd unit carries it too.

**Check — on both boards:**

```bash
echo "host=$(hostname) user=$(whoami)"; nvpmodel -q 2>/dev/null | tail -1; \
python3 -c "import numpy, cv2; print('numpy', numpy.__version__, '| cv2', cv2.__version__)"; \
iw list | grep -c '\* AP'
```

You want: your chosen hostname, `MAXN`, both versions printing, and a non-zero
AP count.

### 2.2 Raspberry Pi ×2

Raspberry Pi OS, 64-bit, Bookworm or later. The rig runs Debian trixie with
kernel 6.12 (Pi 1) and 6.18 (Pi 2), which matters for one reason: **chrony 4.x**,
whose `trust` option Part 7 depends on. The Jetsons' Ubuntu 18.04 carries chrony
3.2, so commands that work on a Pi do not necessarily work on a Nano.

1. Flash with Raspberry Pi Imager. In its settings dialog, **before writing**:
   - hostname: `raspberrypi` for Pi 1, `pi2` for Pi 2
   - username: `admin` for Pi 1, `pi02` for Pi 2
   - enable SSH with password authentication
   - set your Wi-Fi country (the radio stays disabled until you do)
2. Boot, then:
   ```bash
   sudo apt update && sudo apt full-upgrade -y && sudo reboot
   ```
3. Make sure SSH survives a reboot — on Raspberry Pi OS it can be installed but
   disabled, which leaves you needing a monitor:
   ```bash
   sudo systemctl enable --now ssh && systemctl is-enabled ssh
   ```
4. Enable the camera and confirm the OS sees it:
   ```bash
   rpicam-hello --list-cameras
   ```

   A working CSI camera prints a device. `No cameras available!` means the ribbon
   is in backwards, seated badly, or the board genuinely has no camera — check
   `/boot/firmware/config.txt` contains `camera_auto_detect=1`, then reseat the
   cable with the board powered off. (This is real: Pi 2 on this rig reported
   `supported=0 detected=0` and an empty i2c bus with the cable visibly in.)

5. **Set the timezone.** A wrong one makes local timestamps look a day off and
   sends you hunting a clock bug that is not there:
   ```bash
   sudo timedatectl set-timezone Asia/Karachi && timedatectl
   ```

   Pi 1 on this rig has **no RTC hardware** (`RTC time: n/a`), so
   `fake-hwclock` is the only thing carrying its clock across a reboot:
   ```bash
   sudo apt install -y fake-hwclock && sudo fake-hwclock save
   ```

**Check — on both Pis:**

```bash
echo "host=$(hostname) user=$(whoami)"; timedatectl | head -3; \
rpicam-hello --list-cameras 2>&1 | head -3; \
python3 -c "import cv2; print('cv2', cv2.__version__)" 2>&1 | tail -1
```

---

## Part 3 — The administration network

Everything in Parts 4-7 needs `apt`, and NanoNet (Part 6) has **no route to the
internet**. So the boards are administered over the laptop's shared Ethernet,
which NATs internet to them, and that path stays in place permanently — it is how
you reach a board whose Wi-Fi configuration you have just broken.

### 3.1 On the laptop

Find your wired interface, then share it:

```bash
nmcli device status
```

```bash
# Replace enp0s31f6 with your wired device name
sudo nmcli connection modify "Wired connection 1" ipv4.method shared && \
sudo nmcli connection up "Wired connection 1" && \
ip -4 addr show enp0s31f6 | grep inet
```

The laptop takes `10.42.0.1` and hands out DHCP leases on `10.42.0.0/24` with
NAT to whatever the laptop itself is using for internet.

### 3.2 Find each board

Plug a board in, then from the laptop:

```bash
ip neigh | grep 10.42.0
```

You get address/MAC pairs. Match the MAC against the table in Part 1 — **do not
assume the address**, because the leases move between boards across reboots.

> **One cable, four boards.** If you only have one Ethernet port, connect the
> boards one at a time for Parts 4-7; nothing in those parts needs two boards
> talking to each other. From Part 6 onward they talk over NanoNet instead, and
> the wired path is only for administration.

### 3.3 First contact

```bash
ssh admindesktop@10.42.0.43 'whoami; hostname; ping -c2 -W3 8.8.8.8 | tail -2'
```

Internet must work through the laptop. If it does not, the laptop's sharing is
not up, or the board has a default route pointing somewhere else (this becomes a
real problem after Part 6 — see the warning there).

**Check:** all four boards answer SSH and all four can reach `8.8.8.8`.

---

## Part 4 — Base software

### 4.1 Both Jetsons

JetPack already ships Python 3.6.9, OpenCV, CUDA, cuDNN and TensorRT. You add
ZeroMQ and the PyTorch stack.

```bash
sudo apt update && sudo apt install -y python3-pip python3-zmq python3-matplotlib \
    libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev chrony iw network-manager
```

**PyTorch.** The Nano needs NVIDIA's own aarch64 wheel; `pip install torch` gets
you an x86 build or nothing. The rig runs **torch 1.8.0 + torchvision 0.9.0a0**,
which is the pairing that works with JetPack 4.6 and the yolov5 revision here.

```bash
# torch 1.8.0 for JetPack 4.6 / Python 3.6
wget -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl \
  https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
python3 -m pip install --user Cython numpy==1.19.4 && \
python3 -m pip install --user torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

> **Verify that link is live before you rely on it.** NVIDIA rehosts these wheels
> and the box.com URLs do rotate. The authoritative index is the "PyTorch for
> Jetson" sticky post on the NVIDIA developer forum — search for it, take the
> JetPack 4.6 / torch 1.8.0 entry, and use whatever URL it currently gives. If
> that fails entirely, Qengineering's Jetson-Nano wheel mirrors on GitHub carry
> the same builds.

torchvision has no wheel and must be built. This takes about 25 minutes:

```bash
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev \
    libavformat-dev libswscale-dev && \
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision && \
cd torchvision && export BUILD_VERSION=0.9.0 && \
python3 setup.py install --user && cd ..
```

**Check — on both Jetsons.** Every line must print, and the CUDA line must say
`True`:

```bash
export OPENBLAS_CORETYPE=ARMV8
python3 -c "
import torch, torchvision, tensorrt, cv2
print('torch      ', torch.__version__)
print('cuda avail ', torch.cuda.is_available())
print('device     ', torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__)
print('tensorrt   ', tensorrt.__version__)
print('cv2        ', cv2.__version__)
"
```

Expected on this rig: torch `1.8.0`, cuda `True`, device `NVIDIA Tegra X1`,
torchvision `0.9.0a0`, tensorrt `8.2.1.8`.

> **If that command prints nothing and exits 132**, `OPENBLAS_CORETYPE` is not
> set. That is the SIGILL from Part 2.7, and it is silent by design.

### 4.2 Both Raspberry Pis

```bash
sudo apt update && sudo apt install -y python3-pip python3-zmq python3-opencv \
    python3-numpy python3-picamera2 network-manager iw chrony fake-hwclock
```

There is a scripted version of this that also handles the failure modes specific
to each board — an upstream that rejects `apt`'s User-Agent, a missing default
route, the camera check:

```bash
# from the repo, on the laptop
scp scripts/pi1_setup.sh admin@10.42.0.31:~/
ssh -t admin@10.42.0.31 './pi1_setup.sh deps'
```
```bash
scp scripts/pi2_setup.sh pi02@10.42.0.128:~/
ssh -t pi02@10.42.0.128 './pi2_setup.sh prep'
```

**Check — on both Pis:**

```bash
python3 -c "
import cv2, zmq, numpy
print('cv2', cv2.__version__, '| zmq', zmq.__version__, '| numpy', numpy.__version__)
try:
    from picamera2 import Picamera2; print('picamera2 present (CSI path)')
except ImportError:
    print('no picamera2 — will fall back to /dev/video via OpenCV')
"
```

### 4.3 The laptop

```bash
sudo apt update && sudo apt install -y python3-pip python3-zmq python3-opencv \
    python3-pil python3-pil.imagetk python3-tk chrony
```

**Check:**

```bash
python3 -c "
import cv2, zmq, tkinter
from PIL import Image, ImageTk
print('cv2', cv2.__version__, '| zmq', zmq.__version__, '| tk ok | PIL ok')
"
```

---

## Part 5 — The model

Both Jetsons need the yolov5 repository and a model file, one directory *above*
where our code goes. `mec_node.py` resolves the model as the parent of its own
directory, so this layout is load-bearing:

```
~/yolov5/                    ← the yolov5 repo: models/, utils/, export.py
├── yolov5n.pt               ← PyTorch weights (fallback)
├── yolov5n.engine           ← TensorRT engine (what actually gets used)
└── swarm/                   ← OUR code goes here
    ├── mec_node.py
    ├── swarm_net.py
    ├── ...
    └── runtime/             ← created at run time: results, telemetry, state
```

### 5.1 Get yolov5 and the weights — on both Jetsons

```bash
cd ~ && git clone https://github.com/ultralytics/yolov5.git && cd yolov5 && \
git checkout v6.2 && \
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt && \
mkdir -p swarm && ls -l yolov5n.pt
```

> **Pin the revision.** `mec_node.py` imports
> `models.common.DetectMultiBackend`, `utils.general.non_max_suppression`,
> `utils.general.scale_coords` and `utils.dataloaders.letterbox`. `scale_coords`
> was renamed `scale_boxes` in later yolov5 releases and `utils.dataloaders` was
> `utils.datasets` in earlier ones, so an unpinned clone breaks the imports in
> one direction or the other. v6.2 is the revision these names are correct for.

Install yolov5's own requirements, minus the ones JetPack already provides:

```bash
cd ~/yolov5 && python3 -m pip install --user \
    "PyYAML>=5.3.1" "tqdm>=4.41.0" "seaborn>=0.11.0" "pandas>=1.1.4" \
    "scipy>=1.4.1" "requests>=2.23.0" onnx
```

### 5.2 Build the TensorRT engine — on each Jetson, natively

**Build it on the board that will run it.** An engine is serialised against a
specific TensorRT build and device; TensorRT warns
`Using an engine plan file across different models of devices is not recommended`
if you move one, and the two boards on this rig report *different* device models
despite both being R32 t210ref with 4 GB.

Stop everything first — the build wants the whole board:

```bash
sudo systemctl stop swarm_net mec_node 2>/dev/null; pkill -f mec_node.py; \
sudo nvpmodel -m 0 && sudo jetson_clocks && free -m | head -2
```

Then build. **This takes 15-25 minutes and looks stalled for long stretches.**
Run it detached so an SSH drop does not kill it:

```bash
cd ~/yolov5 && nohup env OPENBLAS_CORETYPE=ARMV8 python3 export.py \
    --weights yolov5n.pt --include engine --half --imgsz 640 640 \
    --device 0 --workspace 4 > ~/export.log 2>&1 < /dev/null & echo started
```

Watch it:

```bash
tail -f ~/export.log
```

`--half` is FP16 and is worth roughly 2× on this GPU. `--workspace 4` gives
TensorRT 4 GB of tactic-search headroom; on this rig peak usage during the build
was only 564 MiB, so workspace was never the binding constraint, but there is no
reason to leave it tight.

**Check — on both Jetsons:**

```bash
cd ~/yolov5/swarm && python3 engine_info.py
```

(That file arrives in Part 8; run this check again after deploying.) You want
FP16 bindings, input `(1, 3, 640, 640)`, and the same TensorRT version on both
boards. Record the md5 — you will compare it before every measurement run.

### 5.3 Measure each board on its own

Before the network, the scheduler or the queues can confound anything:

```bash
cd ~/yolov5/swarm && python3 bench_infer.py -n 200
```

This prints the board's power mode, pinned clocks, GPU clock through the run,
temperature, and the median/mean/p95 of 200 back-to-back inferences.

**Run it on both boards and write both numbers down.** This is the single most
important number in the project and the one most likely to surprise you:

> **The two boards are not interchangeable.** On this rig, measured with this
> exact command, one board ran YOLOv5n at a **50.9 ms** median and the other at
> **95-103 ms** — a 2× gap under identical power mode, identical pinned CPU and
> GPU clocks, identical TensorRT 8.2.1.8, both FP16 with identical bindings, and
> no thermal throttling on either. The cause was traced to cuDNN convolution
> tactic availability differing between L4T R32.7.1 and R32.7.6; it is not
> fixable without reflashing. **Later runs on the same rig showed both boards at
> ~51 ms, which contradicts that and has not been re-benched** — which is exactly
> why this measurement is a step in the build rather than an assumption.
>
> The consequence for every comparison: **the same board must be master across
> all runs in a set**, and you must record which one it was. With a 2× asymmetry,
> swapping the master silently inverts the result — on this rig, one assignment
> made offloading look like it *saved* 40 ms and the other made it *cost* 79 ms.

---

## Part 6 — NanoNet, the swarm Wi-Fi

One SSID, `NanoNet`, on 2.4 GHz. Whichever Jetson holds `192.168.50.1` is
serving it and is therefore the master.

> ### KEEP A WAY BACK IN
>
> The Nano profiles are created with **`autoconnect no`**, so nothing brings
> `wlan0` up except `swarm_net.py`. If that service fails to start, the board is
> off the Wi-Fi entirely and **SSH over Wi-Fi will not reach it**. Do this part
> with either a monitor and keyboard attached, or the Ethernet from Part 3 still
> plugged in. Part 3's wired path is your lifeline; do not remove it.

### 6.1 Both Jetsons — two profiles each

Both boards get the **same AP definition on purpose**: whichever one ends up
holding it presents an identical network, so the Pis and the laptop reconnect
with no reconfiguration at all. That is what makes the failover invisible to them.

🟣 **Nano 1 — access point:**

```bash
sudo nmcli connection delete NanoNet-ap 2>/dev/null; \
sudo nmcli connection add type wifi ifname wlan0 con-name NanoNet-ap \
  ssid NanoNet autoconnect no \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
  802-11-wireless.powersave 2 \
  ipv4.method shared ipv4.addresses 192.168.50.1/24 \
  ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "nanonet123"
```

🟣 **Nano 1 — client, address `.51`:**

```bash
sudo nmcli connection delete NanoNet-client 2>/dev/null; \
sudo nmcli connection add type wifi ifname wlan0 con-name NanoNet-client \
  ssid NanoNet autoconnect no \
  802-11-wireless.powersave 2 \
  ipv4.method manual ipv4.addresses 192.168.50.51/24 \
  ipv4.never-default yes ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "nanonet123"
```

🔵 **Nano 2 — access point, byte-identical to Nano 1's:**

```bash
sudo nmcli connection delete NanoNet-ap 2>/dev/null; \
sudo nmcli connection add type wifi ifname wlan0 con-name NanoNet-ap \
  ssid NanoNet autoconnect no \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
  802-11-wireless.powersave 2 \
  ipv4.method shared ipv4.addresses 192.168.50.1/24 \
  ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "nanonet123"
```

🔵 **Nano 2 — client, note the `.55`:**

```bash
sudo nmcli connection delete NanoNet-client 2>/dev/null; \
sudo nmcli connection add type wifi ifname wlan0 con-name NanoNet-client \
  ssid NanoNet autoconnect no \
  802-11-wireless.powersave 2 \
  ipv4.method manual ipv4.addresses 192.168.50.55/24 \
  ipv4.never-default yes ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "nanonet123"
```

**`ipv4.never-default yes` on the client profiles is not cosmetic.** NanoNet has
no route to anywhere. A default route over `wlan0` silently kills both the
board's internet and the wired SSH path you administer it over — which is the
same trap that makes `apt` unreachable later.

**Check — on both boards, both must read `no`:**

```bash
nmcli -t -f NAME,AUTOCONNECT connection show | grep NanoNet
```

Both profiles on both boards must say `no`. A `yes` here means boot races the
role state machine and you can get two access points on one SSID.

### 6.2 Both Raspberry Pis — one client profile

The Pis are plain clients and, unlike the Nanos, they **do** autoconnect, because
nothing on them creates the network — they just need to be on it whenever it
exists, including after a master failover they know nothing about.

🟢 **Pi 1 — `.11`:**

```bash
sudo nmcli connection delete NanoNet 2>/dev/null; \
sudo nmcli connection add type wifi ifname wlan0 con-name NanoNet \
  ssid NanoNet autoconnect yes \
  connection.autoconnect-priority 100 \
  connection.autoconnect-retries 0 \
  802-11-wireless.powersave 2 \
  ipv4.method manual ipv4.addresses 192.168.50.11/24 \
  ipv4.never-default yes ipv6.method disabled \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "nanonet123"
```

🟢 **Pi 2 — `.12`, and that is the only difference:**

```bash
sudo nmcli connection delete NanoNet 2>/dev/null; \
sudo nmcli connection add type wifi ifname wlan0 con-name NanoNet \
  ssid NanoNet autoconnect yes \
  connection.autoconnect-priority 100 \
  connection.autoconnect-retries 0 \
  802-11-wireless.powersave 2 \
  ipv4.method manual ipv4.addresses 192.168.50.12/24 \
  ipv4.never-default yes ipv6.method disabled \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "nanonet123"
```

Three settings there each fix a failure that happened on this rig:

- **`connection.autoconnect-retries 0`** — infinite. NetworkManager's default is
  **4**. Nano 2 claims the AP at 5 s and Nano 1 at ~30 s, so a Pi powered on at
  the same time spends its first half-minute looking for an SSID that does not
  exist yet, exhausts all four attempts, and then **sits idle forever** — NM will
  not retry again without an external event. The board looks healthy, the profile
  says `autoconnect yes`, and no frame ever leaves it.
- **`connection.autoconnect-priority 100`** — highest of any saved profile, so
  NanoNet always wins the radio over a home network the Pi still remembers.
- **`802-11-wireless.powersave 2`** — disabled. Power save parks the radio
  between beacons; on this rig it showed up as the link dropping every 20-40 s in
  contiguous 120-140 frame blocks, which reads in the results as the scheduler
  shedding frames.

The running interface only picks up the powersave change on the next activation,
so tell it directly too:

```bash
sudo iw dev wlan0 set power_save off && iw dev wlan0 get power_save
```

### 6.3 The laptop — `.23`

```bash
sudo nmcli connection delete NanoNet 2>/dev/null; \
sudo nmcli connection add type wifi ifname wlp3s0 con-name NanoNet \
  ssid NanoNet autoconnect yes \
  connection.autoconnect-retries 0 \
  802-11-wireless.powersave 2 \
  ipv4.method manual ipv4.addresses 192.168.50.23/24 \
  ipv4.never-default yes ipv6.method disabled \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "nanonet123"
```

Replace `wlp3s0` with your own Wi-Fi device from `nmcli device status`. On
Windows or macOS, join `NanoNet` normally and set a manual IPv4 of
`192.168.50.23` / `255.255.255.0`, **with no gateway and no DNS**.

The laptop is the other end of the master→GCS hop, so its power save matters as
much as the Pis':

```bash
sudo iw dev wlp3s0 set power_save off && iw dev wlp3s0 get power_save
```

**Check.** Nothing can come up until a Nano has built the network, so this check
belongs at the end of Part 10. For now, confirm the profiles exist:

```bash
nmcli -t -f NAME,TYPE,AUTOCONNECT connection show | grep -i nanonet
```

---

## Part 7 — Time synchronisation

Six of the per-frame trace segments span two machines, so they are only as good
as the clock agreement between them. At the ~15 µs this rig holds, a 30 ms
network hop is measured to four significant figures. Without it, those segments
are worthless and can come out **negative**.

**Nano 2 is the reference clock.** Not the laptop. What the measurement needs is
one clock the whole swarm agrees on, not the correct time — every cross-device
figure is a *difference* between two timestamps, so a shared error cancels out.
Nano 2 sits inside NanoNet permanently, which the laptop does not, so it is the
source that is still there when the laptop wanders off.

### 7.1 Install chrony BEFORE joining NanoNet

`apt` needs the archives and NanoNet has no route to them. Do this while each
device is still on the wired path from Part 3.

There is one script for all four devices, one argument each:

```bash
# from the repo, on the laptop — copy it out first
for T in admindesktop@10.42.0.43 admindesktop1@10.42.0.226 \
         admin@10.42.0.31 pi02@10.42.0.128; do
  scp scripts/chrony_setup.sh "$T":~/ ; done
```

```bash
ssh -t admindesktop1@10.42.0.226 './chrony_setup.sh nano2'   # the reference
```
```bash
ssh -t admindesktop@10.42.0.43   './chrony_setup.sh nano1'   # fallback server
```
```bash
ssh -t admin@10.42.0.31          './chrony_setup.sh pi'
```
```bash
ssh -t pi02@10.42.0.128          './chrony_setup.sh pi'
```
```bash
./scripts/chrony_setup.sh laptop      # optional but worth doing
```

The script writes a delimited, replaceable block into `/etc/chrony/chrony.conf`,
so it is safe to re-run.

### 7.2 Why each client carries two `server` lines

Nano 2 has no fixed address: it is `192.168.50.55` as a worker and
`192.168.50.1` once it claims the AP, and `swarm_net.py` swaps those profiles at
runtime. So every client lists **both** and uses whichever answers. Nothing has
to be reconfigured during a failover.

**What settles a disagreement is `trust`, not stratum.** chrony ranks by stratum
only among sources it has already judged truthful. Given two sources that
disagree past tolerance it forms no majority, marks *both* `x`, and syncs to
neither. Observed on this rig: Nano 1 held `.1`, polled itself through that line,
deadlocked against Nano 2 which was **3.66 s** away and fully reachable, and ran
on `local stratum 10` / refid `7F7F0101`. Adding `trust` to the `.55` line fixed
it immediately. chrony 3.2 on Ubuntu 18.04 does support `trust`.

`makestep 0.01 -1` is the other line that matters, and it matters most for boards
with no RTC. By default chrony *slews* the clock, which takes hours to close a
large offset. This steps it immediately, any time, however far out it is.

### 7.3 Let NTP through on Nano 2

A `Reference ID` still reading `00000000` a minute after the network is up almost
always means Nano 2 is dropping UDP 123:

```bash
ssh -t admindesktop1@10.42.0.226 \
  'sudo ufw status | grep -q inactive || sudo ufw allow from 192.168.50.0/24 to any port 123 proto udp'
```

**Check — run on each device once NanoNet is up (after Part 10):**

```bash
./scripts/preflight.sh clocks
```

A healthy client shows `^*` against `192.168.50.55` and `^-` against the
master's address. **`^-` means reachable-but-unused and is correct, not a
fault.** Two `^x` means the deadlock above is back.

Confirm from the server side too — this lists every device currently drawing
time from Nano 2, and each Pi plus Nano 1 should appear:

```bash
ssh -t admindesktop1@10.42.0.226 'sudo chronyc clients'
```

Record the `System time` figure from `chronyc tracking`. Quoting the residual
offset is what makes a validation chapter credible rather than merely plausible.

---

## Part 8 — Deploy the code

Get the repository onto the laptop:

```bash
git clone <your-repo-url> uav-swarm-mec && cd uav-swarm-mec
```

Put your own addresses and logins into one file:

```bash
$EDITOR scripts/hosts.env
```

Then deploy everything, including the systemd units:

```bash
WITH_UNITS=1 ./scripts/deploy.sh all
```

That copies the Jetson payload to `~/yolov5/swarm/` on both Nanos, the Pi payload
to `~/pi_sensor/` on both Pis, substitutes each board's actual login and sensor id
into the unit files, installs them, and runs `daemon-reload`.

Every file is **md5-verified against the local copy after the transfer** and the
script exits non-zero on any mismatch. It also removes `__pycache__` on each
target, because Python only recompiles when the source mtime is newer and `scp`
preserves nothing by default — a stale `.pyc` is how a board kept running the
previous scheduler after its source had been replaced.

**It deliberately does not restart anything.** Deciding when a node may drop out
is yours; the commands are printed at the end.

**Check:**

```bash
./scripts/deploy.sh all | tail -20
```

Every line a `✓`, and `every file verified byte-identical on every target`.

---

## Part 9 — The services

Four units across four boards. Which of them start themselves is a deliberate
split.

| Unit | Board | Starts itself | Why |
|---|---|:---:|---|
| `swarm_net.service` | both Jetsons | **yes** | The network must form on power-on with nobody logged in. |
| `mec_node.service` | both Jetsons | optional | Unattended operation yes; a measurement run needs it **stopped** so you can set `MEC_SCHED` on the command line. |
| `nanonet-link.service` | both Pis | **yes** | Keeps `wlan0` on NanoNet forever, whatever NetworkManager gives up on. |
| `pi-sensor.service` | both Pis | **yes** | The camera feed has no successful terminal state. |

### 9.1 Enable them

```bash
for T in admindesktop@10.42.0.43 admindesktop1@10.42.0.226; do
  ssh -t "$T" 'sudo systemctl enable --now swarm_net && systemctl is-enabled swarm_net swarm_net'
done
```

```bash
for T in admin@10.42.0.31 pi02@10.42.0.128; do
  ssh -t "$T" 'sudo systemctl enable --now nanonet-link pi-sensor && \
               systemctl is-active nanonet-link pi-sensor'
done
```

Leave `mec_node` disabled for now — Part 11 runs it by hand so you can watch it.
Enable it later only for unattended demos:

```bash
# later, for a demo that must survive a reboot with nobody at the keyboard
ssh -t admindesktop1@10.42.0.226 'sudo systemctl enable --now mec_node'
```

### 9.2 What each one does when things go wrong

**`nanonet-link.service`** is the one worth understanding, because it fixes
problems that are invisible from inside the sensor. It runs as root, checks every
2 seconds, and escalates: rescan → bring the connection up → bounce the device →
bounce the radio → restart NetworkManager (at most once per 5 minutes, because
that drops every connection on the board including your SSH).

It probes **reachability of `192.168.50.1`**, not just interface state, because
*associated is not the same as connected*: after a failover a Pi can stay
associated to a radio that is no longer forwarding anything, and NetworkManager
sees a connected device and does nothing. It tolerates 8 consecutive failed pings
(16 s) before forcing a reassociation — comfortably longer than any planned
failover, so it does not add its own recovery time on top of the swarm's.

It deliberately **never touches the profile's IP address**, because that is the
one setting that legitimately differs between the two Pis, and a supervisor that
normalised it would hand both boards the same address.

**`pi-sensor.service`** carries a progress watchdog with two timers, because the
feed has stopped in the field with the process still running:

- **no frame delivered for 10 s but the loop is still turning** → the socket is
  rebuilt in place. Cheap, and invisible to the master beyond a reconnect.
- **the capture loop itself has not come round for 20 s** → the process exits
  with code **3** and systemd restarts it clean. Nothing in-process can fix a
  wedged `Picamera2.capture_array()`, because the thread that would do the fixing
  is the one that is stuck.

It also takes an advisory lock on `/tmp/pi_sensor.lock`, so starting it by hand
while the service is already running **refuses** instead of quietly streaming two
sensors under one id — which would interleave two frame sequences in one
`FrameStore` keyspace, complete the run, and make every number in it wrong.

**Check:**

```bash
for T in admindesktop@10.42.0.43 admindesktop1@10.42.0.226; do
  echo "== $T"; ssh "$T" 'systemctl is-active swarm_net; journalctl -u swarm_net -n 5 --no-pager'
done
```

---

## Part 10 — First bring-up

### 10.1 Power on in this order

**Nano 2 first, alone. Wait 60 seconds. Then everything else.**

Nano 2 claims the access point at 5 s; Nano 1 waits 30 s. Giving Nano 2 a clear
head start means it wins the race deterministically, and it is the board you want
as master — it is also the chrony reference.

```
t=0     power Nano 2
t=5s    Nano 2 has scanned, found nothing, and claimed 192.168.50.1
t=60s   power Nano 1, both Pis, and join the laptop to NanoNet
t=~65s  Nano 1 scans, FINDS NanoNet, joins as a station at .55
```

> **Why the delays are asymmetric.** If both boards boot together with the same
> delay, both scan, both find nothing, and both claim. Two access points answer
> to one SSID, the Pis split between them, and both boards believe they are
> master — with no way to reconcile. The race is symmetric, so the tie-break has
> to break the symmetry; no amount of rescanning helps. 8 s is the minimum that
> works (the AP takes 3-5 s to start beaconing, plus a scan cycle), and 25 s of
> margin is used here because this is the demo configuration and the cost of the
> wrong board holding `.1` in front of an audience is far higher than the cost of
> Nano 1 waiting when it is genuinely alone.

### 10.2 Confirm the roles

```bash
./scripts/preflight.sh roles
```

Exactly one board must report **MASTER — holds 192.168.50.1**. If both do, your
claim delays are equal; if neither does, the AP profile failed to activate.

Watch it happen live:

```bash
ssh admindesktop1@10.42.0.226 'journalctl -u swarm_net -f'
```

```
[role] scan started — looking for 'NanoNet', will claim after 5s
[role] no network found — nothing after 5s — claiming the access point
[role] role master — access point up, holding 192.168.50.1
master | nano2 44C cpu 18% ram 42% · nano1 41C cpu 9% ram 38% (mec idle)
```

### 10.3 Confirm everyone joined

```bash
ssh -t admindesktop1@10.42.0.226 'iw dev wlan0 station dump | grep -c Station'
```

Expect **3** — Nano 1, both Pis — plus the laptop if it has joined, so 4.

```bash
for A in 192.168.50.1 192.168.50.51 192.168.50.55 192.168.50.11 192.168.50.12 192.168.50.23; do
  printf '%-16s ' "$A"; ping -c1 -W2 "$A" >/dev/null 2>&1 && echo up || echo DOWN; done
```

`.55` and `.1` will not both answer — whichever Nano is master holds `.1` and
does not also hold its client address. That is correct.

### 10.4 Confirm the handoff file

```bash
ssh admindesktop1@10.42.0.226 'cat ~/yolov5/swarm/runtime/neighbors.json | python3 -m json.tool'
```

You want `"role": "master"`, and an entry for each Nano with `online: true`.
`mec_active` will be `false` until Part 11 — that field is how the master knows a
peer is available for *work* rather than merely present on the network.

### 10.5 Run the full preflight

```bash
./scripts/preflight.sh
```

Then, before any run whose numbers you intend to quote, the gate:

```bash
./scripts/preflight.sh soak
```

**0% loss and a maximum under about 200 ms.** Any loss, or a maximum in seconds,
means the link problem is still there and the run is wasted. This takes nine
minutes and it is cheaper than the run.

---

## Part 11 — Run it

Start order matters: **the ground station must be listening before the master
connects out to it**, because the master is the one that connects and the GCS is
the one that binds.

### 11.1 The ground station — laptop

```bash
cd gcs && MEC_DEADLINE_MS=1200 python3 gcs.py
```

The window opens with a pane per sensor, both saying `waiting for pi1` /
`waiting for pi2`. There are two builds and the difference is the whole point of
shipping both:

| | Playback | Use it for |
|---|---|---|
| `gcs.py` | **Direct** — every frame drawn on arrival, in arrival order | Lowest achievable display latency. Reports `Gaps`: sequence numbers that never arrived with no drop notice to explain them. |
| `gcs_buffered.py` | **Buffered** — frames held per sensor and shown in capture order | Smooth and correctly ordered, and every frame pays the buffer wait, which is measured and folded into the displayed latency. |

`MEC_DEADLINE_MS` must match the master's. The two drifted apart for weeks — 0.6 s
on the nodes against 1.2 s here — and the on-screen latency colours were tuned
against a budget that had since doubled, painting 71% of a healthy run red. The
master now sends its own value on the wire and the GCS logs a warning if they
disagree, but set both anyway.

### 11.2 The worker — Nano 1

No environment variables; it runs no scheduler. Start it once and **leave it up
across all four measurement runs** so its TensorRT engine stays warm:

```bash
ssh -t admindesktop@10.42.0.43 'cd ~/yolov5/swarm && python3 mec_node.py'
```

### 11.3 The master — Nano 2

```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && python3 mec_node.py'
```

> **Give it 10-30 seconds before it looks alive.** The TensorRT engine loads
> before any role is taken. Wait for `Model ready`, then `Role: MASTER` — that is
> the point it is actually listening. The model then stays resident for the life
> of the process, which is what makes a later role change cost a second or two
> instead of another full load.
>
> `Waiting for the network layer` means `swarm_net` is not running. The
> processing layer waits rather than guessing a role.

You should see, before the model loads:

```
Clock synchronised — 0.4 ms offset
```

If instead you get `CLOCK NOT SYNCHRONISED` or `Clock is 412 ms off the
reference`, **wait and restart it** rather than burning a run. That window is
exactly the one you are tempted to start in: chrony cannot correct anything until
a Nano has built the network and the others have joined.

### 11.4 The sensors — both Pis

They are already running as services. Watch one:

```bash
ssh admin@10.42.0.31 'journalctl -u pi-sensor -f'
```

```
Streaming to 192.168.50.1:5000 — id=pi1 15 FPS, JPEG q80, Picamera2
150 sent | 15.0/15 fps | 0 dropped this interval | 0 total
```

If the achieved rate is below 95% of target the line appends
`** 87% of target`. That is the shortfall being made visible while the run is
happening, rather than inferred from the data afterwards.

For a bounded manual run instead of the service:

```bash
ssh -t admin@10.42.0.31 'sudo systemctl stop pi-sensor && \
  cd ~/pi_sensor && PI_SENSOR_ID=pi1 PI_TARGET_FPS=20 timeout 500 python3 pi_sensor.py'
```

### 11.5 What good looks like

- [ ] Both panes show live video with detection boxes
- [ ] Each pane names the node that processed its last frame — `MASTER` in teal,
      `NANO1` in violet
- [ ] The split bar at the bottom right shows a local/offloaded ratio that moves
- [ ] The master logs a summary every 10 s:

      600 frames | lat avg 78ms p95 121ms | offload 41% | drop 1.4%
        nodes: nano2 72C cpu 61% ram 68% · nano1 68C cpu 44% ram 57%
        clock: pi1 +0ms · pi2 +0ms

- [ ] The Events tab logs `worker joined` when Nano 1's processing layer appears
- [ ] Drop % stays low

> **A low offload percentage is often the correct answer, not a bug.** With one
> camera at 15 fps a single Nano keeps up alone. Sending a frame you could
> finish locally in 50 ms to a node 58 ms away plus a link is a loss. A
> well-behaved scheduler mostly declines. **If you see near-100% offload,
> something is wrong.** To find out *why* a worker is not being used:
>
> ```bash
> MEC_VERBOSE=1 python3 mec_node.py
> ```
>
> That prints, per 10 s, how many frames each worker was `excluded` (a health
> cut: battery, RAM, reliability), `infeasible` (finishable, but not before the
> deadline), `costlier` (available and the scheduler preferred local), or
> `chosen`. Those have different fixes: `costlier` dominant is a tuning problem,
> `infeasible` dominant is a deadline or capacity problem, `excluded` dominant is
> a health-signal problem.

---

## Part 12 — Measurement runs

### 12.1 The rules that make runs comparable

1. **Same board is master across the whole set.** Check with
   `./scripts/preflight.sh roles` before *every* run and write it down.
2. **Same frame count, set by `MEC_RUN_FRAMES`, never by hand.** Stopping by hand
   gives the runs different lengths, and length is not neutral — a longer run
   spends more of itself in warmed-up steady state, so whichever ran longest
   looks best for a reason that has nothing to do with scheduling.
3. **Leave the worker up across all runs** so its engine stays warm.
4. **~5 minutes of cooldown between runs**, or the last one simply prints the
   highest temperature.
5. **Re-pin the clocks after any power cycle** (`sudo jetson_clocks`).
6. **Rotate the trace file before each capture.** `gcs_logs/frame_trace.jsonl` is
   appended to, never truncated:

   ```bash
   mkdir -p gcs_logs/old && \
   mv gcs_logs/frame_trace.jsonl gcs_logs/old/frame_trace_$(date +%H%M%S).jsonl
   ```

   > Two runs in one file blend silently and every aggregate over it is
   > unquotable. Nothing looks wrong — the coverage header still reads
   > `100.0% coverage`, because the Pi restarts its frame ids each session so the
   > header dedupes by id while the totals do not. The tell is **identical
   > extremes across two supposedly separate runs**.

### 12.2 The four policies

The algorithm is chosen by an environment variable, never by editing a file:

```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && \
  MEC_SCHED=lyapunov MEC_RUN_FRAMES=5000 python3 mec_node.py'
```
```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && \
  MEC_SCHED=greedy MEC_RUN_FRAMES=5000 python3 mec_node.py'
```
```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && \
  MEC_SCHED=rr MEC_RUN_FRAMES=5000 python3 mec_node.py'
```
```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && \
  MEC_SCHED=fixed MEC_OFFLOAD_PCT=50 MEC_RUN_FRAMES=5000 python3 mec_node.py'
```

> **Why an environment variable and not an import line.** This used to be three
> imports with two commented out — eight edits across two boards per comparison
> round. An edit that landed on the wrong board produced a run *labelled* as an
> algorithm it did not use, and nothing downstream could detect it because the
> results file looked completely normal. A typo in the variable fails at startup
> instead, and the value appears in the command, the startup log line, and the
> output filename.

Output files auto-number by algorithm, so nothing is overwritten:
`results_lyapunov_001.csv`, `results_greedy_ect_001.csv`, `results_rr_001.csv`,
`results_fixed50_001.csv`. The fixed run carries its ratio in the name, so a
sweep at 35/50/65 does not produce three files you cannot tell apart.

### 12.3 The load points

Utilisation ρ is the offered frame rate divided by the pair's combined service
capacity, measured at ~37 fps on this rig.

| | Sensors | Rate | ρ | What it shows |
|---|---|---|---|---|
| **LP1** | 1 | 15-16 fps | 0.43 | Master has headroom. Greedy correctly declines to offload. |
| **LP2** | 1 | 30 fps | 0.80 | Master saturated. All three deliver every frame; the comparison is entirely about the latency *distribution*. |
| **LP3** | 1 | 30 fps, worker on battery | — | Heterogeneous nodes, k ≈ 1.95. |
| **LP4** | 2 | ~40 fps aggregate | 1.07 | Overload. The phase the whole design exists for. |

Set the rate on the Pi, not on the Nano:

```bash
ssh -t admin@10.42.0.31 'sudo systemctl stop pi-sensor && cd ~/pi_sensor && \
  PI_SENSOR_ID=pi1 PI_TARGET_FPS=30 python3 pi_sensor.py'
```

For LP4, start **both** Pis, each with its own id. Nothing on the Nanos changes:
frames are keyed `sensor_id:frame_id`, so `pi1:1200` and `pi2:1200` cannot
collide, and the master adds unseen sensors on sight. A third camera would work
the same way.

### 12.4 The per-frame timestamp trace

This splits the end-to-end latency of individual frames into every stage across
all four machines, in two shapes:

| | Path | Extra stages |
|---|---|---|
| **Time 01** | processed on the master | local decode queue + master GPU |
| **Time 02** | diverted to the worker | dispatch queue + master→worker + worker decode+inference + worker→master |

Start **both** Nanos with tracing on. No `--role` — omitting it is what makes the
node follow the network layer, which is the whole point of the role being
claimed:

```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm && \
  MEC_TRACE_N=all python3 mec_node.py'
```

`MEC_TRACE_N` is frames kept **per path**: a number, or `all` for every frame
past the warm-up. **Use `all` for anything you intend to report** — one frame per
path is an anecdote, and no reader will average fewer than five. It costs a
~400-byte JSON blob per frame against a ~100 kB JPEG.

`MEC_TRACE_SKIP` defaults to **100 and you almost never want it lower.** Trace
the first frames and you measure TensorRT warming up, not the pipeline:

| | first frame | warm |
|---|---:|---:|
| master inference | 364.7 ms | 92.2 ms |
| worker inference | 549 ms | 118.7 ms |
| master → GCS | 276.9 ms | 38.0 ms |

And the damage spreads — the frame behind a cold one inherits the wait. One frame
recorded a 317 ms "local queue wait" and began inference 0.2 ms after its
predecessor finished: it was not queued, it was behind a warm-up.

Read the trace back:

```bash
python3 shared/frame_trace.py gcs_logs/frame_trace.jsonl
```

With five or more frames on a path you get mean/median/p95/min/max per segment,
the queue depths behind each wait, the payload throughput each wire segment
achieved, and a Time 01 vs Time 02 comparison.

**Quote the median, not the mean, and show p95 next to it.** A single stalled
frame moves the mean and says nothing about the typical one; the gap between
median and p95 is the variance a deadline actually has to survive. Read the
**min** column for throughput, not p95 — p95 of a rate is the fast tail, and the
worst the link managed is what matters.

**A negative segment is not a slow link, it is clock skew**, and the reader says
so rather than letting you publish it.

### 12.5 Collect the output

```bash
P=LP2; mkdir -p ~/uav_project/$P && \
scp 'admindesktop1@10.42.0.226:~/yolov5/swarm/runtime/results_*.csv' \
    'admindesktop1@10.42.0.226:~/yolov5/swarm/runtime/telemetry_*.csv' \
    ~/uav_project/$P/ && \
cp gcs_logs/*.csv ~/uav_project/$P/ && ls -la ~/uav_project/$P/
```

| File | Written by | Holds |
|---|---|---|
| `results_<algo>_NNN.csv` | master | per frame: latency, RTT, inference, queue depth, decision cost, drop reason |
| `telemetry_NNN.csv` | master | temperature, CPU, RAM per node, once a second |
| `role_events_NNN.csv` | both Nanos | every network transition — the failover audit trail |
| `dual_direct_NNN.csv` | laptop | `feed lost` / `feed resumed` / `worker joined` / drops |
| `frame_trace.jsonl` | laptop | per-frame stage timestamps |

Clear the master's `runtime/` between *sets* so the report script only sees one
set:

```bash
ssh -t admindesktop1@10.42.0.226 'cd ~/yolov5/swarm/runtime && mkdir -p ../old_runs && \
  mv results_*.csv telemetry_*.csv ../old_runs/ 2>/dev/null; ls'
```

---

## Part 13 — The failover drill

Measure it **at the laptop, never on a Nano.** The ground station stays up
throughout and keeps one continuous clock; the Nanos have no shared time source
and one of them is dead for part of the window.

| Time | Do this | Expect |
|---|---|---|
| T+0 | Start the GCS, both Nanos, then the Pis | one MASTER, one WORKER |
| **T+150 s** | **Cut power to the master (Nano 2)** | GCS logs `feed lost` |
| ~T+165 s | — | Nano 1 logs `Network role changed: WORKER → MASTER`; GCS logs `feed resumed` |
| **T+300 s** | **Power Nano 2 back on** | it rejoins as a **worker**, not a master |
| T+500 s | Run ends | repeat 5× for a mean and a spread |

Nano 1 uses `RECLAIM_DELAY_SEC = 5.0` rather than its 30 s cold-boot delay, which
is deliberate: the long delay exists to lose a race against the other board at
boot, and coming back after holding a role is not that situation — the network
demonstrably existed a moment ago and has just gone. Without that distinction,
failover would take as long as the boot delay, which is thirty seconds of dead
air every time.

Pull the recovery time straight out of the GCS log:

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

> **A node pinned with `--role` will not fail over.** Use plain `mec_node.py` for
> this drill. `--role master` / `--role worker --master-ip 192.168.50.1` skips the
> network layer entirely and is for repeating a run or bench-testing one node.

---

## Part 14 — Build the result tables

```bash
python3 tools/algo_report.py ~/uav_project/LP2 -o ~/uav_project/tables_LP2
```

That writes `comparison_latency.csv`, `comparison_offload_temp.csv`, and a
per-run `comparison_runs.csv`. Useful flags:

```bash
python3 tools/algo_report.py ~/uav_project/LP2 -o out/ --per-run
python3 tools/algo_report.py ~/uav_project/LP2 -o out/ --only lyapunov,greedy_ect,rr
python3 tools/algo_report.py ~/uav_project/LP2 -o out/ --min-frames 4000
```

How it handles the two things that most easily produce a wrong table:

- **Drops are excluded from every average and reported separately.** A drop has
  no latency — it has a reason. Counting it as zero pulls the mean down for
  whichever algorithm dropped most, which *inverts the ranking*: the
  worst-behaved run prints the best latency.
- **Results and telemetry are paired on time, not on serial number.** They are
  written by different threads with independent counters, so
  `results_rr_002.csv` does not necessarily belong with `telemetry_002.csv`. Each
  results file covers a wall-clock interval and the telemetry rows inside it are
  the ones that describe it. That is also what makes the temperatures honest — a
  maximum over the whole telemetry file would include the idle minutes either
  side, or the previous algorithm's run still cooling down.

The script prints warnings for anything that would make its own tables
misleading: runs of unequal length, a policy that dropped over 5%, a policy that
offloaded nothing, telemetry that does not overlap the run window.

**Then read [`05-results.md`](05-results.md)** for what the measured numbers on
this rig actually were, and in particular for why **drop rate and mean latency
both misrank the schedulers under overload** — which is the main methodological
finding and the reason the results chapter leads with useful yield instead.

---

## When it goes wrong

See [`06-troubleshooting.md`](06-troubleshooting.md) — every symptom that has
actually occurred on this rig, with its cause and its fix.
