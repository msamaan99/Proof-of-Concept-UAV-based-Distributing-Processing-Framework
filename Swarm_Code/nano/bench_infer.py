#!/usr/bin/env python3
"""Measure one board's inference speed on its own, with no pipeline attached.

Answers "is this Nano actually slower than the other one?" without the network,
the scheduler, the queues or the GPU clock ramp confounding it. Run the same
command on both boards and compare the medians.

    python3 bench_infer.py                 # 200 frames back-to-back
    python3 bench_infer.py --idle 1.0      # 1 s gap between frames
    python3 bench_infer.py -n 400 --decode # include JPEG decode, like the worker

Why this exists: the pipeline's own numbers are not comparable between the two
roles. The master times `infer_start -> infer_done`, which excludes the JPEG
decode it does in its own stage; the worker times `worker_recv -> worker_done`,
which includes it. And a worker at ~10% duty cycle sits at the 76.8 MHz idle
clock, so every frame pays the ramp back up to 921.6 MHz.

`--idle` reproduces that on demand: run with and without, and the difference is
the clock ramp rather than the hardware.

Everything it prints alongside the timings — backend, power mode, GPU clock,
temperature — exists because "the medians differ" is the start of the question,
not the answer. A board on PyTorch instead of TensorRT, or in a low nvpmodel
mode, looks identical in the pipeline and is obvious here.
"""

import argparse
import os

# Before numpy, cv2 or torch — all three pull in OpenBLAS, whose CPU detection
# misfires on the Jetson's Cortex-A57 and kills the process with SIGILL. The
# failure prints nothing at all: no traceback, no "Illegal instruction", just an
# exit before the first line of output, which reads exactly like the script
# doing nothing.
#
# The Nanos' .bashrc already exports this, but only for interactive shells, so
# `python3 bench_infer.py` typed at the board's own prompt works while
# `ssh nano 'python3 bench_infer.py'` dies silently. Setting it here removes
# that difference — the script behaves the same however it is started.
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import statistics as st
import sys
import time

import cv2
import numpy as np

# Before anything pulls in utils.metrics, which imports pyplot and picks a
# backend on the spot. Over SSH there is no display, so it would land on TkAgg
# and then print a ten-line warning about the backend already being chosen —
# noise in the middle of the output you are trying to read. mec_node.py does the
# same thing for the same reason.
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(HERE)       # the yolov5 repo, one level up

# Must happen before anything imports from the repo. This used to sit inside
# load_model(), which meant `from utils.general import ...` in main() ran first
# and died with "No module named 'utils'" — the repo was never on the path at
# the moment the import was attempted. Path setup belongs at module scope, where
# it cannot be ordered after a dependent import.
if YOLO_ROOT not in sys.path:
    sys.path.append(YOLO_ROOT)


# ── board state ────────────────────────────────────────────────────────────

def _read(path, cast=str, default=None):
    """Read a sysfs value, or return default. Nothing here is worth an exception."""
    try:
        with open(path) as fh:
            return cast(fh.read().strip())
    except Exception:
        return default


def gpu_clock_mhz():
    """Current GPU clock. Tegra exposes this under the gp10b devfreq node."""
    for p in ("/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq",
              "/sys/class/devfreq/57000000.gpu/cur_freq",
              "/sys/kernel/debug/clk/gbus/clk_rate"):
        v = _read(p, int)
        if v:
            return v / 1e6
    return None


def emc_clock_mhz():
    """External memory controller clock.

    Reported because YOLOv5n on a Nano is memory-bandwidth-bound, not
    ALU-bound: the GPU can sit at its full 921.6 MHz and still run at half
    speed if EMC is low. jetson_clocks pins EMC alongside the GPU, so this is
    the clock that quietly differs when two boards look identical everywhere
    else. Most of these paths need root, hence the "needs sudo" fallback.
    """
    for p in ("/sys/kernel/debug/clk/emc/clk_rate",
              "/sys/kernel/debug/clk/override.emc/clk_rate",
              "/sys/kernel/debug/tegra_bwmgr/emc_rate",
              "/sys/kernel/debug/bpmp/debug/clk/emc/rate"):
        v = _read(p, int)
        if v:
            return v / 1e6
    return None


def board_model():
    """The device tree's own name for this board, NUL-padded in sysfs."""
    v = _read("/proc/device-tree/model")
    return v.replace("\x00", "").strip() if v else None


def gpu_temp_c():
    """Warmest thermal zone. Which zone is the GPU varies by L4T version, and
    the hottest one is the one that will throttle first either way."""
    best = None
    for i in range(8):
        v = _read("/sys/devices/virtual/thermal/thermal_zone%d/temp" % i, int)
        if v and 0 < v < 200000:
            c = v / 1000.0
            best = c if best is None else max(best, c)
    return best


def power_mode():
    """nvpmodel mode name, without needing sudo."""
    raw = _read("/etc/nvpmodel.conf")
    cur = _read("/var/lib/nvpmodel/status")          # e.g. "pmode:0000 fmode:fanNull"
    mode = None
    if cur and "pmode:" in cur:
        try:
            mode = int(cur.split("pmode:")[1].split()[0])
        except Exception:
            mode = None
    if mode is None:
        return "unknown"
    name = None
    if raw:
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("< POWER_MODEL") and ("ID=%d" % mode) in line.replace(" ", ""):
                pass
            if line.startswith("< POWER_MODEL") and "ID=" in line and "NAME=" in line:
                try:
                    ident = int(line.split("ID=")[1].split()[0].strip(">"))
                    if ident == mode:
                        name = line.split("NAME=")[1].split()[0].strip(">")
                except Exception:
                    pass
    return "%d (%s)" % (mode, name) if name else str(mode)


def clocks_pinned():
    """True when every online CPU sits at its maximum frequency.

    jetson_clocks pins CPU and GPU together, so the CPUs are a reliable proxy
    and they are readable without sudo. Reported rather than asserted: an
    unpinned board is the single most common reason two identical Nanos
    disagree, and it does not survive a reboot.
    """
    states = []
    for i in range(8):
        base = "/sys/devices/system/cpu/cpu%d/cpufreq/" % i
        cur = _read(base + "scaling_cur_freq", int)
        mx = _read(base + "scaling_max_freq", int)
        mn = _read(base + "scaling_min_freq", int)
        if cur and mx:
            states.append((mn == mx, cur, mx))
    if not states:
        return None, []
    return all(p for p, _, _ in states), states


# ── model ──────────────────────────────────────────────────────────────────

def load_model():
    """Same selection rule as mec_node.load_model — engine if present, else .pt."""
    import torch
    from models.common import DetectMultiBackend

    device = torch.device("cuda:0")
    engine = os.path.join(YOLO_ROOT, "yolov5n.engine")
    weights = os.path.join(YOLO_ROOT, "yolov5n.pt")
    path = engine if os.path.exists(engine) else weights
    if not os.path.exists(path):
        raise SystemExit(
            "No model found. Expected %s or %s" % (engine, weights))
    backend = "TensorRT" if path.endswith(".engine") else "PyTorch"

    t0 = time.time()
    model = DetectMultiBackend(path, device=device, fp16=True)
    model(torch.zeros((1, 3, 640, 640), device=device).half())
    return model, device, path, backend, time.time() - t0


def make_frame():
    """One deterministic synthetic frame, identical on both boards.

    Seeded, so the two boards benchmark byte-identical input and any difference
    in the result is the board.

    Composed rather than random, and the composition is tuned rather than
    arbitrary: a gradient with shapes over it plus sigma-6 noise encodes to
    ~27 KB at quality 75, which is what pi_sensor.py actually puts on the wire
    (median 26.5 KB in the 2026-09-05 trace). Pure noise came out at 115 KB and
    a flat field at 11 KB — either would make `--decode` measure a decode this
    pipeline never performs.
    """
    rng = np.random.RandomState(20260905)
    y, x = np.mgrid[0:480, 0:640].astype(np.float32)
    img = np.zeros((480, 640, 3), np.float32)
    img[..., 0] = 40 + x * 0.18 + y * 0.06
    img[..., 1] = 70 + y * 0.20
    img[..., 2] = 120 - x * 0.10 + y * 0.05
    img = np.clip(img, 0, 255).astype(np.uint8)

    cv2.rectangle(img, (80, 60), (300, 340), (30, 30, 200), -1)
    cv2.rectangle(img, (360, 140), (560, 400), (200, 160, 20), -1)
    cv2.circle(img, (480, 90), 46, (40, 220, 60), -1)
    cv2.putText(img, "MEC", (100, 440), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (250, 250, 250), 4)

    noise = rng.normal(0, 6.0, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def mean(values):
    """Plain arithmetic mean.

    Not statistics.fmean — that is Python 3.8+, and the Jetsons run 3.6.9 on
    Ubuntu 18.04. Anything in this file has to work there, because there is
    where it runs.
    """
    return sum(values) / float(len(values))


def summarise(name, values):
    if not values:
        return
    values = sorted(values)
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    print("  %-22s n=%-5d mean %7.2f   median %7.2f   p95 %7.2f   min %7.2f   max %7.2f"
          % (name, len(values), mean(values), st.median(values),
             p95, values[0], values[-1]))


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark this board's YOLOv5n inference, standalone.")
    ap.add_argument("-n", type=int, default=200, help="frames to time (default 200)")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="seconds to sleep between frames, to force the GPU clock down")
    ap.add_argument("--decode", action="store_true",
                    help="include JPEG encode+decode, matching what the worker times")
    ap.add_argument("--warmup", type=int, default=20,
                    help="untimed frames first (default 20)")
    args = ap.parse_args()

    import torch
    from utils.general import non_max_suppression, scale_coords
    from utils.dataloaders import letterbox

    print("\n\033[1mBoard\033[0m")
    print("  %-22s %s" % ("hostname", os.uname()[1]))
    print("  %-22s %s" % ("model", board_model() or "unreadable"))
    print("  %-22s %s" % ("power mode", power_mode()))
    pinned, states = clocks_pinned()
    if pinned is None:
        print("  %-22s %s" % ("cpu clocks", "unreadable"))
    else:
        freqs = " ".join("%d" % (c // 1000) for _, c, _ in states)
        print("  %-22s %s  (%s MHz)"
              % ("cpu clocks", "PINNED" if pinned else "NOT PINNED — run sudo jetson_clocks",
                 freqs))
    g = gpu_clock_mhz()
    print("  %-22s %s" % ("gpu clock at start", "%.1f MHz" % g if g else "unreadable"))
    e = emc_clock_mhz()
    print("  %-22s %s" % ("emc (memory) clock",
                          "%.1f MHz" % e if e else "needs sudo — see jetson_clocks --show"))
    t = gpu_temp_c()
    print("  %-22s %s" % ("temperature", "%.1f C" % t if t else "unreadable"))

    model, device, path, backend, load_s = load_model()
    print("  %-22s %s" % ("backend", backend))
    print("  %-22s %s" % ("weights", path))
    print("  %-22s %.1f s" % ("load + warmup", load_s))

    img = make_frame()
    jpeg = None
    if args.decode:
        jpeg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])[1].tobytes()
        print("  %-22s %.1f KB" % ("jpeg payload", len(jpeg) / 1024.0))

    def one_frame():
        """Time one frame the way the pipeline does, split by stage.

        Returns (decode, preprocess, model, nms) in ms. Split because a single
        total cannot tell you *what* is slow: the TensorRT engine, the CPU-side
        letterbox and host-to-device copy, or torchvision's NMS. Those have three
        different causes and three different fixes, and on two boards with
        identical clocks the difference has to live in one of them.

        Each stage is synchronised before the next is timed. CUDA calls are
        asynchronous, so without this the model's cost would land in whichever
        stage next touched the GPU rather than in the model.
        """
        dec_ms = 0.0
        frame = img
        if jpeg is not None:
            t_d = time.perf_counter()
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            dec_ms = (time.perf_counter() - t_d) * 1000

        t0 = time.perf_counter()
        fmt = letterbox(frame, new_shape=(640, 640), auto=False)[0]
        fmt = np.ascontiguousarray(fmt.transpose((2, 0, 1))[::-1])
        tensor = torch.from_numpy(fmt).to(device).half() / 255.0
        if len(tensor.shape) == 3:
            tensor = tensor[None]
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        raw = model(tensor)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        pred = non_max_suppression(raw, 0.60, 0.55, max_det=15)
        if len(pred[0]) > 0:
            scale_coords(tensor.shape[2:], pred[0][:, :4], frame.shape)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        return (dec_ms, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)

    print("\n\033[1mRunning\033[0m")
    print("  %d warmup + %d timed frames%s%s"
          % (args.warmup, args.n,
             ", %.2fs idle between" % args.idle if args.idle else ", back-to-back",
             ", with JPEG decode" if args.decode else ""))

    for _ in range(args.warmup):
        one_frame()

    decs, pres, mods, posts, infs, temps, clocks = [], [], [], [], [], [], []
    t_start = time.time()
    for i in range(args.n):
        if args.idle:
            time.sleep(args.idle)
        d, pre, mod, post = one_frame()
        decs.append(d)
        pres.append(pre)
        mods.append(mod)
        posts.append(post)
        infs.append(pre + mod + post)
        if i % 10 == 0:
            c = gpu_clock_mhz()
            if c:
                clocks.append(c)
            tc = gpu_temp_c()
            if tc:
                temps.append(tc)
    elapsed = time.time() - t_start

    print("\n\033[1mResults  (ms)\033[0m")
    if args.decode:
        summarise("jpeg decode", decs)
    summarise("preprocess (CPU)", pres)
    summarise("model (TensorRT)", mods)
    summarise("nms (torchvision)", posts)
    print("  " + "-" * 100)
    summarise("inference total", infs)
    if args.decode:
        summarise("decode + inference", [a + b for a, b in zip(decs, infs)])

    print("\n\033[1mDuring the run\033[0m")
    if clocks:
        print("  %-22s min %.1f   median %.1f   max %.1f MHz"
              % ("gpu clock", min(clocks), st.median(clocks), max(clocks)))
    if temps:
        print("  %-22s start %.1f   end %.1f   max %.1f C"
              % ("temperature", temps[0], temps[-1], max(temps)))
    busy = sum(infs) / 1000.0
    print("  %-22s %.1f s wall, %.1f s inferring (%.0f%% duty)"
          % ("elapsed", elapsed, busy, 100.0 * busy / elapsed if elapsed else 0))
    print("  %-22s %.1f fps\n"
          % ("sustained", args.n / elapsed if elapsed else 0))

    med = st.median(infs)
    if pinned is False:
        print("  \033[33m! CPU clocks are not pinned — run `sudo jetson_clocks` and re-run.\033[0m")
    if backend != "TensorRT":
        print("  \033[33m! Running %s, not TensorRT. Build yolov5n.engine or this board\n"
              "    will be several times slower than one that has it.\033[0m" % backend)
    print("  \033[1mmedian inference %.2f ms\033[0m — compare this number across boards.\n" % med)


if __name__ == "__main__":
    main()
