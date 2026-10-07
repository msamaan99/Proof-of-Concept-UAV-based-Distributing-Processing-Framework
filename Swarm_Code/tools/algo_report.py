#!/usr/bin/env python3
"""Turn a set of measurement runs into the two comparison tables, as CSV.

Reads the raw per-frame files the master writes (`results_<algo>_<nnn>.csv`) and
the once-a-second telemetry beside them (`telemetry_<nnn>.csv`), and produces:

    comparison_latency.csv        total frames, latency, RTT, inference, overhead
    comparison_offload_temp.csv   local/offloaded counts, offload %, temperatures
    comparison_runs.csv           the same numbers per run, before averaging

Usage, from the folder holding the runs you pulled off the master:

    python3 algo_report.py node_dump/runtime
    python3 algo_report.py node_dump/runtime -o tables/
    python3 algo_report.py node_dump/runtime --only lyapunov,greedy_ect,rr,fixed25

Pairing results with telemetry
------------------------------
The two files are written by different threads with independent serial numbers,
so `results_rr_002.csv` does not necessarily go with `telemetry_002.csv`. They
are matched on time instead: each results file covers a wall-clock interval, and
the telemetry rows inside that interval are the ones that describe it. That is
also what makes the temperatures honest — a maximum taken over the whole
telemetry file would include the idle minutes before and after the run, or worse,
the tail of the previous algorithm's run still cooling down.

What is averaged over what
--------------------------
Dropped frames are excluded from every average and reported separately. A drop
has no latency, no inference time and no RTT — it has a reason. Including them
as zeros would pull the mean down for whichever algorithm dropped most, which
inverts the ranking: the worst-behaved run would print the best latency.

`Avg Network RTT` averages over *all delivered frames*, counting locally
processed ones as 0 ms, because a local frame really did cross no network. This
matches the tables you already have. `Avg Offloaded RTT` beside it averages over
offloaded frames only — the actual per-hop cost, undiluted by however many
frames stayed home. Read them as a pair: the first is what the run paid, the
second is what a hop costs.
"""

import argparse
import collections
import csv
import glob
import os
import re
import statistics as st
import sys


RESULTS_RE = re.compile(r"results_(?P<algo>.+)_(?P<serial>\d+)\.csv$")

# Printed in this order when present, so the tables read the way your slides do
# rather than alphabetically. Anything not listed sorts after, by name.
ALGO_ORDER = ["lyapunov", "greedy_ect", "rr"]

PRETTY = {
    "lyapunov": "Lyapunov",
    "greedy_ect": "Greedy (ECT)",
    "rr": "Round Robin",
}


def pretty(algo):
    if algo in PRETTY:
        return PRETTY[algo]
    m = re.fullmatch(r"fixed(\d+)", algo)
    if m:
        return "Fixed %s%%" % m.group(1)
    return algo


def sort_key(algo):
    if algo in ALGO_ORDER:
        return (0, ALGO_ORDER.index(algo), "")
    m = re.fullmatch(r"fixed(\d+)", algo)
    if m:
        return (1, int(m.group(1)), "")
    return (2, 0, algo)


# ── reading ────────────────────────────────────────────────────────────────

def _num(row, key):
    """Parse one cell as a float, or None when it is blank or non-numeric."""
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_results(path):
    """One results file → per-frame lists, already split delivered/dropped."""
    run = {
        "path": path,
        "latency": [], "rtt_all": [], "rtt_offloaded": [],
        "gpu": [], "gpu_local": [], "gpu_offloaded": [],
        "decision": [],
        "local": 0, "offloaded": 0, "dropped": 0,
        "drop_reasons": collections.Counter(),
        "by_worker": collections.Counter(),
        "t_start": None, "t_end": None,
    }

    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            node = (row.get("Assigned Node") or "").strip().upper()
            clock = _num(row, "Wall Clock")
            if clock is not None:
                run["t_start"] = clock if run["t_start"] is None else min(run["t_start"], clock)
                run["t_end"] = clock if run["t_end"] is None else max(run["t_end"], clock)

            # Decision overhead is the one measurement that exists for dropped
            # frames too: the scheduler still ran, and deciding to drop is a
            # decision. Excluding drops here would understate the cost of
            # whichever algorithm spends its time rejecting frames.
            dec = _num(row, "Decision Overhead (ms)")
            if dec is not None:
                run["decision"].append(dec)

            if node == "DROPPED":
                run["dropped"] += 1
                run["drop_reasons"][(row.get("Drop Reason") or "UNKNOWN").strip()] += 1
                continue

            lat = _num(row, "Total Latency (ms)")
            rtt = _num(row, "Network RTT (ms)")
            gpu = _num(row, "GPU Inference (ms)")

            # -1 is the sentinel the master writes when a frame has no latency
            # to report. It is not a measurement and must never reach a mean.
            if lat is not None and lat >= 0:
                run["latency"].append(lat)
            if gpu is not None:
                run["gpu"].append(gpu)
            if rtt is not None:
                run["rtt_all"].append(rtt)

            if node == "MASTER":
                run["local"] += 1
                if gpu is not None:
                    run["gpu_local"].append(gpu)
            else:
                run["offloaded"] += 1
                run["by_worker"][node] += 1
                if gpu is not None:
                    run["gpu_offloaded"].append(gpu)
                if rtt is not None:
                    run["rtt_offloaded"].append(rtt)

    return run


def read_telemetry(paths):
    """Every telemetry row from every file, as (timestamp, node, temperature)."""
    rows = []
    for path in paths:
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    ts = _num(row, "Timestamp")
                    temp = _num(row, "Temperature_C")
                    node = (row.get("Node") or "").strip().lower()
                    if ts is not None and temp is not None and node:
                        rows.append((ts, node, temp))
        except OSError as e:
            print("  ! could not read %s: %s" % (path, e), file=sys.stderr)
    rows.sort()
    return rows


def temps_during(telemetry, t_start, t_end, pad=2.0):
    """{node: [temperatures]} for the window this run occupied.

    `pad` widens the window by a couple of seconds at each end. Telemetry ticks
    once a second on a thread of its own, so a run of a few minutes would
    otherwise lose its first and last sample to rounding — which matters most
    for the maximum, since a thermal peak often lands near the end.
    """
    if t_start is None or t_end is None:
        return {}
    lo, hi = t_start - pad, t_end + pad
    out = collections.defaultdict(list)
    for ts, node, temp in telemetry:
        if lo <= ts <= hi:
            out[node].append(temp)
    return out


# ── summarising ────────────────────────────────────────────────────────────

def mean(values):
    return st.fmean(values) if values else None


def fmt(value, digits=2):
    """Numbers for a table: fixed decimals, and blank rather than a fake zero."""
    return "" if value is None else ("%.*f" % (digits, value))


def summarise(run, telemetry, master_node=None):
    """One run → the flat dict both tables are built from."""
    # Drops were never assigned anywhere, so the offload percentage is a share
    # of the frames that actually went somewhere.
    delivered = run["local"] + run["offloaded"]
    assigned = delivered
    temps = temps_during(telemetry, run["t_start"], run["t_end"])

    # Which node was the master is not in the telemetry — it is the node that
    # wrote this results file, named in the file's own "Node ID" column. Every
    # other node reporting temperature during the window is a worker.
    master_series = temps.get(master_node, [])
    worker_temps = {n: v for n, v in temps.items() if n != master_node}
    # Reported per worker rather than collapsed, so a two-worker run does not
    # hide one board running hot behind an average with a cool one.
    worker_max = {n: max(v) for n, v in worker_temps.items() if v}

    return {
        "algorithm": run["algo"],
        "run": os.path.basename(run["path"]),
        "duration_s": (run["t_end"] - run["t_start"]) if run["t_start"] is not None else None,
        "total_frames": delivered + run["dropped"],
        "delivered": delivered,
        "dropped": run["dropped"],
        "drop_pct": (100.0 * run["dropped"] / (delivered + run["dropped"]))
                    if (delivered + run["dropped"]) else None,
        "avg_latency": mean(run["latency"]),
        "p95_latency": (st.quantiles(run["latency"], n=20)[18]
                        if len(run["latency"]) >= 20 else None),
        "avg_rtt_all": mean(run["rtt_all"]),
        "avg_rtt_offloaded": mean(run["rtt_offloaded"]),
        "avg_gpu": mean(run["gpu"]),
        "avg_gpu_local": mean(run["gpu_local"]),
        "avg_gpu_offloaded": mean(run["gpu_offloaded"]),
        "avg_decision": mean(run["decision"]),
        "local": run["local"],
        "offloaded": run["offloaded"],
        "offload_pct": (100.0 * run["offloaded"] / assigned) if assigned else None,
        "master_avg_temp": mean(master_series),
        "master_max_temp": max(master_series) if master_series else None,
        "worker_max_temp": worker_max,
        "drop_reasons": run["drop_reasons"],
        "by_worker": run["by_worker"],
    }


def combine(summaries):
    """Several runs of one algorithm → one row.

    Frame-weighted, not run-weighted. A 5000-frame run and a 200-frame aborted
    one are not two equal opinions about the same quantity, and averaging their
    averages would treat them as though they were.
    """
    if len(summaries) == 1:
        return dict(summaries[0])

    out = dict(summaries[0])
    out["run"] = "%d runs" % len(summaries)

    def weighted(key, weight_key):
        num = sum(s[key] * s[weight_key] for s in summaries
                  if s[key] is not None and s[weight_key])
        den = sum(s[weight_key] for s in summaries
                  if s[key] is not None and s[weight_key])
        return (num / den) if den else None

    for key in ("total_frames", "delivered", "dropped", "local", "offloaded"):
        out[key] = sum(s[key] for s in summaries)
    for key in ("duration_s",):
        vals = [s[key] for s in summaries if s[key] is not None]
        out[key] = sum(vals) if vals else None

    out["avg_latency"] = weighted("avg_latency", "delivered")
    out["avg_rtt_all"] = weighted("avg_rtt_all", "delivered")
    out["avg_rtt_offloaded"] = weighted("avg_rtt_offloaded", "offloaded")
    out["avg_gpu"] = weighted("avg_gpu", "delivered")
    out["avg_gpu_local"] = weighted("avg_gpu_local", "local")
    out["avg_gpu_offloaded"] = weighted("avg_gpu_offloaded", "offloaded")
    out["avg_decision"] = weighted("avg_decision", "total_frames")
    out["p95_latency"] = None       # not reconstructable from per-run p95s

    out["offload_pct"] = (100.0 * out["offloaded"] / out["delivered"]) if out["delivered"] else None
    out["drop_pct"] = (100.0 * out["dropped"] / out["total_frames"]) if out["total_frames"] else None

    temps = [s["master_avg_temp"] for s in summaries if s["master_avg_temp"] is not None]
    out["master_avg_temp"] = mean(temps)
    maxes = [s["master_max_temp"] for s in summaries if s["master_max_temp"] is not None]
    out["master_max_temp"] = max(maxes) if maxes else None

    merged = collections.defaultdict(list)
    for s in summaries:
        for node, value in s["worker_max_temp"].items():
            merged[node].append(value)
    out["worker_max_temp"] = {n: max(v) for n, v in merged.items()}

    reasons = collections.Counter()
    workers = collections.Counter()
    for s in summaries:
        reasons.update(s["drop_reasons"])
        workers.update(s["by_worker"])
    out["drop_reasons"] = reasons
    out["by_worker"] = workers
    return out


# ── writing ────────────────────────────────────────────────────────────────

def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print("  wrote %s" % path)


def latency_table(rows, worker_names):
    header = ["Algorithm", "Total Frames", "Delivered", "Dropped", "Drop %",
              "Avg Total Latency (ms)", "P95 Total Latency (ms)",
              "Avg Network RTT (ms)", "Avg Offloaded RTT (ms)",
              "Avg GPU Inference (ms)", "Avg Decision Overhead (ms)"]
    out = []
    for r in rows:
        out.append([
            pretty(r["algorithm"]), r["total_frames"], r["delivered"], r["dropped"],
            fmt(r["drop_pct"]), fmt(r["avg_latency"]), fmt(r["p95_latency"]),
            fmt(r["avg_rtt_all"]), fmt(r["avg_rtt_offloaded"]),
            fmt(r["avg_gpu"]), fmt(r["avg_decision"], 4),
        ])
    return header, out


def offload_table(rows, worker_names):
    header = (["Algorithm", "Local Tasks (Master)", "Offloaded", "Offload %",
               "Master Avg Temp (C)", "Master Max Temp (C)"]
              + ["%s Max Temp (C)" % n.upper() for n in worker_names])
    out = []
    for r in rows:
        out.append([
            pretty(r["algorithm"]), r["local"], r["offloaded"], fmt(r["offload_pct"]),
            fmt(r["master_avg_temp"]), fmt(r["master_max_temp"], 1),
        ] + [fmt(r["worker_max_temp"].get(n), 1) for n in worker_names])
    return header, out


def runs_table(rows, worker_names):
    header = (["Algorithm", "Run File", "Duration (s)", "Total Frames", "Delivered",
               "Dropped", "Drop %", "Avg Total Latency (ms)",
               "Avg Network RTT (ms)", "Avg Offloaded RTT (ms)",
               "Avg GPU Inference (ms)", "Avg GPU Local (ms)",
               "Avg GPU Offloaded (ms)", "Avg Decision Overhead (ms)",
               "Local Tasks (Master)", "Offloaded", "Offload %",
               "Master Avg Temp (C)", "Master Max Temp (C)"]
              + ["%s Max Temp (C)" % n.upper() for n in worker_names]
              + ["Top Drop Reason"])
    out = []
    for r in rows:
        top = r["drop_reasons"].most_common(1)
        out.append([
            pretty(r["algorithm"]), r["run"], fmt(r["duration_s"], 1),
            r["total_frames"], r["delivered"], r["dropped"], fmt(r["drop_pct"]),
            fmt(r["avg_latency"]), fmt(r["avg_rtt_all"]), fmt(r["avg_rtt_offloaded"]),
            fmt(r["avg_gpu"]), fmt(r["avg_gpu_local"]), fmt(r["avg_gpu_offloaded"]),
            fmt(r["avg_decision"], 4),
            r["local"], r["offloaded"], fmt(r["offload_pct"]),
            fmt(r["master_avg_temp"]), fmt(r["master_max_temp"], 1),
        ] + [fmt(r["worker_max_temp"].get(n), 1) for n in worker_names]
          + ["%s x%d" % top[0] if top else ""])
    return header, out


def show(header, rows):
    """Print a table wide enough to read in a terminal."""
    cols = [list(map(str, col)) for col in zip(header, *rows)] if rows else []
    widths = [max(len(c) for c in col) for col in cols]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print("\n" + line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build the algorithm comparison tables from raw run CSVs.")
    ap.add_argument("source", nargs="+",
                    help="folder(s) holding results_*.csv and telemetry_*.csv")
    ap.add_argument("-o", "--out", default=".", help="where to write the tables")
    ap.add_argument("--only", help="comma-separated algorithms to include")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="skip runs shorter than this many frames (aborted attempts)")
    ap.add_argument("--per-run", action="store_true",
                    help="one row per run instead of one row per algorithm")
    args = ap.parse_args()

    results_paths, telemetry_paths = [], []
    for src in args.source:
        if os.path.isfile(src):
            results_paths.append(src)
            telemetry_paths += glob.glob(os.path.join(os.path.dirname(src) or ".",
                                                      "telemetry_*.csv"))
            continue
        results_paths += sorted(glob.glob(os.path.join(src, "results_*.csv")))
        telemetry_paths += sorted(glob.glob(os.path.join(src, "telemetry_*.csv")))

    if not results_paths:
        raise SystemExit("No results_*.csv found under: %s" % ", ".join(args.source))

    keep = None
    if args.only:
        keep = {a.strip().lower() for a in args.only.split(",") if a.strip()}

    telemetry = read_telemetry(sorted(set(telemetry_paths)))
    print("Read %d telemetry rows from %d file(s)"
          % (len(telemetry), len(set(telemetry_paths))))

    summaries = []
    for path in results_paths:
        m = RESULTS_RE.search(os.path.basename(path))
        if not m:
            continue
        algo = m.group("algo").lower()
        if keep and algo not in keep:
            continue

        run = read_results(path)
        run["algo"] = algo
        total = run["local"] + run["offloaded"] + run["dropped"]
        if total < args.min_frames:
            print("  skipped %s — %d frames, under --min-frames %d"
                  % (os.path.basename(path), total, args.min_frames))
            continue
        if total == 0:
            print("  skipped %s — no frames" % os.path.basename(path))
            continue

        # The master is the node that wrote the file, named in its own rows.
        master_node = None
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                master_node = (row.get("Node ID") or "").strip().lower()
                if master_node:
                    break

        s = summarise(run, telemetry, master_node)
        s["master_node"] = master_node
        summaries.append(s)
        print("  %-28s %-12s %5d frames  master=%s"
              % (os.path.basename(path), algo, total, master_node or "?"))

    if not summaries:
        raise SystemExit("Nothing to report — every run was filtered out.")

    # Worker columns come from the data, so a run with one worker gets one
    # column and a three-Nano run gets two, without editing this file.
    worker_names = sorted({n for s in summaries for n in s["worker_max_temp"]})

    if args.per_run:
        rows = sorted(summaries, key=lambda s: (sort_key(s["algorithm"]), s["run"]))
    else:
        grouped = collections.defaultdict(list)
        for s in summaries:
            grouped[s["algorithm"]].append(s)
        rows = [combine(v) for _, v in sorted(grouped.items(),
                                              key=lambda kv: sort_key(kv[0]))]

    os.makedirs(args.out, exist_ok=True)

    print("\n── Latency comparison " + "─" * 40)
    h1, r1 = latency_table(rows, worker_names)
    show(h1, r1)
    write_csv(os.path.join(args.out, "comparison_latency.csv"), h1, r1)

    print("\n── Offloading and temperature " + "─" * 32)
    h2, r2 = offload_table(rows, worker_names)
    show(h2, r2)
    write_csv(os.path.join(args.out, "comparison_offload_temp.csv"), h2, r2)

    h3, r3 = runs_table(sorted(summaries, key=lambda s: (sort_key(s["algorithm"]), s["run"])),
                        worker_names)
    write_csv(os.path.join(args.out, "comparison_runs.csv"), h3, r3)

    # Anything that would make the tables misleading if read without it.
    notes = []
    frame_counts = {r["total_frames"] for r in rows}
    if len(frame_counts) > 1:
        notes.append("Runs are not the same length: %s frames. Compare with care — "
                     "run length affects how much of a run is warmed-up steady state."
                     % ", ".join(str(c) for c in sorted(frame_counts)))
    for r in rows:
        if r["dropped"] and r["drop_pct"] and r["drop_pct"] > 5:
            notes.append("%s dropped %.1f%% of frames (%s). Its averages describe "
                         "the frames that survived."
                         % (pretty(r["algorithm"]), r["drop_pct"],
                            ", ".join("%s x%d" % kv
                                      for kv in r["drop_reasons"].most_common(3))))
        if r["offloaded"] == 0:
            notes.append("%s offloaded nothing — was the worker up for that run?"
                         % pretty(r["algorithm"]))
        if not r["worker_max_temp"]:
            notes.append("%s has no worker temperatures — telemetry may not overlap "
                         "the run window." % pretty(r["algorithm"]))
    if notes:
        print("\n── Notes " + "─" * 53)
        for n in notes:
            print("  · " + n)


if __name__ == "__main__":
    main()
