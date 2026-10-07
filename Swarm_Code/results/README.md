# Results

Derived tables and a sample trace. **The raw per-frame CSVs are not here** — each
run produces ~200 kB and there are dozens of them across four load points, so they
live outside the repository. What is here is enough to check the arithmetic in
[`../docs/05-results.md`](../docs/05-results.md).

## `tables/`

| File | Load point | Source |
|---|---|---|
| `LP1.csv` | 1 sensor, 15.0-15.9 fps, ρ = 0.43 | `algo_report.py` over the LP1 run set |
| `LP2.csv` | 1 sensor, 29.9 fps, ρ = 0.80 | same, LP2 |
| `LP3.csv` | 1 sensor, 29.8 fps, worker battery-derated | same, LP3 — **round robin run is contaminated** |
| `comparison_2sensor_40fps.csv` | 2 sensors, 39.7-40.0 fps, ρ = 1.07 (LP4) | same, LP4; includes the fixed-ratio baselines |

Regenerate any of them from a run set with:

```bash
python3 ../tools/algo_report.py <dir of results_*.csv and telemetry_*.csv> -o out/
```

**Drops are excluded from every average and reported separately.** A drop has no
latency — it has a reason. Counting it as zero pulls the mean down for whichever
policy dropped most, which *inverts* the ranking.

## `traces/frame_trace.jsonl`

Four traced frames — two processed on the master, two offloaded — with wall-clock
stamps at every stage across all four machines. Read it back with:

```bash
python3 ../shared/frame_trace.py traces/frame_trace.jsonl
```

Five or more frames on a path gives per-segment mean/median/p95/min/max, the queue
depths behind each wait, the throughput each wire segment achieved, and a Time 01
vs Time 02 comparison. Below five it prints the individual breakdown instead, which
is what this sample does.

> This file is a **sample**, not a dataset. The aggregate stage table in
> `../docs/05-results.md` comes from a continuous 2488-frame capture.
>
> `frame_trace.jsonl` is **appended to, never truncated.** Rotate it before every
> capture or two runs blend silently — and nothing looks wrong, because the
> coverage header dedupes by frame id while the totals do not. The tell is
> identical extremes across two supposedly separate runs.
