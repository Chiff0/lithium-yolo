#!/usr/bin/env python3
"""Power/throughput sweep over every engine in engines.py (runs on the Orin).

For each engine and fps cap: start the runner, let it settle, then average the
INA3221 rails and the runner's own fps reports over a fixed window. Energy per
frame is board watts / achieved fps; the marginal figure subtracts the idle
floor measured at the start, so it is the energy the inference itself costs.

  sweep.py --out DIR [--caps 0,30] [--window 15] [--settle 5] [--repeats 1]
           [--only id,id,...] [--skip id,...]

Writes DIR/results.csv, DIR/timeseries.csv and DIR/meta.json.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import platform
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import engines  # noqa: E402


def gpu_temp():
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            with open(f"{zone}/type") as f:
                if f.read().strip().lower().startswith("gpu"):
                    with open(f"{zone}/temp") as t:
                        return int(t.read()) / 1000.0
        except Exception:
            continue
    return None


def other_gpu_work():
    """Anything that would share the GPU with the measurement."""
    out = subprocess.run(["pgrep", "-af", "runner_|lithium_|darknet_c|trt_runner|trtexec|dash.py"],
                         capture_output=True, text=True).stdout
    me = str(os.getpid())
    return [l for l in out.splitlines() if l and not l.startswith(me + " ") and "pgrep" not in l]


def measure_idle(sampler, seconds):
    t0 = time.monotonic()
    time.sleep(seconds)
    rows = sampler.since(t0)
    return {
        "vdd_in": common.mean(r[1] for r in rows),
        "vdd_cpu_gpu_cv": common.mean(r[2] for r in rows),
        "vdd_soc": common.mean(r[3] for r in rows),
        "gpu_mhz": common.mean(r[4] for r in rows),
        "samples": len(rows),
    }


def run_case(sampler, engine, cap, settle, window, series, tag):
    runner = common.Runner(engine, cap)
    try:
        # the runner does its own 20-iteration warmup before reporting
        deadline = time.monotonic() + 120
        while runner.alive() and runner.latest() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if not runner.alive() or runner.latest() is None:
            return None, runner.failure() or "no output within 120 s"
        time.sleep(settle)
        t0 = time.monotonic()
        temp0 = gpu_temp()
        rpm0, pwm0 = common.fan()
        time.sleep(window)
        rpm1, pwm1 = common.fan()
        if not runner.alive():
            return None, runner.failure()
        rows = sampler.since(t0)
        reps = runner.since(t0)
        temp1 = gpu_temp()
    finally:
        runner.stop()

    for r in rows:
        series.append([tag, round(r[0] - t0, 3), round(r[1], 4), round(r[2], 4), round(r[3], 4),
                       r[4] if r[4] is None else round(r[4]), r[5]])
    if not reps:
        return None, "runner reported nothing during the window"
    return {
        "fps": common.mean(r[1] for r in reps),
        "ms": common.mean(r[2] for r in reps),
        "duty": common.mean(r[3] for r in reps),
        "vdd_in": common.mean(r[1] for r in rows),
        "vdd_cpu_gpu_cv": common.mean(r[2] for r in rows),
        "vdd_soc": common.mean(r[3] for r in rows),
        "gpu_mhz": common.mean(r[4] for r in rows),
        "gpu_load": common.mean(r[5] for r in rows),
        "gpu_temp_start": temp0,
        "gpu_temp_end": temp1,
        "fan_rpm": common.mean([rpm0, rpm1]),
        "fan_pwm": common.mean([pwm0, pwm1]),
        "samples": len(rows),
    }, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--caps", default="0,30", help="fps caps, 0 = uncapped")
    ap.add_argument("--window", type=float, default=15.0, help="seconds measured per case")
    ap.add_argument("--settle", type=float, default=5.0, help="seconds after warmup before measuring")
    ap.add_argument("--idle", type=float, default=15.0, help="seconds of idle baseline")
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--force", action="store_true", help="run even if other GPU work is found")
    ap.add_argument("--commit", default="", help="git revision of the code being measured")
    a = ap.parse_args()

    # run_case stops its runner in a finally; make hangup/terminate reach it
    def _bail(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, _bail)
    signal.signal(signal.SIGTERM, _bail)

    busy = other_gpu_work()
    if busy and not a.force:
        print("other GPU work is running, which would contaminate the numbers:")
        for line in busy:
            print("  " + line)
        print("stop it (or pass --force)")
        return 1

    caps = [int(c) for c in a.caps.split(",") if c.strip()]
    only = {s for s in a.only.split(",") if s}
    skip = {s for s in a.skip.split(",") if s}
    todo = [e for e in engines.ENGINES
            if (not only or e["id"] in only) and e["id"] not in skip]
    missing = [e["id"] for e in todo if not engines.available(e)]
    todo = [e for e in todo if engines.available(e)]
    if missing:
        print("not built / files missing, skipping: " + ", ".join(missing))

    os.makedirs(a.out, exist_ok=True)
    sampler = common.Sampler(hz=50).start()
    time.sleep(0.5)
    if sampler.error:
        print(sampler.error)
        return 1

    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "power_mode": common.power_mode(),
        "caps": caps, "window_s": a.window, "settle_s": a.settle, "repeats": a.repeats,
        "hwmon": common.HWMON,
        "missing": missing,
        "commit": a.commit,
        "fan_at_start": dict(zip(("rpm", "pwm"), common.fan())),
    }
    try:
        with open("/etc/nv_tegra_release") as f:
            meta["l4t"] = f.readline().strip()
    except Exception:
        pass

    print(f"power mode {meta['power_mode']}, measuring idle floor ({a.idle:.0f} s)...")
    idle = measure_idle(sampler, a.idle)
    meta["idle"] = idle
    print(f"  idle {idle['vdd_in']:.3f} W board, {idle['vdd_cpu_gpu_cv']:.3f} W compute\n")

    fields = ["engine", "framework", "precision", "label", "caveat", "cap", "repeat",
              "fps", "ms", "duty", "vdd_in", "vdd_cpu_gpu_cv", "vdd_soc",
              "gpu_mhz", "gpu_load", "gpu_temp_start", "gpu_temp_end", "fan_rpm", "fan_pwm",
              "j_per_frame", "j_per_frame_marginal", "compute_j_per_frame", "samples", "error"]
    results = []
    series = []
    total = len(todo) * len(caps) * a.repeats
    n = 0
    hdr = f"{'engine':<26} {'cap':>4} {'fps':>8} {'ms':>8} {'W':>7} {'W cmp':>7} {'mJ/fr':>8} {'mJ marg':>8}"
    print(hdr)
    print("-" * len(hdr))
    for rep in range(a.repeats):
        for eng in todo:
            for cap in caps:
                n += 1
                tag = f"{eng['id']}@{cap}#{rep}"
                r, err = run_case(sampler, eng, cap, a.settle, a.window, series, tag)
                row = {"engine": eng["id"], "framework": eng["framework"],
                       "precision": eng["precision"], "label": eng["label"],
                       "caveat": eng.get("caveat", ""),
                       "cap": cap, "repeat": rep, "error": err}
                if r:
                    row.update(r)
                    fps = r["fps"]
                    if fps > 0:
                        row["j_per_frame"] = r["vdd_in"] / fps
                        row["j_per_frame_marginal"] = (r["vdd_in"] - idle["vdd_in"]) / fps
                        row["compute_j_per_frame"] = r["vdd_cpu_gpu_cv"] / fps
                    print(f"{eng['label']:<26} {cap or 'max':>4} {fps:8.1f} {r['ms']:8.2f} "
                          f"{r['vdd_in']:7.3f} {r['vdd_cpu_gpu_cv']:7.3f} "
                          f"{1000 * row.get('j_per_frame', float('nan')):8.1f} "
                          f"{1000 * row.get('j_per_frame_marginal', float('nan')):8.1f}"
                          f"   [{n}/{total}]", flush=True)
                else:
                    print(f"{eng['label']:<26} {cap or 'max':>4}  FAILED: {err}", flush=True)
                results.append(row)
                time.sleep(a.cooldown)

    sampler.stop()
    with open(f"{a.out}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)
    with open(f"{a.out}/timeseries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "t", "vdd_in", "vdd_cpu_gpu_cv", "vdd_soc", "gpu_mhz", "gpu_load"])
        w.writerows(series)
    with open(f"{a.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {a.out}/results.csv, timeseries.csv, meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
