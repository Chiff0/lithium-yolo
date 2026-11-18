#!/usr/bin/env python3
"""Optimisation ladder: one row per step of bench/ladder/steps.tsv, with TensorRT
and Darknet as reference rows (runs on the PC).

  bench/.venv/bin/python bench/plot_ladder.py bench/results/<ladder run>

Writes figures/ladder.{png,pdf} and ladder.md (the cumulative table) into the
run directory. Expects a sweep over the ladder_* engines plus the references,
with caps 0 (uncapped) and 30.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REFERENCES = ["tensorrt_fp16", "tensorrt_fp32", "darknet_cudnn_fp16", "darknet_cudnn_fp32"]
STEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ladder", "steps.tsv")
FIRST_GPU = 1


def read_steps():
    steps = []
    with open(STEPS) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                sid, commit, _api, _opts, label = line.rstrip("\n").split("\t")
                steps.append({"id": int(sid), "engine": f"ladder_{sid}", "commit": commit, "label": label})
    return steps


def held(a, cap):
    return a is not None and a["fps"] is not None and a["fps"] >= 0.95 * cap


def figure(steps, agg, idle, outdir):
    rows = [s for s in steps if (s["engine"], 0) in agg]
    n = len(rows)
    y = list(range(n))[::-1]
    fig, (ax_fps, ax_w) = plt.subplots(1, 2, figsize=(11.5, 0.36 * n + 1.9), sharey=True,
                                       gridspec_kw={"width_ratios": [1.35, 1]})
    blue = P.FRAMEWORK["lithium"]

    for yi, s in zip(y, rows):
        a = agg[(s["engine"], 0)]
        ax_fps.barh(yi, a["fps"], height=0.66, color=blue, edgecolor=P.SURFACE, linewidth=2)
        ax_fps.text(a["fps"], yi, f"  {a['fps']:.3g}", va="center", fontsize=8, color=P.INK)

        b = agg.get((s["engine"], 30))
        if held(b, 30):
            ax_w.barh(yi, idle, height=0.66, color=P.IDLE, edgecolor=P.SURFACE, linewidth=2)
            ax_w.barh(yi, b["vdd_in"] - idle, left=idle, height=0.66, color=blue,
                      edgecolor=P.SURFACE, linewidth=2)
            ax_w.text(b["vdd_in"], yi, f"  {b['vdd_in']:.2f}", va="center", fontsize=8, color=P.INK)
        else:
            ax_w.text(0.05, yi, "cannot hold 30 fps", va="center", fontsize=8, color=P.INK2)

    for ref, style in zip(REFERENCES, ["-", "--", "-", "--"]):
        a = agg.get((ref, 0))
        b = agg.get((ref, 30))
        color = P.FRAMEWORK[a["framework"]] if a else None
        if a:
            ax_fps.axvline(a["fps"], color=color, lw=1.4, ls=style, zorder=0)
            ax_fps.text(a["fps"], n - 0.35, f"{a['label']} {a['fps']:.0f}", rotation=90, ha="right",
                        va="top", fontsize=7.5, color=P.INK2)
        if held(b, 30):
            ax_w.axvline(b["vdd_in"], color=P.FRAMEWORK[b["framework"]], lw=1.4, ls=style, zorder=0)

    ax_fps.set_xscale("log")
    ax_fps.set_xlim(left=max(0.3, min(agg[(s["engine"], 0)]["fps"] for s in rows) * 0.6), right=1500)
    ax_fps.set_xlabel("frames per second, uncapped (log)")
    ax_fps.set_title("Throughput per step", fontsize=10)
    ax_fps.set_yticks(y)
    ax_fps.set_yticklabels([f"{s['id']:02d}  {s['label']}" for s in rows], fontsize=8.5)
    ax_fps.tick_params(axis="y", length=0)
    ax_fps.grid(axis="y", visible=False)

    top = max([agg[(s["engine"], 30)]["vdd_in"] for s in rows if held(agg.get((s["engine"], 30)), 30)]
              + [agg[(r, 30)]["vdd_in"] for r in REFERENCES if held(agg.get((r, 30)), 30)])
    ax_w.set_xlim(0, top * 1.18)
    ax_w.set_xlabel("board power at 30 fps, W (grey = idle floor)")
    ax_w.set_title("Power at 30 fps", fontsize=10)
    ax_w.grid(axis="y", visible=False)
    ax_w.tick_params(axis="y", length=0)

    present = {agg[(r, 0)]["framework"] for r in REFERENCES if (r, 0) in agg}
    handles = [plt.Rectangle((0, 0), 1, 1, color=blue)]
    labels = ["lithium, one bar per step"]
    for f in ("tensorrt", "darknet"):
        if f in present:
            handles += [plt.Line2D([], [], color=P.FRAMEWORK[f], lw=1.4),
                        plt.Line2D([], [], color=P.FRAMEWORK[f], lw=1.4, ls="--")]
            labels += [f"{P.FRAMEWORK_NAME[f]} FP16", f"{P.FRAMEWORK_NAME[f]} FP32"]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, -0.02), ncol=5, fontsize=8.5)
    fig.suptitle("lithium optimisation ladder, YOLOv3-tiny 416x416 on Jetson Orin Nano",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    P.save(fig, outdir, "ladder")


def table(steps, agg, meta, path):
    def ms(a):
        return 1000.0 / a["fps"] if a and a["fps"] else None

    trt = agg.get(("tensorrt_fp16", 0))
    first = agg.get((f"ladder_{FIRST_GPU:02d}", 0))
    lines = ["# Optimisation ladder", "",
             f"- device: {meta.get('host')}, {meta.get('power_mode')}, {meta.get('l4t', '')}",
             f"- idle floor {meta['idle']['vdd_in']:.3f} W; {meta['window_s']} s window after "
             f"{meta['settle_s']} s settle, {meta['repeats']} repeat(s)",
             "- every row: input uploaded from pinned host memory, forward incl. YOLO activation, "
             "head outputs downloaded to host memory",
             "",
             "| # | step | commit | ms | fps | × previous | × first GPU | vs TensorRT FP16 | W @30 fps | mJ/frame uncapped |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    prev = None
    for s in steps:
        a = agg.get((s["engine"], 0))
        if a is None:
            lines.append(f"| {s['id']:02d} | {s['label']} | `{s['commit']}` | — | failed | | | | | |")
            continue
        b = agg.get((s["engine"], 30))
        cells = [f"{s['id']:02d}", s["label"], f"`{s['commit']}`", f"{ms(a):.2f}", f"{a['fps']:.1f}",
                 f"{a['fps'] / prev['fps']:.2f}×" if prev else "",
                 f"{a['fps'] / first['fps']:.2f}×" if first and s["id"] >= FIRST_GPU else "",
                 f"{100 * a['fps'] / trt['fps']:.0f}%" if trt else "",
                 f"{b['vdd_in']:.2f}" if held(b, 30) else "—",
                 f"{1000 * a['j_per_frame']:.1f}" if a["j_per_frame"] else ""]
        lines.append("| " + " | ".join(cells) + " |")
        prev = a
    for ref in REFERENCES:
        a = agg.get((ref, 0))
        if a is None:
            continue
        b = agg.get((ref, 30))
        lines.append("| | **" + a["label"] + "** | | " + " | ".join([
            f"{ms(a):.2f}", f"{a['fps']:.1f}", "", "",
            f"{100 * a['fps'] / trt['fps']:.0f}%" if trt else "",
            f"{b['vdd_in']:.2f}" if held(b, 30) else "—",
            f"{1000 * a['j_per_frame']:.1f}" if a["j_per_frame"] else ""]) + " |")
    lines += ["", "— = cannot hold 30 fps", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  {os.path.basename(path)}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = sys.argv[1].rstrip("/")
    rows, meta, _ = P.load(root)
    agg = P.aggregate(rows)
    steps = read_steps()
    outdir = os.path.join(root, "figures")
    os.makedirs(outdir, exist_ok=True)
    figure(steps, agg, meta["idle"]["vdd_in"], outdir)
    table(steps, agg, meta, os.path.join(root, "ladder.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
