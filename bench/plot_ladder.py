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
    # the CPU step (~0.3 fps) stays in the table: on a linear axis it is invisible,
    # and a log axis would hide the size of the GPU steps
    rows = [("step", s["engine"], f"{s['id']:02d}  {s['label']}") for s in steps
            if s["id"] >= FIRST_GPU and (s["engine"], 0) in agg]
    refs = [("ref", r, agg[(r, 0)]["label"]) for r in REFERENCES if (r, 0) in agg]
    entries = rows + refs
    # top to bottom, with a gap between the ladder and the reference rows
    y = [-(i + (0.8 if kind == "ref" else 0.0)) for i, (kind, _, _) in enumerate(entries)]

    fig, (ax_fps, ax_w) = plt.subplots(1, 2, figsize=(11.5, 0.36 * len(entries) + 2.2), sharey=True,
                                       gridspec_kw={"width_ratios": [1.35, 1]})
    for yi, (kind, engine, _) in zip(y, entries):
        a = agg[(engine, 0)]
        color = P.FRAMEWORK[a["framework"]]
        ax_fps.barh(yi, a["fps"], height=0.66, color=color, edgecolor=P.SURFACE, linewidth=2)
        ax_fps.text(a["fps"], yi, f"  {a['fps']:.0f}", va="center", fontsize=8, color=P.INK)

        b = agg.get((engine, 30))
        if held(b, 30):
            ax_w.barh(yi, idle, height=0.66, color=P.IDLE, edgecolor=P.SURFACE, linewidth=2)
            ax_w.barh(yi, b["vdd_in"] - idle, left=idle, height=0.66, color=color,
                      edgecolor=P.SURFACE, linewidth=2)
            ax_w.text(b["vdd_in"], yi, f"  {b['vdd_in']:.2f}", va="center", fontsize=8, color=P.INK)
        else:
            ax_w.text(0.05, yi, "cannot hold 30 fps", va="center", fontsize=8, color=P.INK2)

    top_fps = max(agg[(e, 0)]["fps"] for _, e, _ in entries)
    ax_fps.set_xlim(0, top_fps * 1.12)
    ax_fps.set_xlabel("frames per second, uncapped")
    ax_fps.set_title("Throughput", fontsize=10)
    ax_fps.set_yticks(y)
    ax_fps.set_yticklabels([label for _, _, label in entries], fontsize=8.5)
    for tick, (kind, _, _) in zip(ax_fps.get_yticklabels(), entries):
        if kind == "ref":
            tick.set_fontweight("bold")
    ax_fps.tick_params(axis="y", length=0)
    ax_fps.grid(axis="y", visible=False)

    held_w = [agg[(e, 30)]["vdd_in"] for _, e, _ in entries if held(agg.get((e, 30)), 30)]
    ax_w.set_xlim(0, max(held_w) * 1.15)
    ax_w.set_xlabel("board power at 30 fps, W (grey = idle floor)")
    ax_w.set_title("Power at 30 fps", fontsize=10)
    ax_w.grid(axis="y", visible=False)
    ax_w.tick_params(axis="y", length=0)

    present = {agg[(e, 0)]["framework"] for _, e, _ in entries}
    handles = [plt.Rectangle((0, 0), 1, 1, color=P.FRAMEWORK[f]) for f in P.FRAMEWORK if f in present]
    labels = [P.FRAMEWORK_NAME[f] for f in P.FRAMEWORK if f in present]
    fig.legend(handles + [plt.Rectangle((0, 0), 1, 1, color=P.IDLE)], labels + ["idle floor"],
               loc="lower left", bbox_to_anchor=(0.01, -0.01), ncol=4, fontsize=8.5)
    cpu = agg.get((f"ladder_{0:02d}", 0))
    note = f"step 00 (CPU backend, {cpu['fps']:.2f} fps) is in the table only" if cpu else ""
    fig.text(0.99, 0.005, note, ha="right", va="bottom", fontsize=7.5, color=P.INK2)
    fig.suptitle("lithium optimisation ladder, YOLOv3-tiny 416x416 on Jetson Orin Nano (one bar per commit)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
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
