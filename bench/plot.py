#!/usr/bin/env python3
"""Graphs from a sweep.py result directory (runs on the PC).

  bench/.venv/bin/python bench/plot.py bench/results/<stamp>

Writes PNG + PDF figures and summary.md into <stamp>/figures/.
"""

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# reference palette, light mode; the first three slots pass the all-pairs
# colour-vision checks, which is what a scatter needs
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
IDLE = "#c9c8c2"
FRAMEWORK = {"lithium": "#2a78d6", "darknet": "#eb6834", "tensorrt": "#1baf7a"}
FRAMEWORK_NAME = {"lithium": "lithium (ours)", "darknet": "Darknet", "tensorrt": "TensorRT"}
RAILS = [("vdd_cpu_gpu_cv", "VDD_CPU_GPU_CV", "#2a78d6"),
         ("vdd_soc", "VDD_SOC", "#eb6834"),
         ("rest", "rest of board", "#1baf7a")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def load(d):
    with open(f"{d}/results.csv") as f:
        rows = list(csv.DictReader(f))
    with open(f"{d}/meta.json") as f:
        meta = json.load(f)
    series = defaultdict(list)
    ts = f"{d}/timeseries.csv"
    if os.path.exists(ts):
        with open(ts) as f:
            for r in csv.DictReader(f):
                series[r["case"]].append((float(r["t"]), float(r["vdd_in"]), float(r["vdd_cpu_gpu_cv"])))
    return rows, meta, series


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def aggregate(rows):
    """Mean and stdev over repeats, per (engine, cap)."""
    groups = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            groups[(r["engine"], int(r["cap"]))].append(r)
    out = {}
    keys = ["fps", "ms", "vdd_in", "vdd_cpu_gpu_cv", "vdd_soc", "gpu_mhz", "gpu_load",
            "j_per_frame", "j_per_frame_marginal", "compute_j_per_frame"]
    for k, rs in groups.items():
        caveat = rs[0].get("caveat", "")
        a = {"engine": k[0], "cap": k[1], "framework": rs[0]["framework"],
             "label": rs[0]["label"] + (" *" if caveat else ""), "caveat": caveat,
             "precision": rs[0]["precision"], "n": len(rs)}
        for key in keys:
            vals = [num(r.get(key)) for r in rs]
            vals = [v for v in vals if v is not None]
            a[key] = statistics.mean(vals) if vals else None
            a[key + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[k] = a
    return out


def save(fig, outdir, name):
    fig.savefig(f"{outdir}/{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def legend_frameworks(ax, present, extra=()):
    handles = [plt.Rectangle((0, 0), 1, 1, color=FRAMEWORK[f]) for f in FRAMEWORK if f in present]
    labels = [FRAMEWORK_NAME[f] for f in FRAMEWORK if f in present]
    for h, l in extra:
        handles.append(h)
        labels.append(l)
    below(ax, handles, labels)


def below(ax, handles, labels):
    """Legend in one row under the x-axis label, clear of every mark."""
    tr = transforms.offset_copy(ax.transAxes, fig=ax.figure, y=-34, units="points")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, 0), bbox_transform=tr,
              ncol=len(handles), fontsize=8.5, handlelength=1.2, columnspacing=1.4)


def split_offscale(items, key, factor=6.0):
    """Bars more than factor x the median would flatten the rest; pull them out
    and name them in a footnote instead."""
    vals = sorted(a[key] for a in items)
    med = vals[len(vals) // 2]
    keep = [a for a in items if a[key] <= factor * med]
    off = [a for a in items if a[key] > factor * med]
    return (keep, off) if keep else (items, [])


def footnote(fig, items, off=(), key=None, fmt=None):
    lines = []
    if off:
        lines.append("Off scale, not drawn: "
                     + "; ".join(f"{a['label']} {fmt.format(a[key])}" for a in off))
    for a in sorted({a["label"]: a for a in list(items) + list(off) if a.get("caveat")}.values(),
                    key=lambda a: a["label"]):
        lines.append(f"{a['label'].rstrip(' *')} *: {a['caveat']}")
    if lines:
        fig.text(0.01, -0.06, "\n".join(lines), ha="left", va="top", fontsize=8, color=INK2)


def dodge(fig, texts, gap_px=2):
    """Push overlapping point labels down until none collide. texts must be
    ordered top to bottom; each one only moves down past those already placed."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    for t in texts:
        for _ in range(50):
            bb = Text.get_window_extent(t, renderer)  # text only, not the leader line
            hit = next((p for p in placed if bb.overlaps(p)), None)
            if hit is None:
                break
            dy_px = bb.y1 - hit.y0 + gap_px
            x, y = t.get_position()
            t.set_position((x, y - dy_px * 72.0 / fig.dpi))
        placed.append(Text.get_window_extent(t, renderer))


def subtitle(meta, cap):
    rate = "uncapped" if cap == 0 else f"capped at {cap} fps"
    return f"YOLOv3-tiny 416x416, Orin Nano {meta.get('power_mode', '')}, {rate}"


def hbar(ax, items, value, err=None, fmt="{:.0f}", stack_under=None):
    """Horizontal bars, best at the top. stack_under(item) draws a grey idle
    share under the coloured part when given."""
    y = list(range(len(items)))[::-1]
    for yi, it in zip(y, items):
        v = it[value]
        base = 0.0
        if stack_under is not None:
            base = stack_under(it)
            ax.barh(yi, base, height=0.62, color=IDLE, edgecolor=SURFACE, linewidth=2)
        ax.barh(yi, v - base, left=base, height=0.62, color=FRAMEWORK[it["framework"]],
                edgecolor=SURFACE, linewidth=2,
                xerr=(it.get(err) or None) if err else None,
                error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        ax.text(v, yi, "  " + fmt.format(v), va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([it["label"] for it in items])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    xmax = max(it[value] for it in items)
    ax.set_xlim(0, xmax * 1.16)


def fig_throughput(agg, meta, outdir):
    items = sorted([a for (e, c), a in agg.items() if c == 0 and a["fps"]], key=lambda a: -a["fps"])
    if not items:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(items) + 1.2))
    hbar(ax, items, "fps", "fps_sd", fmt="{:.3g}")
    ax.set_xlabel("frames per second (higher is better)")
    ax.set_title("Throughput, forward pass only\n" + subtitle(meta, 0), loc="left")
    legend_frameworks(ax, {a["framework"] for a in items})
    footnote(fig, items)
    save(fig, outdir, "throughput")


def fig_energy(agg, meta, outdir, cap):
    items = [a for (e, c), a in agg.items() if c == cap and a["j_per_frame"]]
    if not items:
        return
    for a in items:
        a["mj"] = 1000 * a["j_per_frame"]
        a["mj_sd"] = 1000 * a["j_per_frame_sd"]
    items.sort(key=lambda a: a["mj"])
    items, off = split_offscale(items, "mj")
    idle = meta["idle"]["vdd_in"]
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(items) + 1.2))
    hbar(ax, items, "mj", "mj_sd", fmt="{:.1f}",
         stack_under=lambda a: 1000 * idle / a["fps"])
    ax.set_xlabel("board energy per frame, mJ (lower is better)")
    ax.set_title("Energy per frame (grey = share of the idle floor)\n" + subtitle(meta, cap), loc="left")
    legend_frameworks(ax, {a["framework"] for a in items},
                      extra=[(plt.Rectangle((0, 0), 1, 1, color=IDLE),
                              f"idle floor share ({idle:.2f} W)")])
    footnote(fig, items, off, "mj", "{:.0f} mJ")
    save(fig, outdir, f"energy_per_frame_{'max' if cap == 0 else cap}")


def fig_power_at_cap(agg, meta, outdir, cap):
    """At a fixed frame rate every engine does the same work, so board watts is the comparison."""
    items = [a for (e, c), a in agg.items() if c == cap and a["vdd_in"] and a["fps"]
             and a["fps"] >= 0.95 * cap]
    if not items:
        return
    items.sort(key=lambda a: a["vdd_in"])
    idle = meta["idle"]["vdd_in"]
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(items) + 1.2))
    hbar(ax, items, "vdd_in", "vdd_in_sd", fmt="{:.2f} W", stack_under=lambda a: idle)
    ax.set_xlabel("board power, W (lower is better)")
    ax.set_title(f"Board power while holding {cap} fps\n" + subtitle(meta, cap)
                 + "\nengines that cannot sustain it are left out", loc="left")
    legend_frameworks(ax, {a["framework"] for a in items},
                      extra=[(plt.Rectangle((0, 0), 1, 1, color=IDLE), f"idle floor ({idle:.2f} W)")])
    footnote(fig, items)
    save(fig, outdir, f"power_at_{cap}fps")


def fig_pareto(agg, meta, outdir):
    items = [a for (e, c), a in agg.items() if c == 0 and a["fps"] and a["j_per_frame"]]
    if not items:
        return
    for a in items:
        a["mj"] = 1000 * a["j_per_frame"]
    items, off = split_offscale(items, "mj")
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(items) + 2.6))
    for a in items:
        ax.scatter(a["fps"], a["mj"], s=46, color=FRAMEWORK[a["framework"]],
                   edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = [a["fps"] for a in items]
    ys = [a["mj"] for a in items]
    ax.set_xlim(min(xs) / 1.25, max(xs) * 1.25)
    ax.set_ylim(min(ys) / 1.25, max(ys) * 1.25)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("frames per second, uncapped (log)")
    ax.set_ylabel("board mJ per frame (log)")
    ax.set_title("Speed vs energy: down and to the right is better\n" + subtitle(meta, 0), loc="left")
    callouts(fig, ax, items)
    handles = [plt.Line2D([], [], marker="o", ls="", color=FRAMEWORK[f], markersize=7)
               for f in FRAMEWORK if any(a["framework"] == f for a in items)]
    ax.legend(handles, [FRAMEWORK_NAME[f] for f in FRAMEWORK if any(a["framework"] == f for a in items)],
              loc="lower left", fontsize=8.5)
    footnote(fig, items, off, "mj", "{:.0f} mJ")
    save(fig, outdir, "speed_vs_energy")


def callouts(fig, ax, items, gap_px=3):
    """Labels in a column right of the plot, each as close to its point's
    height as the column allows, joined to the point by a thin leader."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "Xg", fontsize=7.5, transform=None)
    line_h = probe.get_window_extent(renderer).height + gap_px
    probe.remove()
    box = ax.get_window_extent(renderer)
    pts = sorted(((ax.transData.transform((a["fps"], a["mj"]))[1], a) for a in items),
                 key=lambda p: -p[0])
    ys = []
    for y, _ in pts:   # top to bottom, never closer than one line
        ys.append(min(y, ys[-1] - line_h) if ys else min(y, box.y1))
    overflow = box.y0 - ys[-1] if ys else 0
    if overflow > 0:   # ran off the bottom: lift the whole column, still capped at the top
        ys = [min(y + overflow, box.y1 - i * line_h) for i, y in enumerate(ys)]
    to_axes = ax.transAxes.inverted()
    for y, (_, a) in zip(ys, pts):
        ax.annotate(a["label"], (a["fps"], a["mj"]),
                    xytext=(1.03, to_axes.transform((box.x1, y))[1]), textcoords="axes fraction",
                    va="center", ha="left", fontsize=7.5, color=INK2, annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": IDLE, "lw": 0.7,
                                "shrinkA": 2, "shrinkB": 4})


def fig_rails(agg, meta, outdir):
    items = [a for (e, c), a in agg.items() if c == 0 and a["vdd_in"]]
    if not items:
        return
    items.sort(key=lambda a: a["vdd_in"])
    for a in items:
        a["rest"] = max(0.0, a["vdd_in"] - (a["vdd_cpu_gpu_cv"] or 0) - (a["vdd_soc"] or 0))
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(items) + 1.2))
    y = list(range(len(items)))[::-1]
    for yi, a in zip(y, items):
        left = 0.0
        for key, _, col in RAILS:
            v = a[key] or 0.0
            ax.barh(yi, v, left=left, height=0.62, color=col, edgecolor=SURFACE, linewidth=2)
            left += v
        ax.text(left, yi, f"  {a['vdd_in']:.2f} W", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([a["label"] for a in items])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(a["vdd_in"] for a in items) * 1.16)
    ax.set_xlabel("watts, uncapped")
    ax.set_title("Where the power goes, by INA3221 rail\n" + subtitle(meta, 0), loc="left")
    below(ax, [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in RAILS],
          [n for _, n, _ in RAILS])
    footnote(fig, items)
    save(fig, outdir, "power_by_rail")


def fig_traces(agg, series, meta, outdir):
    cases = [(a, f"{a['engine']}@0#0") for (e, c), a in agg.items() if c == 0]
    cases = [(a, k) for a, k in cases if series.get(k)]
    if not cases:
        return
    order = {"lithium": 0, "darknet": 1, "tensorrt": 2}
    cases.sort(key=lambda x: (order.get(x[0]["framework"], 9), x[0]["label"]))
    cols = 3
    rows = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9.6, 1.75 * rows + 0.8), sharex=True, sharey=True,
                             squeeze=False)
    ymax = max(max(p[1] for p in series[k]) for _, k in cases) * 1.1
    idle = meta["idle"]["vdd_in"]
    for ax, (a, k) in zip(axes.flat, cases):
        pts = series[k]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=FRAMEWORK[a["framework"]], lw=1.6)
        ax.plot([p[0] for p in pts], [p[2] for p in pts], color=INK2, lw=1.0, alpha=0.7)
        ax.axhline(idle, color=IDLE, lw=1, ls="--")
        ax.set_title(a["label"], fontsize=8.5, loc="left")
        ax.set_ylim(0, ymax)
    for i in range(len(cases), rows * cols):
        axes.flat[i].set_visible(False)
        if i - cols >= 0:   # the panel above becomes the bottom of its column
            axes.flat[i - cols].xaxis.set_tick_params(labelbottom=True)
            axes.flat[i - cols].set_xlabel("s into window")
    for ax in axes[:, 0]:
        ax.set_ylabel("W")
    for ax in axes[-1]:
        ax.set_xlabel("s into window")
    fig.suptitle("Power during each measurement window: coloured = VDD_IN (board), "
                 "grey = VDD_CPU_GPU_CV, dashed = idle floor", x=0.01, ha="left", fontsize=9.5,
                 color=INK2)
    fig.tight_layout()
    save(fig, outdir, "power_traces")


def fig_overview(agg, meta, outdir, cap=30):
    """One page: uncapped fps, uncapped board power, energy per frame (power / fps),
    and board power while holding `cap` fps. Every panel lists the engines in the
    same order (fastest first) so a row can be read straight across."""
    base = sorted([a for (e, c), a in agg.items() if c == 0 and a["fps"]], key=lambda a: -a["fps"])
    if not base:
        return
    capped = {a["engine"]: a for (e, c), a in agg.items() if c == cap}
    idle = meta["idle"]["vdd_in"]
    n = len(base)
    y = list(range(n))[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 0.62 * n + 3.4), sharey=True)
    (a_fps, a_w), (a_mj, a_cap) = axes

    def bars(ax, vals, errs, fmt, title, xlabel, idle_share=None, limit=None):
        top = limit if limit else max(v for v in vals if v is not None)
        for yi, it, v, e in zip(y, base, vals, errs):
            if v is None:
                continue
            if limit and v > limit:   # off scale: say so instead of drawing it
                ax.text(0, yi, f" {fmt.format(v)} (off scale)", va="center", fontsize=8.5, color=INK2)
                continue
            left = 0.0
            if idle_share is not None:
                left = min(idle_share(it), v)
                ax.barh(yi, left, height=0.62, color=IDLE, edgecolor=SURFACE, linewidth=2)
            ax.barh(yi, v - left, left=left, height=0.62, color=FRAMEWORK[it["framework"]],
                    edgecolor=SURFACE, linewidth=2, xerr=e or None,
                    error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
            ax.text(v + (e or 0), yi, "  " + fmt.format(v), va="center", fontsize=8.5)
        ax.set_xlim(0, top * 1.22)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)

    bars(a_fps, [a["fps"] for a in base], [a["fps_sd"] for a in base], "{:.3g}",
         "Throughput, uncapped", "frames per second (higher is better)")
    bars(a_w, [a["vdd_in"] for a in base], [a["vdd_in_sd"] for a in base], "{:.2f} W",
         "Board power, uncapped", "watts (grey = idle floor)", idle_share=lambda a: idle)
    mj = [1000 * a["j_per_frame"] if a["j_per_frame"] else None for a in base]
    gpu = sorted(v for v, a in zip(mj, base) if v is not None and "CPU" not in a["label"])
    bars(a_mj, mj, [1000 * a["j_per_frame_sd"] for a in base], "{:.0f} mJ",
         "Energy per frame (power / fps), uncapped",
         "mJ per frame (lower is better; grey = idle floor share)",
         idle_share=lambda a: 1000 * idle / a["fps"],
         limit=(gpu[-1] * 1.05) if gpu else None)

    cw = []
    for yi, a in zip(y, base):
        c = capped.get(a["engine"])
        if c and c["fps"] and c["fps"] >= 0.95 * cap:
            cw.append(c["vdd_in"])
        else:
            cw.append(None)
            a_cap.text(0, yi, f" cannot hold {cap} fps (max {a['fps']:.3g})",
                       va="center", fontsize=8.5, color=INK2)
    cerr = [capped[a["engine"]]["vdd_in_sd"] if v is not None else 0 for v, a in zip(cw, base)]
    bars(a_cap, cw, cerr, "{:.2f} W", f"Board power while holding {cap} fps",
         "watts (lower is better; grey = idle floor)", idle_share=lambda a: idle)

    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels([a["label"] for a in base])
    reps = max(a["n"] for a in base)
    H = fig.get_figheight()   # place header and legend a fixed distance from the edges
    fig.suptitle(f"lithium vs Darknet: YOLOv3-tiny 416x416, Orin Nano {meta.get('power_mode', '')}, "
                 f"{meta.get('date', '')[:10]}", x=0.01, y=1 - 0.12 / H, va="top", ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 1 - 0.42 / H, va="top", s=
             (f"Forward pass only. Mean of {reps} run{'s' if reps > 1 else ''} of {meta['window_s']:.0f} s "
             f"each (error bars = 1 sd). Idle floor {idle:.2f} W. Power is the INA3221 VDD_IN rail."),
             ha="left", fontsize=8.5, color=INK2)
    present = {a["framework"] for a in base}
    handles = [plt.Rectangle((0, 0), 1, 1, color=FRAMEWORK[f]) for f in FRAMEWORK if f in present]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=IDLE))
    fig.legend(handles, [FRAMEWORK_NAME[f] for f in FRAMEWORK if f in present] + ["idle floor"],
               loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=len(handles), fontsize=9)
    fig.tight_layout(rect=(0, 0.42 / H, 1, 1 - 0.62 / H))
    footnote(fig, base)
    save(fig, outdir, "overview")


def summary(agg, meta, outdir, caps):
    lines = [f"# Benchmark {meta.get('date', '')}", "",
             f"- device: {meta.get('host')}, {meta.get('l4t', '')}",
             f"- power mode: {meta.get('power_mode')}",
             f"- idle floor: {meta['idle']['vdd_in']:.3f} W board, "
             f"{meta['idle']['vdd_cpu_gpu_cv']:.3f} W CPU+GPU+CV",
             f"- window {meta['window_s']} s after {meta['settle_s']} s settle, {meta['repeats']} repeat(s)",
             ""]
    if meta.get("missing"):
        lines += [f"- not run (not built): {', '.join(meta['missing'])}", ""]
    for cap in caps:
        items = sorted([a for (e, c), a in agg.items() if c == cap], key=lambda a: a["j_per_frame"] or 9e9)
        lines += [f"## {'Uncapped' if cap == 0 else f'{cap} fps cap'}", "",
                  "| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for a in items:
            f = lambda v, d: "" if v is None else f"{v:.{d}f}"  # noqa: E731
            lines.append(f"| {a['label']} | {f(a['fps'], 1)} | {f(a['ms'], 2)} | {f(a['vdd_in'], 3)} | "
                         f"{f(a['vdd_cpu_gpu_cv'], 3)} | {f(a['gpu_mhz'], 0)} | "
                         f"{f(a['j_per_frame'] and 1000 * a['j_per_frame'], 2)} | "
                         f"{f(a['j_per_frame_marginal'] and 1000 * a['j_per_frame_marginal'], 2)} |")
        lines.append("")
    cav = {a["label"]: a["caveat"] for a in agg.values() if a.get("caveat")}
    if cav:
        lines += ["## Caveats", ""] + [f"- **{k}**: {v}" for k, v in sorted(cav.items())] + [""]
    with open(f"{outdir}/summary.md", "w") as fh:
        fh.write("\n".join(lines))
    print("  summary.md")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = sys.argv[1].rstrip("/")
    rows, meta, series = load(d)
    agg = aggregate(rows)
    outdir = f"{d}/figures"
    os.makedirs(outdir, exist_ok=True)
    caps = sorted({c for (_, c) in agg})
    print(f"figures -> {outdir}")
    fig_throughput(agg, meta, outdir)
    for cap in caps:
        fig_energy(agg, meta, outdir, cap)
        if cap:
            fig_power_at_cap(agg, meta, outdir, cap)
    fig_pareto(agg, meta, outdir)
    fig_rails(agg, meta, outdir)
    fig_traces(agg, series, meta, outdir)
    if 30 in caps:
        fig_overview(agg, meta, outdir, 30)
    summary(agg, meta, outdir, caps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
