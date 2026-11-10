#!/usr/bin/env python3
"""Cross-mode figures for a run_modes.sh result directory (runs on the PC).

  bench/.venv/bin/python bench/plot_modes.py bench/results/modes-<stamp>

Expects one sweep result per power mode in <dir>/<mode>/ and writes
power_vs_fps, power_at_30fps_by_mode and summary.md into <dir>/figures/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MODE_ORDER = ["7W", "15W", "MAXN"]


def load_modes(root):
    modes = []
    for name in sorted(os.listdir(root), key=lambda m: (MODE_ORDER.index(m) if m in MODE_ORDER else 99, m)):
        d = os.path.join(root, name)
        if os.path.isfile(os.path.join(d, "results.csv")):
            rows, meta, _ = P.load(d)
            modes.append((name, P.aggregate(rows), meta))
    return modes


def held(a, cap):
    return a is not None and a["fps"] is not None and a["fps"] >= 0.95 * cap


def engine_order(modes):
    """Engines sorted by board power at 30 fps in the middle mode, unholdable last."""
    ref = modes[len(modes) // 2][1]
    engines = {}
    for _, agg, _ in modes:
        for (e, _), a in agg.items():
            engines.setdefault(e, a)
    def key(e):
        a = ref.get((e, 30))
        return a["vdd_in"] if held(a, 30) else 1e9
    return [engines[e] for e in sorted(engines, key=key)]


def fig_power_vs_fps(modes, outdir):
    fig, axes = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 4.4), sharey=True, squeeze=False)
    top = 0.0
    for ax, (mode, agg, meta) in zip(axes[0], modes):
        idle = meta["idle"]["vdd_in"]
        ax.axhline(idle, color=P.IDLE, lw=1.2, ls="--")
        ax.text(1, idle, " idle", va="bottom", ha="left", fontsize=8, color=P.INK2,
                transform=ax.get_yaxis_transform())
        for e in sorted({k[0] for k in agg}):
            pts = []
            for (eng, cap), a in agg.items():
                if eng != e or a["fps"] is None:
                    continue
                if cap == 0 or held(a, cap):
                    pts.append((a["fps"], a["vdd_in"]))
            if not pts:
                continue
            pts.sort()
            any_a = next(a for (eng, _), a in agg.items() if eng == e)
            style = "--" if "FP32" in any_a["label"] else "-"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, marker="o", ms=4.5,
                    color=P.FRAMEWORK[any_a["framework"]], lw=1.8, label=any_a["label"])
            top = max(top, max(p[1] for p in pts))
        ax.set_xscale("log")
        ax.set_xlim(10, max(500, ax.get_xlim()[1]))
        ax.set_xticks([15, 30, 60, 120, 250, 500])
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter("%g"))
        ax.set_title(f"{mode} (idle {idle:.2f} W)", fontsize=10)
        ax.set_xlabel("frames per second (log)")
    axes[0][0].set_ylabel("board power, W")
    axes[0][0].set_ylim(0, top * 1.1)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    for ax in axes[0][:-1]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.suptitle("Board power against frame rate, per power mode "
                 "(points at 15, 30, 60 fps caps and uncapped; solid FP16, dashed FP32)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    P.save(fig, outdir, "power_vs_fps")


def fig_power_at_30(modes, outdir):
    order = engine_order(modes)
    n = len(order)
    y = list(range(n))[::-1]
    fig, axes = plt.subplots(1, len(modes), figsize=(4.0 * len(modes) + 1.6, 0.42 * n + 1.8),
                             sharey=True, squeeze=False)
    top = max((a["vdd_in"] for _, agg, _ in modes for (e, c), a in agg.items()
               if c == 30 and held(a, 30)), default=1.0)
    for ax, (mode, agg, meta) in zip(axes[0], modes):
        idle = meta["idle"]["vdd_in"]
        for yi, eng in zip(y, order):
            a = agg.get((eng["engine"], 30))
            if not held(a, 30):
                best = agg.get((eng["engine"], 0))
                note = f" max {best['fps']:.3g} fps" if best and best["fps"] else " not run"
                ax.text(0, yi, note, va="center", fontsize=8, color=P.INK2)
                continue
            ax.barh(yi, idle, height=0.62, color=P.IDLE, edgecolor=P.SURFACE, linewidth=2)
            ax.barh(yi, a["vdd_in"] - idle, left=idle, height=0.62, color=P.FRAMEWORK[a["framework"]],
                    edgecolor=P.SURFACE, linewidth=2, xerr=a["vdd_in_sd"] or None,
                    error_kw={"ecolor": P.INK2, "elinewidth": 1, "capsize": 2})
            ax.text(a["vdd_in"] + (a["vdd_in_sd"] or 0), yi, f"  {a['vdd_in']:.2f}", va="center", fontsize=8.5)
        ax.set_xlim(0, top * 1.25)
        ax.set_title(f"{mode} (idle {idle:.2f} W)", fontsize=10)
        ax.set_xlabel("board W at 30 fps")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([e["label"] for e in order])
    fig.suptitle("Board power while holding 30 fps, per power mode (grey = that mode's idle floor)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    present = {e["framework"] for e in order}
    handles = [plt.Rectangle((0, 0), 1, 1, color=P.FRAMEWORK[f]) for f in P.FRAMEWORK if f in present]
    fig.legend(handles + [plt.Rectangle((0, 0), 1, 1, color=P.IDLE)],
               [P.FRAMEWORK_NAME[f] for f in P.FRAMEWORK if f in present] + ["idle floor"],
               loc="lower left", bbox_to_anchor=(0.01, -0.06), ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    P.save(fig, outdir, "power_at_30fps_by_mode")


def summary(modes, outdir):
    order = engine_order(modes)
    caps = sorted({c for _, agg, _ in modes for (_, c) in agg if c}, reverse=True)
    meta0 = modes[0][2]
    lines = ["# Power modes", "",
             f"- commit: {meta0.get('commit') or 'not recorded'}",
             f"- device: {meta0.get('host')}, {meta0.get('l4t', '')}",
             f"- window {meta0['window_s']} s after {meta0['settle_s']} s settle, "
             f"{meta0['repeats']} repeat(s)", ""]
    for mode, agg, meta in modes:
        idle = meta["idle"]["vdd_in"]
        lines += [f"## {mode}", "", f"idle floor {idle:.3f} W", "",
                  "| engine | max fps | mJ/frame | " + " | ".join(f"W @{c}" for c in caps) + " |",
                  "|---|---:|---:|" + "---:|" * len(caps)]
        for eng in order:
            m = agg.get((eng["engine"], 0))
            if m is None:
                continue
            cells = []
            for c in caps:
                a = agg.get((eng["engine"], c))
                cells.append(f"{a['vdd_in']:.2f}" if held(a, c) else "—")
            lines.append(f"| {eng['label']} | {m['fps']:.1f} | {1000 * m['j_per_frame']:.1f} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    lines += ["— = cannot hold that frame rate in this mode", ""]
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("  summary.md")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = sys.argv[1].rstrip("/")
    modes = load_modes(root)
    if not modes:
        print(f"no per-mode results under {root}")
        return 1
    outdir = os.path.join(root, "figures")
    os.makedirs(outdir, exist_ok=True)
    print(f"modes: {', '.join(m for m, _, _ in modes)} -> {outdir}")
    fig_power_vs_fps(modes, outdir)
    fig_power_at_30(modes, outdir)
    summary(modes, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
