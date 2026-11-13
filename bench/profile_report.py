#!/usr/bin/env python3
"""Summarise an Nsight Compute CSV from profile.sh: time per kernel category and
the kernel-by-kernel sequence of one forward pass (the second one profiled)."""

import csv
import sys
from collections import defaultdict

CATEGORIES = [
    ("layer_0_conv", "layer 0 (direct conv)"),
    ("im2col", "im2col"),
    ("bias_activate_maxpool", "bias + leaky + maxpool"),
    ("bias_activate", "bias + leaky"),
    ("maxpool", "maxpool"),
    ("upsample", "upsample"),
    ("concat_slice", "route (concat)"),
    ("apply_sigmoid", "yolo sigmoid"),
    ("batchnorm", "batchnorm"),
    ("apply_leaky", "leaky"),
]


def category(name):
    if name == "Kernel":
        return "conv (CUTLASS)"
    for key, label in CATEGORIES:
        if key in name:
            return label
    return "GEMM (cuBLAS)"


def load(path):
    with open(path, newline="") as f:
        lines = [l for l in f if not l.startswith("==")]
    kernels = {}
    for row in csv.DictReader(lines):
        if not row.get("ID", "").strip().isdigit():
            continue
        if "Metric Name" in row:
            if row["Metric Name"] != "gpu__time_duration.sum":
                continue
            ns = row["Metric Value"]
        else:
            ns = row.get("gpu__time_duration.sum", "")
        try:
            kernels[int(row["ID"])] = (row["Kernel Name"], float(ns.replace(",", "")) / 1e6)
        except ValueError:
            continue
    return [kernels[i] for i in sorted(kernels)]


def main():
    kernels = load(sys.argv[1])
    names = [name for name, _ in kernels]
    n = next((n for n in range(len(names) // 2, 0, -1) if names[-n:] == names[-2 * n:-n]), 0)
    if n == 0 and names and names.count(names[0]) > 1:
        # the first pass also runs the autotuner's candidates, so the passes
        # differ; every pass starts with the same kernel, so take the last one
        n = len(names) - max(i for i, name in enumerate(names) if name == names[0])
    if n == 0:
        print("could not find two identical forward passes in this profile")
        return 1
    one = kernels[-n:]

    total = sum(ms for _, ms in one)
    by_cat = defaultdict(float)
    count = defaultdict(int)
    for name, ms in one:
        by_cat[category(name)] += ms
        count[category(name)] += 1

    print(f"one forward pass: {len(one)} kernels, {total:.3f} ms of GPU time\n")
    print(f"{'category':<24} {'kernels':>7} {'ms':>8} {'share':>7}")
    for label, ms in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"{label:<24} {count[label]:>7} {ms:>8.3f} {100 * ms / total:>6.1f}%")

    print("\nkernel sequence")
    for i, (name, ms) in enumerate(one):
        print(f"{i:>3}  {ms:>7.3f} ms  {category(name):<22}  {name[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
