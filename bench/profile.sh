#!/usr/bin/env bash
# Per-kernel GPU time for one forward pass, measured with Nsight Compute on the
# Orin. ncu needs root there, so this asks for the Orin's sudo password.
#
#   bench/profile.sh [fp16|fp32] [full]
#
# Writes bench/results/profile-<variant>-<stamp>/{kernels.csv,summary.txt}.
# With "full", also collects every Nsight Compute section (memory, roofline,
# occupancy, ...) into lithium.ncu-rep for the Nsight Compute GUI. That run
# replays each kernel dozens of times and takes several minutes.
set -euo pipefail
JETSON="${JETSON:-chiff0@192.168.55.1}"
VARIANT="${1:-fp16}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(ssh "$JETSON" date +%Y%m%d-%H%M%S)"
OUT="$HERE/results/profile-$VARIANT-$STAMP"

"$HERE/sync.sh" > /dev/null
ssh "$JETSON" "bash ~/lithium-bench/src/bench/jetson_build.sh $VARIANT" | grep -E 'BUILD_OK|rror'

# two passes, no warmup: the report keeps only the second, so first-call
# setup (cuBLAS heuristics, half-weight upload) stays out of the numbers.
# --clock-control none keeps the governor's clocks instead of base clocks.
APP="../build/$VARIANT/bench reference/yolov3-tiny-letterbox.cfg reference/yolov3-tiny.weights \
    reference/images/dog.jpg 2 0 gpu"
NCU=/usr/local/cuda/bin/ncu
mkdir -p "$OUT"

if [ "${2:-}" = full ]; then
    ssh -t "$JETSON" "cd ~/lithium-bench/src && sudo LC_ALL=C $NCU --set full --clock-control none \
        --export /tmp/lithium --force-overwrite $APP > /dev/null && sudo chmod a+r /tmp/lithium.ncu-rep"
    ssh "$JETSON" "LC_ALL=C $NCU --import /tmp/lithium.ncu-rep --csv --print-units base \
        --print-kernel-base function --page raw --metrics gpu__time_duration.sum" > "$OUT/kernels.csv"
    scp -q "$JETSON:/tmp/lithium.ncu-rep" "$OUT/lithium.ncu-rep"
else
    ssh -t "$JETSON" "cd ~/lithium-bench/src && sudo LC_ALL=C $NCU \
        --csv --print-units base --print-kernel-base function --clock-control none \
        --metrics gpu__time_duration.sum --log-file /tmp/lithium-ncu.csv $APP > /dev/null"
    scp -q "$JETSON:/tmp/lithium-ncu.csv" "$OUT/kernels.csv"
fi

python3 "$HERE/profile_report.py" "$OUT/kernels.csv" | tee "$OUT/summary.txt"
echo
echo "saved to $OUT"
