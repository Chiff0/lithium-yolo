#!/usr/bin/env bash
# Per-kernel GPU time for one forward pass, measured with Nsight Compute on the
# Orin. ncu needs root there, so this asks for the Orin's sudo password.
#
#   bench/profile.sh [fp16|fp32]
#
# Writes bench/results/profile-<variant>-<stamp>/{kernels.csv,summary.txt}.
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
ssh -t "$JETSON" "cd ~/lithium-bench/src && sudo LC_ALL=C /usr/local/cuda/bin/ncu \
    --csv --print-units base --print-kernel-base function --clock-control none \
    --metrics gpu__time_duration.sum --log-file /tmp/lithium-ncu.csv \
    ../build/$VARIANT/bench reference/yolov3-tiny-letterbox.cfg reference/yolov3-tiny.weights \
    reference/images/dog.jpg 2 0 gpu > /dev/null"

mkdir -p "$OUT"
scp -q "$JETSON:/tmp/lithium-ncu.csv" "$OUT/kernels.csv"
python3 "$HERE/profile_report.py" "$OUT/kernels.csv" | tee "$OUT/summary.txt"
echo
echo "saved to $OUT"
