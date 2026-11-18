#!/usr/bin/env bash
# Measure the optimisation ladder: every step built by build.sh plus TensorRT and
# Darknet as references, uncapped and at 30 fps, in the Orin's current power
# mode. Then fetch the results and draw the ladder figure and table.
#
#   bench/ladder/run.sh [sweep.py options]
#
# Results land in bench/results/ladder-<timestamp>/.
set -euo pipefail
JETSON="${JETSON:-chiff0@192.168.55.1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$(dirname "$HERE")"
STAMP="ladder-$(ssh "$JETSON" date +%Y%m%d-%H%M%S)"
CAPS="${CAPS:-0,30}"
REPEATS="${REPEATS:-3}"

ENGINES="$(grep -v '^#' "$HERE/steps.tsv" | cut -f1 | sed 's/^/ladder_/' | paste -sd,)"
ENGINES="$ENGINES,tensorrt_fp16,tensorrt_fp32,darknet_cudnn_fp16,darknet_cudnn_fp32"

"$BENCH/sync.sh" > /dev/null
ssh "$JETSON" 'pkill -f "lithium-bench/src/bench/[d]ash.py" || true; sleep 1'
ssh -t "$JETSON" "python3 ~/lithium-bench/src/bench/sweep.py --out ~/lithium-bench/results/$STAMP \
    --only $ENGINES --caps $CAPS --repeats $REPEATS --commit ladder $*"

mkdir -p "$BENCH/results/$STAMP"
scp -q "$JETSON:lithium-bench/results/$STAMP/*" "$BENCH/results/$STAMP/"
"$BENCH/.venv/bin/python" "$BENCH/plot_ladder.py" "$BENCH/results/$STAMP"
echo
echo "results: $BENCH/results/$STAMP"
