#!/usr/bin/env bash
# Power-mode x frame-rate grid: every engine at every fps cap in every nvpmodel
# mode, then per-mode figures and a cross-mode comparison. Asks for the Orin's
# sudo password once (nvpmodel needs root) and restores the starting mode.
#
#   bench/run_modes.sh [sweep.py options]
#
# Override with environment variables:
#   MODES="1 0 2"     nvpmodel ids, default 7W, 15W, MAXN
#   ENGINES=...       comma-separated engine ids
#   CAPS=0,60,30,15   fps caps, 0 = uncapped
#   REPEATS=3
#   STAMP=modes-...   add to an existing result folder (e.g. a mode that needed a reboot)
#
# Results land in bench/results/modes-<timestamp>/<mode>/ plus the combined
# figures in bench/results/modes-<timestamp>/.
set -euo pipefail
JETSON="${JETSON:-chiff0@192.168.55.1}"
MODES="${MODES:-1 0 2}"
ENGINES="${ENGINES:-lithium_fp16_cutlass,lithium_fp32,darknet_cudnn_fp16,darknet_cudnn_fp32,tensorrt_fp16,tensorrt_fp32}"
CAPS="${CAPS:-0,60,30,15}"
REPEATS="${REPEATS:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="${STAMP:-modes-$(ssh "$JETSON" date +%Y%m%d-%H%M%S)}"
COMMIT="$(git -C "$HERE/.." rev-parse --short HEAD)$(git -C "$HERE/.." diff --quiet -- src test CMakeLists.txt || echo -dirty)"

"$HERE/sync.sh" > /dev/null
if [ "${SKIP_BUILD:-0}" != 1 ]; then
    ssh "$JETSON" 'bash ~/lithium-bench/src/bench/jetson_build.sh' | grep -E 'BUILD_OK|rror'
fi
ssh "$JETSON" 'pkill -f "lithium-bench/src/bench/[d]ash.py" || true; pkill -f "dash/[d]ash.py" || true; sleep 1'

echo "commit $COMMIT, modes [$MODES], caps $CAPS, $REPEATS repeats"
ssh -t "$JETSON" "bash ~/lithium-bench/src/bench/modes_remote.sh ~/lithium-bench/results/$STAMP \
    '$MODES' '$ENGINES' '$CAPS' '$REPEATS' '$COMMIT' $*"

mkdir -p "$HERE/results/$STAMP"
scp -q -r "$JETSON:lithium-bench/results/$STAMP/*" "$HERE/results/$STAMP/"
for d in "$HERE/results/$STAMP"/*/; do
    "$HERE/.venv/bin/python" "$HERE/plot.py" "$d" > /dev/null
done
"$HERE/.venv/bin/python" "$HERE/plot_modes.py" "$HERE/results/$STAMP"
echo
echo "results: $HERE/results/$STAMP"
