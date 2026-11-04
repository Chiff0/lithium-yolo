#!/usr/bin/env bash
# Full benchmark from the PC: sync -> build on the Orin -> sweep -> fetch -> graphs.
#
#   bench/run.sh [sweep.py options]      e.g. bench/run.sh --caps 0,30 --repeats 3
#
# Results land in bench/results/<timestamp>/ with figures/ and summary.md.
# Set SKIP_BUILD=1 to reuse the binaries already on the Orin.
set -euo pipefail
JETSON="${JETSON:-chiff0@192.168.55.1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(ssh "$JETSON" date +%Y%m%d-%H%M%S)"   # the Orin's clock is NTP-synced
OUT="$HERE/results/$STAMP"

"$HERE/sync.sh"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
    ssh "$JETSON" 'bash ~/lithium-bench/src/bench/jetson_build.sh' | grep -E '^(==|--|BUILD_OK)|rror'
fi

# the old ~/dash/dash.py or a live view would share the GPU with the sweep
ssh "$JETSON" 'pkill -f "lithium-bench/src/bench/[d]ash.py" || true; pkill -f "dash/[d]ash.py" || true; sleep 1'

ssh -t "$JETSON" "python3 ~/lithium-bench/src/bench/sweep.py --out ~/lithium-bench/results/$STAMP $*"

mkdir -p "$OUT"
scp -q "$JETSON:lithium-bench/results/$STAMP/*" "$OUT/"
"$HERE/.venv/bin/python" "$HERE/plot.py" "$OUT"
echo
echo "results: $OUT"
