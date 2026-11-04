#!/usr/bin/env bash
# Start the live power view on the Orin and print its address.
#
#   bench/live.sh          (Ctrl+C stops it and returns the Orin to idle)
set -euo pipefail
JETSON="${JETSON:-chiff0@192.168.55.1}"
PORT="${PORT:-8080}"
HERE="$(cd "$(dirname "$0")" && pwd)"

"$HERE/sync.sh" > /dev/null
# only one thing may drive the GPU at a time: stop the old dashboard if it is up
ssh "$JETSON" 'pkill -f "dash/[d]ash.py" || true; pkill -f "lithium-bench/src/bench/[d]ash.py" || true; sleep 1'
echo "live view: http://${JETSON#*@}:$PORT"
exec ssh -t "$JETSON" "python3 ~/lithium-bench/src/bench/dash.py --port $PORT"
