#!/usr/bin/env bash
# Runs on the Orin, started by run_modes.sh. Switches nvpmodel power modes and
# runs one sweep per mode. sudo is asked for once and kept alive for the run;
# the starting mode is restored on exit, however the script ends.
#
#   modes_remote.sh <out dir> <mode ids> <engines> <caps> <repeats> <commit> [sweep options]
set -u
export LC_ALL=C

OUT=$1 MODES=$2 ENGINES=$3 CAPS=$4 REPEATS=$5 COMMIT=$6
shift 6

sudo -v || { echo "sudo failed"; exit 1; }
( while true; do sudo -n true 2> /dev/null; sleep 50; done ) &
KEEPALIVE=$!

START_ID=$(nvpmodel -q 2> /dev/null | sed -n 2p)
restore() {
    echo no | sudo nvpmodel -m "$START_ID" > /dev/null 2>&1
    echo "restored power mode $(nvpmodel -q 2> /dev/null | head -1)"
    kill "$KEEPALIVE" 2> /dev/null
}
trap restore EXIT

for ID in $MODES; do
    NAME=$(grep -oP "ID=$ID NAME=\K\S+" /etc/nvpmodel.conf | head -1)
    echo
    echo "################ power mode $ID ($NAME)"
    # "no" declines a reboot prompt: a mode that needs one is reported and skipped
    echo no | sudo nvpmodel -m "$ID" > /tmp/lithium-nvpmodel.log 2>&1
    sleep 20
    NOW=$(nvpmodel -q 2> /dev/null | sed -n 2p)
    if [ "$NOW" != "$ID" ]; then
        echo "mode $NAME did not take effect (probably needs a reboot), skipped:"
        tail -3 /tmp/lithium-nvpmodel.log
        continue
    fi
    python3 ~/lithium-bench/src/bench/sweep.py --out "$OUT/$NAME" --only "$ENGINES" \
        --caps "$CAPS" --repeats "$REPEATS" --commit "$COMMIT" "$@"
done
