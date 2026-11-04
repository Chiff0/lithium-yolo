#!/usr/bin/env bash
# Copy this working tree (tracked + untracked, minus .gitignore'd files) to the
# Orin at ~/lithium-bench/src. Uncommitted changes go across as they are.
#
# Files land in a staging dir first and only changed ones (by checksum) are
# copied in, stamped with the Orin's own clock, so make's incremental rebuilds
# stay correct even when the two machines' clocks disagree. Files deleted here
# are deleted there too (except the fetched stb_image.h).
set -euo pipefail
JETSON="${JETSON:-chiff0@192.168.55.1}"
cd "$(dirname "$0")/.."
git ls-files -z -co --exclude-standard | tar --null -T - -cf - \
    | ssh "$JETSON" 'set -e; S=~/lithium-bench/.stage; rm -rf $S; mkdir -p $S ~/lithium-bench/src
                     tar xmf - -C $S --warning=no-timestamp
                     rsync -rc --delete --exclude=third_party/stb_image.h --out-format="  %n" $S/ ~/lithium-bench/src/; rm -rf $S'
echo "synced to $JETSON:~/lithium-bench/src"
