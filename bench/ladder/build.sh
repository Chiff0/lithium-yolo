#!/usr/bin/env bash
# Build every step of the optimisation ladder on the Orin. Each step is its
# commit exported with git archive (the repository is not touched), with
# bench/ladder/runner.cpp added as an extra CMake target; the engine sources are
# built exactly as committed. Binaries land in ~/lithium-bench/bin/ladder/<id>.
#
#   bench/ladder/build.sh [id ...]      (default: every step in steps.tsv)
set -euo pipefail
JETSON="${JETSON:-chiff0@192.168.55.1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
REMOTE=lithium-ladder

"$REPO/bench/sync.sh" > /dev/null
ssh "$JETSON" "mkdir -p ~/$REMOTE/src ~/$REMOTE/build ~/lithium-bench/bin/ladder"
scp -q "$HERE/runner.cpp" "$JETSON:$REMOTE/ladder_runner.cpp"

wanted=" $* "
while IFS=$'\t' read -r id commit api options label; do
    [[ -z "$id" || "$id" == \#* ]] && continue
    [[ $# -gt 0 && "$wanted" != *" $id "* ]] && continue

    flags=""
    [ "$options" != "-" ] && flags="$(echo "$options" | tr ',' '\n' | sed 's/^/-D/; s/$/=ON/' | tr '\n' ' ')"
    echo "== $id $commit  ${options}  ($label)"

    if ! ssh -n "$JETSON" "test -d ~/$REMOTE/src/$commit"; then
        git -C "$REPO" archive "$commit" | ssh "$JETSON" "set -e; d=~/$REMOTE/src/$commit; mkdir -p \$d
            tar xmf - -C \$d --warning=no-timestamp
            mkdir -p \$d/third_party && cp ~/lithium-bench/src/third_party/stb_image.h \$d/third_party/
            cp ~/$REMOTE/ladder_runner.cpp \$d/ladder_runner.cpp
            printf '%s\n' '' \
                'add_executable(ladder_runner ladder_runner.cpp)' \
                'target_include_directories(ladder_runner PRIVATE \${CMAKE_CURRENT_SOURCE_DIR}/src)' \
                'target_link_libraries(ladder_runner PRIVATE lithium_core)' \
                'target_compile_definitions(ladder_runner PRIVATE LADDER_API=$api)' >> \$d/CMakeLists.txt"
    else
        ssh -n "$JETSON" "cp ~/$REMOTE/ladder_runner.cpp ~/$REMOTE/src/$commit/ladder_runner.cpp"
    fi

    if ssh -n "$JETSON" "set -e; [ -f ~/lithium-env.sh ] && source ~/lithium-env.sh
        export PATH=/usr/local/cuda/bin:\$PATH LC_ALL=C
        b=~/$REMOTE/build/$id
        cmake -S ~/$REMOTE/src/$commit -B \$b -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER=\${LITHIUM_CXX:-g++-12} $flags > \$b.configure.log 2>&1
        cmake --build \$b -j\${JOBS:-4} --target ladder_runner > \$b.build.log 2>&1
        cp \$b/ladder_runner ~/lithium-bench/bin/ladder/$id"; then
        echo "   ok"
    else
        echo "   FAILED, see ~/$REMOTE/build/$id.{configure,build}.log on the Orin"
    fi
done < "$HERE/steps.tsv"
