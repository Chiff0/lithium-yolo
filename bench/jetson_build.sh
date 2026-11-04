#!/usr/bin/env bash
# Build every benchmark runner on the Orin. Run from the synced source tree:
#
#   ~/lithium-bench/src/bench/jetson_build.sh [variant ...]
#
# Lithium variants are the CMake option sets in engines.py (default: all).
# Darknet runners link the hank-ai Darknet builds in $DARKNET (CPU and the
# cuDNN build with CUDNN_HALF and tensor-op math patched off, so it is FP32).
# TensorRT plans are reused from $LITHIUM_TRT_DIR, built with trtexec if missing.
set -euo pipefail
export LC_ALL=C   # the Orin's locale uses a decimal comma, which breaks awk/printf

[ -f "$HOME/lithium-env.sh" ] && source "$HOME/lithium-env.sh"
export PATH="/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH"

SRC="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(dirname "$SRC")"
BUILD="$ROOT/build"
BIN="$ROOT/bin"
DARKNET="${DARKNET:-$HOME/darknet-bench}"
TRT_DIR="${LITHIUM_TRT_DIR:-$HOME/trt}"
CXX="${LITHIUM_CXX:-g++-12}"
JOBS="${JOBS:-4}"
mkdir -p "$BUILD" "$BIN"

# stb_image.h is fetched, not vendored; reuse the copy already on the device
if [ ! -f "$SRC/third_party/stb_image.h" ]; then
    for c in "$HOME/lithium-yolo/third_party/stb_image.h" "$HOME/lithium-fp16/third_party/stb_image.h"; do
        [ -f "$c" ] && { cp "$c" "$SRC/third_party/"; break; }
    done
fi
[ -f "$SRC/third_party/stb_image.h" ] || { echo "third_party/stb_image.h missing"; exit 1; }

variant_opts() {
    python3 -c "import sys; sys.path.insert(0, '$SRC/bench'); import engines; \
print(' '.join('-D%s=ON' % o for o in engines.LITHIUM_VARIANTS['$1']))"
}

if [ $# -gt 0 ]; then
    VARIANTS="$*"
else
    VARIANTS="$(python3 -c "import sys; sys.path.insert(0, '$SRC/bench'); import engines; print(' '.join(engines.LITHIUM_VARIANTS))")"
fi

echo "== lithium ($VARIANTS)"
for v in $VARIANTS; do
    opts="$(variant_opts "$v")"
    echo "-- $v: ${opts:-CPU only}"
    cmake -S "$SRC" -B "$BUILD/$v" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" $opts \
        > "$BUILD/$v.configure.log" 2>&1 || { tail -20 "$BUILD/$v.configure.log"; exit 1; }
    cmake --build "$BUILD/$v" -j"$JOBS" --target fps_runner bench test_reference \
        > "$BUILD/$v.build.log" 2>&1 || { tail -30 "$BUILD/$v.build.log"; exit 1; }
    cp "$BUILD/$v/fps_runner" "$BIN/lithium_$v"
done

echo "== darknet ($DARKNET)"
for kind in cpu cuda; do
    lib="$DARKNET/build-$kind/src-lib"
    if [ -f "$lib/libdarknet.so" ]; then
        "$CXX" -O2 -std=c++17 -I"$DARKNET/src-lib" "$SRC/bench/runners/fps_runner_darknet.cpp" \
            -L"$lib" -ldarknet -Wl,-rpath,"$lib" -o "$BIN/darknet_$kind"
        echo "-- darknet_$kind"
    else
        echo "-- darknet_$kind skipped: no $lib/libdarknet.so"
    fi
done

echo "== tensorrt"
"$CXX" -O2 -std=c++17 -I/usr/local/cuda/include "$SRC/bench/runners/fps_runner_trt.cpp" \
    -L/usr/local/cuda/lib64 -lcudart -lnvinfer -o "$BIN/trt_runner"
echo "-- trt_runner"
for p in fp32 fp16 int8; do
    plan="$TRT_DIR/yolov3-tiny_$p.plan"
    if [ -f "$plan" ]; then
        echo "-- $plan (reused)"
    elif [ -f "$TRT_DIR/yolov3-tiny.onnx" ]; then
        flag=""; [ "$p" = fp16 ] && flag="--fp16"; [ "$p" = int8 ] && flag="--int8"
        echo "-- building $plan with trtexec $flag"
        trtexec --onnx="$TRT_DIR/yolov3-tiny.onnx" $flag --saveEngine="$plan" > "$TRT_DIR/build_$p.log" 2>&1
    else
        echo "-- $p skipped: no plan and no $TRT_DIR/yolov3-tiny.onnx"
    fi
done
echo "BUILD_OK"
