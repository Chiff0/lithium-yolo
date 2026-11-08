"""Every engine the benchmark and the live view know about.

One table so the sweep and the dashboard can never disagree about what
"lithium cuDNN TF32" means. Runs on the Orin (Python 3.10).

Every runner takes the same trailing <fps cap> argument (0 = uncapped) and
prints one JSON line per second: {"fps": .., "ms": .., "duty": ..}.
"""

import os

HOME = os.path.expanduser("~")
ROOT = os.environ.get("LITHIUM_BENCH_ROOT", f"{HOME}/lithium-bench")
SRC = f"{ROOT}/src"
BIN = f"{ROOT}/bin"
REF = f"{SRC}/reference"
CFG = f"{REF}/yolov3-tiny-letterbox.cfg"
WTS = f"{REF}/yolov3-tiny.weights"
IMG = f"{REF}/images/dog.jpg"
# darknet is fed the exact tensor lithium's preprocess produces, so the input
# is bit-identical and preprocessing stays out of the measured scope
INPUT_BIN = f"{REF}/dog/input_c3_h416_w416.bin"
TRT_DIR = os.environ.get("LITHIUM_TRT_DIR", f"{HOME}/trt")

# lithium build variants: name -> CMake options. Add a line here when a new
# option lands in CMakeLists.txt and the benchmark and live view pick it up.
LITHIUM_VARIANTS = {
    "cpu":  [],
    "fp32": ["LITHIUM_CUDA", "LITHIUM_FP32"],
    "fp16": ["LITHIUM_CUDA"],
}


def _lith(variant, label, env=None, caveat=""):
    return {
        "id": f"lithium_{variant}",
        "framework": "lithium",
        "precision": label,
        "label": f"lithium {label}",
        "cmd": [f"{BIN}/lithium_{variant}", CFG, WTS, IMG],
        "env": env or {},
        "caveat": caveat,
    }


ENGINES = [
    _lith("cpu",  "CPU FP32"),
    _lith("fp32", "GPU FP32"),
    _lith("fp16", "GPU FP16"),
    {
        "id": "darknet_cpu", "framework": "darknet", "precision": "CPU FP32",
        "label": "darknet CPU FP32",
        "cmd": [f"{BIN}/darknet_cpu", CFG, WTS, INPUT_BIN], "env": {},
    },
    {
        "id": "darknet_cudnn_fp32", "framework": "darknet", "precision": "cuDNN FP32",
        "label": "darknet cuDNN FP32",
        "cmd": [f"{BIN}/darknet_cuda", CFG, WTS, INPUT_BIN], "env": {},
    },
    {
        "id": "darknet_cudnn_fp16", "framework": "darknet", "precision": "cuDNN FP16",
        "label": "darknet cuDNN FP16",
        "cmd": [f"{BIN}/darknet_cuda_fp16", CFG, WTS, INPUT_BIN], "env": {},
    },
]

for _p in ("fp32", "fp16", "int8"):
    ENGINES.append({
        "id": f"tensorrt_{_p}", "framework": "tensorrt", "precision": _p.upper(),
        "label": f"TensorRT {_p.upper()}",
        "cmd": [f"{BIN}/trt_runner", f"{TRT_DIR}/yolov3-tiny_{_p}.plan"], "env": {},
        "caveat": ("built with trtexec --int8 and no calibration data; speed only"
                   if _p == "int8" else ""),
    })

BY_ID = {e["id"]: e for e in ENGINES}
FRAMEWORKS = ["lithium", "darknet", "tensorrt"]


def available(engine):
    """An engine is runnable once its binary and every file it reads exist."""
    return all(os.path.exists(p) for p in engine["cmd"] if p.startswith("/"))


def command(engine, fps_cap):
    return list(engine["cmd"]) + [str(fps_cap)]


def environment(engine):
    env = dict(os.environ)
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env.update(engine.get("env", {}))
    return env
