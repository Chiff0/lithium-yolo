# Power benchmark

Compares lithium's build variants against Darknet and TensorRT on the Orin Nano,
measuring throughput and energy per frame from the board's INA3221 power monitor.
Everything is driven from the PC over the USB-C link (`chiff0@192.168.55.1`,
override with `JETSON=user@host`).

## Full benchmark with graphs

```bash
bench/run.sh                      # uncapped and 30 fps, 15 s per case, about 15 min
bench/run.sh --repeats 3          # error bars, about 45 min
bench/run.sh --caps 0,15,30,60    # more frame-rate caps
SKIP_BUILD=1 bench/run.sh         # reuse the binaries already on the Orin
```

Results go to `bench/results/<timestamp>/`: `results.csv` (one row per engine,
cap and repeat), `timeseries.csv` (every 50 Hz power sample), `meta.json`
(power mode, idle floor, L4T), and `figures/` (PNG + PDF + `summary.md`).
To re-draw the graphs: `bench/.venv/bin/python bench/plot.py bench/results/<timestamp>`.
Run `python3 -m venv bench/.venv && bench/.venv/bin/pip install matplotlib` once on a new PC.

## Live view

```bash
bench/live.sh
```

Then open http://192.168.55.1:8080. Pick a framework and a precision, set an
fps cap (0 = as fast as possible), and press run. "Save current reading" adds
the 5-second averages to a table you can copy as CSV. Ctrl+C in the terminal
stops it and the running engine.

The live view and `run.sh` cannot run at the same time; both drive the GPU,
and `run.sh` stops the live view before it starts.

## What is measured

- **Scope**: the forward pass only, on one preprocessed 416x416 frame held in
  memory. No capture, letterbox, decode or NMS. Darknet gets the exact input
  tensor lithium produces (`reference/dog/input_c3_h416_w416.bin`).
- **Power**: mean of the INA3221 rails sampled at 50 Hz over the window, after
  20 warm-up frames and a settle period. `VDD_IN` is the whole board.
- **Energy per frame**: board watts / achieved fps. "Above idle" subtracts the
  idle floor measured at the start of the run, which is the cost of the
  inference itself.
- **At a cap**, the runner sleeps between frames, so the GPU governor drops
  the clock. Compare board watts there, not frame time.

## Engines

| framework | variant | how it is built |
|---|---|---|
| lithium | CPU FP32 | no CUDA |
| lithium | GPU FP16 (default) / FP32 | im2col + cuBLAS, `LITHIUM_CUDA` (+ `LITHIUM_FP32`) |
| Darknet | CPU FP32, cuDNN FP32 | hank-ai fork in `~/darknet-bench`, with `CUDNN_HALF` and tensor-op math patched off |
| Darknet | cuDNN FP16 | the same commit, unpatched, in `~/darknet-fp16`: `CUDNN_HALF` and tensor cores on |
| TensorRT | FP32 / FP16 / INT8 | `trtexec` plans in `~/trt` from `yolov3-tiny.onnx` |

The engines are defined in `engines.py`, and `jetson_build.sh` builds them on
the Orin into `~/lithium-bench/`. To benchmark a new lithium build option, add
it to `LITHIUM_VARIANTS` and `ENGINES` there. lithium is driven by
`test/fps_runner.cpp`.

**Speed-only engine** (asterisked in every graph): TensorRT INT8 was built
without calibration data, so its detections are meaningless. Its speed and
power are real, but it is not accuracy-equivalent to the rest.

FP16 fails `test_reference`'s 1e-5 tolerance by design and lands within about
1% on the YOLO outputs; FP32 is the build that must match all 24 layers.
