# Benchmark 2026-11-10T23:50:14

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 15W
- idle floor: 4.892 W board, 0.532 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT INT8 * | 573.6 | 1.74 | 10.154 | 3.514 | 612 | 17.70 | 9.17 |
| TensorRT FP16 | 426.5 | 2.34 | 10.903 | 4.110 | 612 | 25.56 | 14.09 |
| lithium GPU FP16 CUTLASS | 183.1 | 5.46 | 8.610 | 3.033 | 612 | 47.03 | 20.30 |
| TensorRT FP32 | 207.3 | 4.82 | 11.416 | 4.338 | 612 | 55.07 | 31.47 |
| darknet cuDNN FP16 | 98.5 | 10.15 | 10.182 | 3.436 | 612 | 103.38 | 53.71 |
| lithium GPU FP16 | 86.9 | 11.50 | 9.041 | 3.224 | 612 | 103.99 | 47.72 |
| darknet cuDNN FP32 | 86.6 | 11.55 | 11.204 | 3.971 | 612 | 129.45 | 72.92 |
| lithium GPU FP32 | 51.6 | 19.37 | 10.270 | 4.080 | 612 | 198.90 | 104.15 |
| darknet CPU FP32 | 2.7 | 365.97 | 9.235 | 3.474 | 306 | 3378.80 | 1588.86 |
| lithium CPU FP32 | 0.3 | 3758.94 | 5.838 | 1.190 | 306 | 21623.67 | 3503.67 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT INT8 * | 30.0 | 3.80 | 5.247 | 0.718 | 306 | 174.90 | 11.82 |
| TensorRT FP16 | 30.0 | 4.92 | 5.402 | 0.802 | 306 | 180.08 | 17.00 |
| lithium GPU FP16 CUTLASS | 30.0 | 9.41 | 5.550 | 0.925 | 306 | 185.02 | 21.94 |
| TensorRT FP32 | 30.0 | 9.75 | 5.979 | 1.110 | 306 | 199.31 | 36.23 |
| lithium GPU FP16 | 30.0 | 21.27 | 6.458 | 1.456 | 306 | 215.26 | 52.18 |
| darknet cuDNN FP16 | 30.0 | 18.36 | 6.628 | 1.388 | 306 | 220.94 | 57.86 |
| darknet cuDNN FP32 | 30.0 | 21.55 | 7.262 | 1.704 | 306 | 242.05 | 78.97 |
| lithium GPU FP32 | 30.0 | 22.85 | 8.075 | 2.582 | 510 | 269.18 | 106.10 |
| darknet CPU FP32 | 2.7 | 366.90 | 9.231 | 3.468 | 306 | 3385.68 | 1591.19 |
| lithium CPU FP32 | 0.3 | 3759.93 | 5.836 | 1.188 | 306 | 21613.55 | 3493.55 |

## Caveats

- **TensorRT INT8 ***: built with trtexec --int8 and no calibration data; speed only
