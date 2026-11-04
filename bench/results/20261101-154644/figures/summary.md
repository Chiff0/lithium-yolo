# Benchmark 2026-11-01T15:46:48

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 15W
- idle floor: 4.868 W board, 0.559 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 1 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT INT8 * | 576.2 | 1.74 | 10.129 | 3.545 | 612 | 17.58 | 9.13 |
| TensorRT FP16 | 426.3 | 2.35 | 10.806 | 4.121 | 612 | 25.35 | 13.93 |
| TensorRT FP32 | 207.5 | 4.82 | 11.333 | 4.343 | 612 | 54.61 | 31.15 |
| lithium cuDNN TF32 | 90.6 | 11.04 | 10.209 | 3.693 | 612 | 112.75 | 58.99 |
| darknet cuDNN FP32 | 86.5 | 11.57 | 11.101 | 3.979 | 612 | 128.40 | 72.09 |
| lithium im2col INT8 * | 70.2 | 14.25 | 9.105 | 3.198 | 612 | 129.77 | 60.39 |
| lithium im2col FP16 | 70.7 | 14.15 | 9.230 | 3.232 | 612 | 130.60 | 61.72 |
| lithium cuDNN FP32 | 80.6 | 12.41 | 11.193 | 4.156 | 612 | 138.89 | 78.48 |
| lithium im2col TF32 | 68.3 | 14.63 | 10.290 | 3.636 | 612 | 150.57 | 79.34 |
| lithium cuDNN FP16 | 59.1 | 16.93 | 9.148 | 3.417 | 612 | 154.90 | 72.47 |
| lithium cuDNN NHWC FP16 | 57.5 | 17.38 | 9.436 | 3.334 | 612 | 163.98 | 79.38 |
| lithium im2col FP32 | 47.4 | 21.08 | 10.322 | 4.178 | 612 | 217.58 | 114.97 |
| darknet CPU FP32 | 2.8 | 363.48 | 9.201 | 3.499 | 306 | 3344.07 | 1574.79 |
| lithium CPU FP32 | 0.3 | 3799.83 | 5.721 | 1.176 | 306 | 22002.28 | 3279.03 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT INT8 * | 30.0 | 3.77 | 5.197 | 0.717 | 306 | 173.24 | 10.97 |
| TensorRT FP16 | 30.0 | 4.94 | 5.354 | 0.813 | 306 | 178.48 | 16.21 |
| TensorRT FP32 | 30.0 | 9.73 | 5.946 | 1.155 | 306 | 198.20 | 35.93 |
| lithium cuDNN TF32 | 30.0 | 19.39 | 6.779 | 1.559 | 306 | 225.95 | 63.68 |
| lithium im2col INT8 * | 30.0 | 20.07 | 6.783 | 1.677 | 408 | 226.08 | 63.82 |
| lithium im2col FP16 | 30.0 | 20.29 | 6.841 | 1.688 | 408 | 228.04 | 65.77 |
| lithium cuDNN FP16 | 30.0 | 20.22 | 7.155 | 2.042 | 510 | 238.49 | 76.22 |
| darknet cuDNN FP32 | 30.0 | 21.59 | 7.195 | 1.723 | 306 | 239.84 | 77.57 |
| lithium cuDNN FP32 | 30.0 | 22.86 | 7.369 | 1.848 | 306 | 245.62 | 83.35 |
| lithium cuDNN NHWC FP16 | 30.0 | 20.42 | 7.370 | 2.060 | 510 | 245.68 | 83.41 |
| lithium im2col TF32 | 30.0 | 21.04 | 7.374 | 1.903 | 408 | 245.79 | 83.52 |
| lithium im2col FP32 | 30.0 | 21.46 | 8.421 | 2.920 | 612 | 280.71 | 118.44 |
| darknet CPU FP32 | 2.8 | 361.96 | 9.196 | 3.515 | 306 | 3328.37 | 1566.41 |
| lithium CPU FP32 | 0.3 | 3796.99 | 5.739 | 1.176 | 306 | 22072.73 | 3349.48 |

## Caveats

- **TensorRT INT8 ***: built with trtexec --int8 and no calibration data; speed only
- **lithium im2col INT8 ***: output does not match the reference from layer 2 on; speed only
