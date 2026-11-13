# Benchmark 2026-11-13T00:55:43

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 7W
- idle floor: 4.807 W board, 0.475 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 186.8 | 5.35 | 7.742 | 1.990 | 408 | 41.44 | 15.71 |
| lithium GPU FP16 CUTLASS graphs | 150.3 | 6.65 | 7.119 | 1.908 | 408 | 47.37 | 15.38 |
| lithium GPU FP16 CUTLASS | 144.5 | 6.92 | 7.126 | 1.871 | 408 | 49.33 | 16.05 |
| TensorRT FP32 | 83.7 | 11.94 | 7.776 | 1.983 | 408 | 92.87 | 35.46 |
| darknet cuDNN FP16 | 40.7 | 24.59 | 7.253 | 1.632 | 408 | 178.38 | 60.15 |
| darknet cuDNN FP32 | 35.8 | 27.90 | 7.719 | 1.865 | 408 | 215.33 | 81.22 |
| lithium GPU FP32 | 24.8 | 40.36 | 7.204 | 1.911 | 408 | 290.75 | 96.73 |

## 15 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 CUTLASS graphs | 15.0 | 9.41 | 4.977 | 0.527 | 306 | 331.78 | 11.30 |
| lithium GPU FP16 CUTLASS | 15.0 | 9.61 | 4.988 | 0.534 | 306 | 332.52 | 12.04 |
| TensorRT FP16 | 15.0 | 7.68 | 4.988 | 0.519 | 306 | 332.56 | 12.08 |
| TensorRT FP32 | 15.0 | 16.34 | 5.303 | 0.671 | 306 | 353.56 | 33.08 |
| darknet cuDNN FP16 | 15.0 | 31.66 | 5.690 | 0.840 | 316 | 379.35 | 58.87 |
| darknet cuDNN FP32 | 15.0 | 35.61 | 6.030 | 1.004 | 324 | 401.99 | 81.51 |
| lithium GPU FP32 | 15.0 | 40.69 | 6.268 | 1.335 | 408 | 417.84 | 97.36 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 CUTLASS graphs | 30.0 | 9.37 | 5.228 | 0.679 | 306 | 174.27 | 14.03 |
| TensorRT FP16 | 30.0 | 7.60 | 5.242 | 0.638 | 306 | 174.74 | 14.50 |
| lithium GPU FP16 CUTLASS | 30.0 | 9.59 | 5.253 | 0.685 | 306 | 175.10 | 14.86 |
| TensorRT FP32 | 30.0 | 16.29 | 5.881 | 0.957 | 306 | 196.02 | 35.78 |
| darknet cuDNN FP16 | 30.0 | 25.05 | 6.622 | 1.318 | 408 | 220.74 | 60.50 |
| darknet cuDNN FP32 | 30.0 | 28.34 | 7.258 | 1.632 | 408 | 241.92 | 81.68 |
| lithium GPU FP32 | 24.8 | 40.36 | 7.217 | 1.911 | 408 | 291.30 | 97.27 |

## 60 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 CUTLASS graphs | 60.0 | 9.37 | 5.723 | 0.997 | 306 | 95.39 | 15.27 |
| TensorRT FP16 | 60.0 | 7.59 | 5.752 | 0.917 | 306 | 95.86 | 15.74 |
| lithium GPU FP16 CUTLASS | 60.0 | 9.57 | 5.777 | 0.997 | 306 | 96.29 | 16.17 |
| TensorRT FP32 | 60.0 | 12.41 | 6.953 | 1.553 | 408 | 115.88 | 35.76 |
| darknet cuDNN FP16 | 40.7 | 24.58 | 7.256 | 1.632 | 408 | 178.37 | 60.20 |
| darknet cuDNN FP32 | 35.9 | 27.88 | 7.729 | 1.868 | 408 | 215.49 | 81.46 |
| lithium GPU FP32 | 24.8 | 40.36 | 7.215 | 1.911 | 408 | 291.19 | 97.17 |
