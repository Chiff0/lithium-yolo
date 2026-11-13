# Benchmark 2026-11-13T00:00:57

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: MAXN
- idle floor: 4.917 W board, 0.561 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 669.8 | 1.49 | 18.413 | 8.998 | 1020 | 27.49 | 20.15 |
| lithium GPU FP16 CUTLASS | 512.2 | 1.96 | 15.816 | 8.812 | 1020 | 30.90 | 21.25 |
| lithium GPU FP16 CUTLASS graphs | 567.7 | 1.76 | 17.704 | 9.680 | 1020 | 31.20 | 22.54 |
| TensorRT FP32 | 311.6 | 3.21 | 19.228 | 8.857 | 1020 | 61.70 | 45.93 |
| darknet cuDNN FP16 | 154.2 | 6.48 | 16.605 | 7.311 | 1020 | 107.67 | 75.79 |
| darknet cuDNN FP32 | 135.5 | 7.38 | 18.728 | 8.539 | 1020 | 138.26 | 101.96 |
| lithium GPU FP32 | 109.3 | 9.15 | 18.767 | 10.038 | 1020 | 171.69 | 126.71 |

## 15 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 15.0 | 4.98 | 5.199 | 0.709 | 306 | 346.62 | 18.84 |
| lithium GPU FP16 CUTLASS | 15.0 | 6.25 | 5.208 | 0.737 | 306 | 347.19 | 19.41 |
| lithium GPU FP16 CUTLASS graphs | 15.0 | 5.89 | 5.257 | 0.784 | 306 | 350.46 | 22.68 |
| TensorRT FP32 | 15.0 | 9.82 | 5.489 | 0.861 | 306 | 365.92 | 38.14 |
| darknet cuDNN FP16 | 15.0 | 18.45 | 5.820 | 1.016 | 306 | 387.98 | 60.20 |
| darknet cuDNN FP32 | 15.0 | 21.66 | 6.087 | 1.165 | 306 | 405.80 | 78.02 |
| lithium GPU FP32 | 15.0 | 28.28 | 6.372 | 1.507 | 349 | 424.77 | 96.99 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 30.0 | 4.93 | 5.410 | 0.843 | 306 | 180.32 | 16.43 |
| lithium GPU FP16 CUTLASS | 30.0 | 6.10 | 5.434 | 0.898 | 306 | 181.13 | 17.24 |
| lithium GPU FP16 CUTLASS graphs | 30.0 | 6.21 | 5.527 | 0.977 | 306 | 184.25 | 20.36 |
| TensorRT FP32 | 30.0 | 9.75 | 5.955 | 1.137 | 306 | 198.50 | 34.61 |
| darknet cuDNN FP16 | 30.0 | 18.35 | 6.653 | 1.461 | 306 | 221.77 | 57.88 |
| darknet cuDNN FP32 | 30.0 | 21.56 | 7.307 | 1.793 | 306 | 243.58 | 79.69 |
| lithium GPU FP32 | 30.0 | 18.25 | 7.723 | 2.435 | 510 | 257.43 | 93.54 |

## 60 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 60.0 | 4.93 | 5.863 | 1.093 | 306 | 97.72 | 15.78 |
| lithium GPU FP16 CUTLASS | 60.0 | 6.09 | 5.912 | 1.207 | 306 | 98.54 | 16.59 |
| lithium GPU FP16 CUTLASS graphs | 60.0 | 6.28 | 5.937 | 1.248 | 306 | 98.96 | 17.01 |
| TensorRT FP32 | 60.0 | 9.75 | 7.028 | 1.722 | 306 | 117.13 | 35.18 |
| darknet cuDNN FP16 | 60.0 | 12.05 | 8.251 | 2.405 | 510 | 137.52 | 55.58 |
| darknet cuDNN FP32 | 60.0 | 11.64 | 10.329 | 3.104 | 612 | 172.15 | 90.20 |
| lithium GPU FP32 | 60.0 | 10.65 | 11.520 | 5.540 | 918 | 192.00 | 110.06 |
