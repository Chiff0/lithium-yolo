# Benchmark 2026-11-12T23:25:21

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 15W
- idle floor: 4.906 W board, 0.552 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 427.1 | 2.34 | 10.915 | 4.137 | 612 | 25.55 | 14.07 |
| lithium GPU FP16 CUTLASS graphs | 377.9 | 2.65 | 10.260 | 4.281 | 612 | 27.15 | 14.17 |
| lithium GPU FP16 CUTLASS | 349.4 | 2.86 | 10.013 | 4.028 | 612 | 28.66 | 14.62 |
| TensorRT FP32 | 207.6 | 4.82 | 11.446 | 4.369 | 612 | 55.13 | 31.50 |
| darknet cuDNN FP16 | 98.3 | 10.18 | 10.138 | 3.412 | 612 | 103.17 | 53.24 |
| darknet cuDNN FP32 | 86.3 | 11.58 | 11.195 | 3.970 | 612 | 129.69 | 72.86 |
| lithium GPU FP32 | 65.8 | 15.21 | 10.739 | 4.455 | 612 | 163.31 | 88.70 |

## 15 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 15.0 | 5.01 | 5.159 | 0.686 | 306 | 343.94 | 16.86 |
| lithium GPU FP16 CUTLASS graphs | 15.0 | 5.78 | 5.161 | 0.708 | 306 | 344.10 | 17.03 |
| lithium GPU FP16 CUTLASS | 15.0 | 5.93 | 5.197 | 0.726 | 306 | 346.49 | 19.42 |
| TensorRT FP32 | 15.0 | 9.86 | 5.448 | 0.833 | 306 | 363.17 | 36.10 |
| darknet cuDNN FP16 | 15.0 | 18.41 | 5.796 | 0.986 | 306 | 386.42 | 59.34 |
| darknet cuDNN FP32 | 15.0 | 21.73 | 6.099 | 1.136 | 306 | 406.61 | 79.54 |
| lithium GPU FP32 | 15.0 | 28.86 | 6.343 | 1.435 | 331 | 422.84 | 95.77 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 CUTLASS graphs | 30.0 | 5.68 | 5.384 | 0.849 | 306 | 179.48 | 15.94 |
| TensorRT FP16 | 30.0 | 4.92 | 5.399 | 0.811 | 306 | 179.96 | 16.42 |
| lithium GPU FP16 CUTLASS | 30.0 | 5.88 | 5.425 | 0.862 | 306 | 180.83 | 17.30 |
| TensorRT FP32 | 30.0 | 9.75 | 5.965 | 1.106 | 306 | 198.83 | 35.29 |
| darknet cuDNN FP16 | 30.0 | 18.34 | 6.626 | 1.394 | 306 | 220.87 | 57.33 |
| darknet cuDNN FP32 | 30.0 | 21.57 | 7.255 | 1.700 | 306 | 241.85 | 78.31 |
| lithium GPU FP32 | 30.0 | 18.25 | 7.678 | 2.362 | 510 | 255.92 | 92.38 |

## 60 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 CUTLASS graphs | 60.0 | 5.68 | 5.845 | 1.132 | 306 | 97.41 | 15.65 |
| TensorRT FP16 | 60.0 | 4.93 | 5.852 | 1.048 | 306 | 97.54 | 15.77 |
| lithium GPU FP16 CUTLASS | 60.0 | 5.88 | 5.871 | 1.145 | 306 | 97.84 | 16.07 |
| TensorRT FP32 | 60.0 | 9.75 | 6.969 | 1.644 | 306 | 116.14 | 34.38 |
| darknet cuDNN FP16 | 60.0 | 12.03 | 8.213 | 2.317 | 510 | 136.88 | 55.12 |
| darknet cuDNN FP32 | 60.0 | 11.96 | 9.379 | 2.998 | 612 | 156.32 | 74.55 |
| lithium GPU FP32 | 60.0 | 15.18 | 10.298 | 4.152 | 612 | 171.64 | 89.87 |
