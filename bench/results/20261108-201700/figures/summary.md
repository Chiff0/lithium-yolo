# Benchmark 2026-11-08T20:17:14

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 15W
- idle floor: 4.872 W board, 0.532 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 | 86.3 | 11.59 | 9.030 | 3.233 | 612 | 104.65 | 48.19 |
| darknet cuDNN FP32 | 86.5 | 11.56 | 11.101 | 3.962 | 612 | 128.33 | 72.01 |
| lithium GPU FP32 | 51.6 | 19.36 | 10.222 | 4.063 | 612 | 197.94 | 103.60 |
| darknet CPU FP32 | 2.8 | 359.71 | 9.262 | 3.520 | 306 | 3331.80 | 1579.22 |
| lithium CPU FP32 | 0.3 | 3761.39 | 5.791 | 1.174 | 306 | 21449.67 | 3406.29 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 | 30.0 | 21.45 | 6.404 | 1.433 | 306 | 213.46 | 51.07 |
| darknet cuDNN FP32 | 30.0 | 21.56 | 7.178 | 1.702 | 306 | 239.26 | 76.87 |
| lithium GPU FP32 | 30.0 | 22.79 | 8.046 | 2.579 | 510 | 268.20 | 105.81 |
| darknet CPU FP32 | 2.8 | 358.94 | 9.255 | 3.528 | 306 | 3321.15 | 1572.92 |
| lithium CPU FP32 | 0.3 | 3761.20 | 5.789 | 1.175 | 306 | 21441.27 | 3397.89 |
