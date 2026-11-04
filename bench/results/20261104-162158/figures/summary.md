# Benchmark 2026-11-04T16:22:07

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 15W
- idle floor: 4.861 W board, 0.554 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| darknet cuDNN FP32 | 86.6 | 11.55 | 11.180 | 4.005 | 612 | 129.09 | 72.96 |
| lithium GPU FP16 | 64.4 | 15.54 | 9.307 | 3.207 | 612 | 144.60 | 69.07 |
| lithium GPU TF32 | 62.2 | 16.07 | 10.281 | 3.620 | 612 | 165.24 | 87.11 |
| lithium GPU FP32 | 43.9 | 22.76 | 10.245 | 4.065 | 612 | 233.22 | 122.55 |
| darknet CPU FP32 | 2.8 | 360.53 | 9.227 | 3.506 | 306 | 3326.17 | 1573.67 |
| lithium CPU FP32 | 0.3 | 3798.35 | 5.728 | 1.191 | 306 | 22030.81 | 3332.93 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 | 30.0 | 21.86 | 7.060 | 1.797 | 408 | 235.33 | 73.28 |
| darknet cuDNN FP32 | 30.0 | 21.54 | 7.240 | 1.740 | 306 | 241.32 | 79.28 |
| lithium GPU TF32 | 30.0 | 22.89 | 7.600 | 2.012 | 408 | 253.32 | 91.27 |
| lithium GPU FP32 | 30.0 | 22.99 | 8.668 | 3.057 | 612 | 288.94 | 126.89 |
| darknet CPU FP32 | 2.8 | 360.38 | 9.227 | 3.511 | 306 | 3324.96 | 1573.08 |
| lithium CPU FP32 | 0.3 | 3797.14 | 5.736 | 1.191 | 306 | 22062.32 | 3364.44 |
