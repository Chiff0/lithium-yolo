# Benchmark 2026-11-10T22:46:36

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 15W
- idle floor: 4.947 W board, 0.592 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 CUTLASS | 148.1 | 6.76 | 9.002 | 3.333 | 612 | 60.81 | 27.39 |
| darknet cuDNN FP16 | 98.4 | 10.16 | 10.216 | 3.476 | 612 | 103.84 | 53.55 |
| lithium GPU FP16 | 87.4 | 11.44 | 9.161 | 3.328 | 612 | 104.82 | 48.21 |
| darknet cuDNN FP32 | 86.5 | 11.56 | 11.234 | 3.994 | 612 | 129.92 | 72.70 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 CUTLASS | 30.0 | 12.10 | 5.840 | 1.126 | 306 | 194.68 | 29.77 |
| lithium GPU FP16 | 30.0 | 21.11 | 6.504 | 1.499 | 306 | 216.80 | 51.90 |
| darknet cuDNN FP16 | 30.0 | 18.37 | 6.678 | 1.437 | 306 | 222.61 | 57.70 |
| darknet cuDNN FP32 | 30.0 | 21.54 | 7.282 | 1.724 | 306 | 242.74 | 77.84 |
