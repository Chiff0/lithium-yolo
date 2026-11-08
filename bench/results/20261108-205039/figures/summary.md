# Benchmark 2026-11-08T20:50:48

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 15W
- idle floor: 4.876 W board, 0.558 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 427.0 | 2.34 | 10.861 | 4.124 | 612 | 25.43 | 14.01 |
| darknet cuDNN FP16 | 98.4 | 10.16 | 10.122 | 3.426 | 612 | 102.88 | 53.32 |
| lithium GPU FP16 | 86.6 | 11.54 | 9.040 | 3.238 | 612 | 104.37 | 48.08 |
| darknet cuDNN FP32 | 86.4 | 11.57 | 11.174 | 3.975 | 612 | 129.34 | 72.90 |
| lithium GPU FP32 | 51.8 | 19.31 | 10.227 | 4.042 | 612 | 197.47 | 103.32 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| TensorRT FP16 | 30.0 | 4.94 | 5.389 | 0.819 | 306 | 179.63 | 17.10 |
| lithium GPU FP16 | 30.0 | 21.47 | 6.416 | 1.439 | 306 | 213.85 | 51.32 |
| darknet cuDNN FP16 | 30.0 | 18.41 | 6.605 | 1.393 | 306 | 220.17 | 57.64 |
| darknet cuDNN FP32 | 30.0 | 21.59 | 7.216 | 1.704 | 306 | 240.54 | 78.01 |
| lithium GPU FP32 | 30.0 | 22.81 | 8.102 | 2.628 | 510 | 270.08 | 107.55 |
