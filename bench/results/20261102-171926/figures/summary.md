# Benchmark 2026-11-02T17:19:35

- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- power mode: 15W
- idle floor: 4.880 W board, 0.584 W CPU+GPU+CV
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## Uncapped

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| darknet cuDNN FP32 | 86.6 | 11.55 | 11.213 | 4.050 | 612 | 129.50 | 73.14 |
| lithium GPU FP16 | 62.3 | 16.06 | 9.325 | 3.264 | 612 | 149.73 | 71.37 |
| lithium GPU TF32 | 60.0 | 16.66 | 10.227 | 3.623 | 612 | 170.36 | 89.07 |
| lithium GPU FP32 | 43.3 | 23.10 | 10.271 | 4.113 | 612 | 237.27 | 124.54 |
| darknet CPU FP32 | 2.8 | 360.37 | 9.229 | 3.505 | 306 | 3325.50 | 1567.12 |
| lithium CPU FP32 | 0.3 | 3802.88 | 5.736 | 1.199 | 306 | 22062.49 | 3293.67 |

## 30 fps cap

| engine | fps | ms | board W | compute W | GPU MHz | mJ/frame | mJ above idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| lithium GPU FP16 | 30.0 | 22.78 | 7.114 | 1.839 | 408 | 237.13 | 74.47 |
| darknet cuDNN FP32 | 30.0 | 21.56 | 7.198 | 1.720 | 306 | 239.94 | 77.27 |
| lithium GPU TF32 | 30.0 | 23.62 | 7.667 | 2.073 | 408 | 255.56 | 92.90 |
| lithium GPU FP32 | 30.0 | 23.40 | 8.711 | 3.104 | 612 | 290.35 | 127.69 |
| darknet CPU FP32 | 2.8 | 361.05 | 9.219 | 3.495 | 306 | 3328.13 | 1566.41 |
| lithium CPU FP32 | 0.3 | 3803.78 | 5.722 | 1.190 | 306 | 22007.16 | 3238.34 |
