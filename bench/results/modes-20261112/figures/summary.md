# Power modes

- commit: f784e36-dirty
- device: chiff0-desktop, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- window 15.0 s after 5.0 s settle, 3 repeat(s)

## 15W

idle floor 4.906 W

| engine | max fps | mJ/frame | W @60 | W @30 | W @15 |
|---|---:|---:|---:|---:|---:|
| TensorRT FP16 | 427.1 | 25.6 | 5.85 | 5.40 | 5.16 |
| lithium GPU FP16 CUTLASS | 349.4 | 28.7 | 5.87 | 5.42 | 5.20 |
| lithium GPU FP16 CUTLASS graphs | 377.9 | 27.1 | 5.84 | 5.38 | 5.16 |
| TensorRT FP32 | 207.6 | 55.1 | 6.97 | 5.96 | 5.45 |
| darknet cuDNN FP16 | 98.3 | 103.2 | 8.21 | 6.63 | 5.80 |
| darknet cuDNN FP32 | 86.3 | 129.7 | 9.38 | 7.26 | 6.10 |
| lithium GPU FP32 | 65.8 | 163.3 | 10.30 | 7.68 | 6.34 |

## MAXN

idle floor 4.917 W

| engine | max fps | mJ/frame | W @60 | W @30 | W @15 |
|---|---:|---:|---:|---:|---:|
| TensorRT FP16 | 669.8 | 27.5 | 5.86 | 5.41 | 5.20 |
| lithium GPU FP16 CUTLASS | 512.2 | 30.9 | 5.91 | 5.43 | 5.21 |
| lithium GPU FP16 CUTLASS graphs | 567.7 | 31.2 | 5.94 | 5.53 | 5.26 |
| TensorRT FP32 | 311.6 | 61.7 | 7.03 | 5.95 | 5.49 |
| darknet cuDNN FP16 | 154.2 | 107.7 | 8.25 | 6.65 | 5.82 |
| darknet cuDNN FP32 | 135.5 | 138.3 | 10.33 | 7.31 | 6.09 |
| lithium GPU FP32 | 109.3 | 171.7 | 11.52 | 7.72 | 6.37 |

— = cannot hold that frame rate in this mode
