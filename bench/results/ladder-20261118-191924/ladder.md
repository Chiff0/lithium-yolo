# Optimisation ladder

- device: chiff0-desktop, 15W, # R36 (release), REVISION: 4.2, GCID: 38685322, BOARD: generic, EABI: aarch64, DATE: Fri Dec 13 00:16:27 UTC 2024
- idle floor 4.921 W; 15.0 s window after 5.0 s settle, 3 repeat(s)
- every row: input uploaded from pinned host memory, forward incl. YOLO activation, head outputs downloaded to host memory

| # | step | commit | ms | fps | × previous | × first GPU | vs TensorRT FP16 | W @30 fps | mJ/frame uncapped |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 00 | CPU backend (im2col + GEMM) | `4e9317d` | 3846.15 | 0.3 |  |  | 0% | — | 22139.7 |
| 01 | GPU: im2col + cuBLAS FP32, NCHW | `4e9317d` | 23.30 | 42.9 | 165.07× | 1.00× | 11% | 8.73 | 238.5 |
| 02 | TF32 GEMMs | `11f53ed` | 16.72 | 59.8 | 1.39× | 1.39× | 15% | 7.66 | 171.4 |
| 03 | FP16 GEMM inputs | `f9631d0` | 16.25 | 61.5 | 1.03× | 1.43× | 15% | 7.16 | 151.1 |
| 04 | batchnorm folding | `1a2fac8` | 15.77 | 63.4 | 1.03× | 1.48× | 16% | 7.10 | 146.5 |
| 05 | NHWC + weight reorder | `a5c6794` | 14.33 | 69.8 | 1.10× | 1.63× | 18% | 6.93 | 134.8 |
| 06 | FP16 end to end | `a85800e` | 13.37 | 74.8 | 1.07× | 1.74× | 19% | 6.67 | 121.4 |
| 07 | bias + leaky fused | `c5f3245` | 12.54 | 79.8 | 1.07× | 1.86× | 20% | 6.57 | 113.7 |
| 08 | bias + leaky + maxpool fused | `ab2bc6c` | 11.68 | 85.6 | 1.07× | 1.99× | 22% | 6.44 | 105.4 |
| 09 | CUTLASS implicit GEMM | `346ddb3` | 6.85 | 145.9 | 1.70× | 3.40× | 37% | 5.79 | 60.7 |
| 10 | layer 0 on CUTLASS (padded) + autotuning | `c0a77cc` | 5.59 | 178.7 | 1.23× | 4.16× | 45% | 5.57 | 48.2 |
| 11 | direct layer-0 kernel | `420601e` | 5.60 | 178.4 | 1.00× | 4.16× | 45% | 5.57 | 48.3 |
| 12 | layer-0 conv + maxpool fused | `f784e36` | 4.85 | 206.0 | 1.15× | 4.80× | 52% | 5.46 | 40.0 |
| 13 | constant memory + device memory | `a439c86` | 2.90 | 345.1 | 1.68× | 8.04× | 87% | 5.40 | 28.9 |
| 14 | CUDA graphs | `a439c86` | 2.68 | 373.5 | 1.08× | 8.70× | 94% | 5.39 | 27.5 |
| 15 | bias + leaky in the CUTLASS epilogue | `38b7ad5` | 2.58 | 388.0 | 1.04× | 9.04× | 98% | 5.37 | 26.6 |
| 16 | YOLO convs on CUTLASS, route alias, upsample into route | `6a05cf6` | 2.37 | 421.6 | 1.09× | 9.82× | 106% | 5.36 | 25.1 |
| | **TensorRT FP16** | | 2.51 | 397.9 |  |  | 100% | 5.44 | 27.8 |
| | **TensorRT FP32** | | 5.05 | 198.1 |  |  | 50% | 6.01 | 58.7 |
| | **darknet cuDNN FP16** | | 10.15 | 98.5 |  |  | 25% | 6.63 | 103.0 |
| | **darknet cuDNN FP32** | | 11.57 | 86.4 |  |  | 22% | 7.26 | 129.9 |

— = cannot hold 30 fps
