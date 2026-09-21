# Lithium-YOLO

Reference tensors for building a `yolov3-tiny` CPU forward pass against
Darknet as ground truth.

Nothing here is engine code. These are Darknet's own intermediate tensors,
dumped layer by layer, so that each op you write has an oracle the moment it
compiles rather than after the whole network is wired up.

## What is here

```
reference/
  dog/                        24 layer dumps + the input tensor
  dog-letterbox/              the same, with letterboxing enabled
  yolov3-tiny-letterbox.cfg   the cfg used for the letterbox run
  manifest.txt                sha256 + byte count for all 50 files
```

Files are raw little-endian FP32, contiguous NCHW, batch 1, no header. The
shape is in the filename, so no sidecar metadata is needed:

```
input_c3_h416_w416.bin        the tensor fed to layer 0
layer_00_c16_h416_w416.bin    ... through layer_23_c255_h26_w26.bin
```

Read one as `count = c*h*w` floats and reshape to `(c, h, w)`.

## Read this before comparing anything

**The two directories use different preprocessing, and `dog/` is not
letterboxed.** Darknet only letterboxes when the cfg sets `letter_box=1` in
`[net]`, and stock `yolov3-tiny.cfg` does not. Its `detector test` path calls
`resize_image()` instead, which squashes 768×576 to 416×416 and does not
preserve aspect ratio.

|  | `dog/` | `dog-letterbox/` |
| --- | --- | --- |
| preprocessing | plain resize, aspect distorted | letterbox, 0.5 padding |
| input padding | none | rows 0–51 and 364–415 are exactly 0.5 |
| dog | 80.92% | 56.39% |
| bicycle | 37.63% | 58.64% |
| car | 71.05% | 61.75% |
| truck | 61.70% | 55.35% |

Those detections differ substantially, so decide which convention you are
implementing before you start, and diff against the matching directory.
`dog-letterbox/` is the one that matches a letterbox implementation: pad
value 0.5, centered with integer division, RGB planar in [0,1], align-corners
bilinear resize.

**Layer 16 and layer 23 are not plain copies of the head convs.** Darknet's
yolo layer applies a logistic in place, verified against these dumps:

| channel | layer 15 → layer 16 |
| --- | --- |
| tx, ty | sigmoid applied |
| tw, th | passed through raw |
| objectness | sigmoid applied |
| all 80 class logits | sigmoid applied |

So layer 15 is the raw head convolution and layer 16 is partially activated.
Diff your conv against **15**, and your decoder's pre-box stage against
**16** — comparing against 16 with your own sigmoid applied double-counts it.

**Batch norm is already folded.** Darknet calls `fuse_conv_batchnorm()` before
inference, the same as folding at load time. These dumps are post-fold, so
they are directly comparable.

## Shapes

| layer | type | out c×h×w | | layer | type | out c×h×w |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | conv | 16×416×416 | | 12 | conv | 1024×13×13 |
| 1 | maxpool | 16×208×208 | | 13 | conv 1×1 | 256×13×13 |
| 2 | conv | 32×208×208 | | 14 | conv | 512×13×13 |
| 3 | maxpool | 32×104×104 | | 15 | conv 1×1 linear | 255×13×13 |
| 4 | conv | 64×104×104 | | 16 | yolo | 255×13×13 |
| 5 | maxpool | 64×52×52 | | 17 | route 13 | 256×13×13 |
| 6 | conv | 128×52×52 | | 18 | conv 1×1 | 128×13×13 |
| 7 | maxpool | 128×26×26 | | 19 | upsample | 128×26×26 |
| 8 | conv | 256×26×26 | | 20 | route 19,8 | 384×26×26 |
| 9 | maxpool | 256×13×13 | | 21 | conv | 256×26×26 |
| 10 | conv | 512×13×13 | | 22 | conv 1×1 linear | 255×26×26 |
| 11 | maxpool 2×2/1 | 512×13×13 | | 23 | yolo | 255×26×26 |

Layer 11 is the asymmetric-padding maxpool: 2×2 stride 1, and it must leave
the spatial size at 13×13. If it does not, every later layer is garbage.

## Reading the diffs

Look for **the first layer where the error jumps by orders of magnitude**,
not the layer with the largest error. The largest is almost always the last,
because FP32 error accumulates with depth and a different summation order is
not a bug. A jump from 1e-6 to 1e-2 at layer 7 is a bug, at layer 7.

Rough FP32 expectations: layers 0–5 around 1e-7 to 1e-6, layers 6–15 up to
~1e-5, the heads up to ~1e-4 on activations with magnitude in the tens.

If layer 0 already disagrees, it is the input tensor or the weights, not your
kernel.

## Regenerating

The dumps come from `~/Projects/darknet-ref`, a pruned CPU-only Darknet built
by `~/Projects/yolo-engine/tools/setup_darknet_ref.sh`. Its patch adds a dump
hook to `forward_network()` that is inert unless `DARKNET_DUMP_DIR` is set:

```bash
cd ~/Projects/darknet-ref
DARKNET_DUMP_DIR=~/Projects/Lithium-YOLO/reference/dog \
  ./build/src-cli/darknet detector test \
  cfg/coco.data cfg/yolov3-tiny.cfg \
  ~/Projects/yolo-engine/weights/yolov3-tiny.weights \
  artwork/dog.jpg -dont_show
```

Swap in `reference/yolov3-tiny-letterbox.cfg` for the letterboxed set. On a
multi-image or video run each frame overwrites the last.

Verify integrity with `sha256sum -c` against `manifest.txt` — worth doing
after any unclean shutdown, since a crash on btrfs can silently truncate
recently written files to zero length.
