# LFDM: Label-Efficient Fine-Tuning for Remote Sensing Imagery Segmentation with Diffusion Models

Official PyTorch implementation of **Label-Efficient Fine-Tuning for Remote
Sensing Imagery Segmentation with Diffusion Models (LFDM)**, published in
*Remote Sensing* 2025, 17(15), 2579.

[[Paper]](https://www.mdpi.com/2072-4292/17/15/2579)
[[DOI]](https://doi.org/10.3390/rs17152579)

LFDM uses a diffusion model pre-trained on unlabeled remote-sensing imagery as
a frozen representation backbone. A lightweight semantic segmentation decoder
then combines multi-scale diffusion features extracted at selected noise time
steps. This design targets label-efficient fine-tuning: the expensive visual
representation is learned from unlabeled imagery, while only a small decoder
is optimized with semantic annotations.

## Repository Status

This repository contains the cleaned training and evaluation code for the
published paper. Pretrained checkpoint download links will be added when they
are prepared for public distribution. The code is released under the MIT
License.

## Method Overview

1. Train an unconditional diffusion backbone on unlabeled remote-sensing
   patches with `train_diffusion.py`.
2. Freeze the diffusion backbone and extract decoder features at configured
   time steps such as `t = [50, 100]`.
3. Train a compact multi-scale segmentation head with limited labeled images
   using `train_segmentation.py`.
4. Evaluate with mean intersection-over-union (`mIoU`) and pixel accuracy,
   while saving predicted semantic masks.

## Installation

The code was developed with Python 3.9 and PyTorch 1.11.

```bash
conda create -n lfdm python=3.9
conda activate lfdm
pip install -r requirements.txt
```

## Data Layout

Unlabeled images for diffusion pre-training:

```text
datasets/
  unlabeled/
    train/
      patch_0001.png
      ...
```

Semantic segmentation datasets use aligned image and integer-label mask
folders. File stems must match between each `images` and `labels` directory.

```text
datasets/
  GID-5/
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/
      images/
      labels/
```

Masks are expected to contain class indices. The released templates use
`label_offset: 1`, matching the one-indexed masks used by our experiments.
Set it to `0` when using ordinary `0..K-1` masks.

## Configurations

Diffusion backbone:

- `config/diffusion_train.json`: train a diffusion backbone.
- `config/diffusion_sampling.json`: sample images from a trained backbone.

Semantic segmentation:

- `config/segmentation_gid5.json`
- `config/segmentation_gid15.json`
- `config/segmentation_dfc2022.json`
- `config/segmentation_mini_france_domain.json`

Update dataset paths and `path.resume_state` before running an experiment.
`path.resume_state` is the diffusion checkpoint prefix, without the
`_gen.pth` suffix. For evaluation, set `model_seg.resume_state` to a trained
segmentation checkpoint file.

## Training

Train the diffusion backbone:

```bash
python train_diffusion.py --config config/diffusion_train.json
```

Generate unconditional samples from a trained backbone:

```bash
python train_diffusion.py --config config/diffusion_sampling.json --phase val
```

Fine-tune the segmentation head while keeping the diffusion backbone frozen:

```bash
python train_segmentation.py --config config/segmentation_gid5.json --phase train
```

Outputs, TensorBoard logs, and checkpoints are written below `experiments/`.
The best segmentation decoder is stored as
`checkpoint/segmentation_best.pth` in the generated experiment directory.

## Evaluation

Set `model_seg.resume_state` in the selected configuration, then run:

```bash
python train_segmentation.py --config config/segmentation_gid5.json --phase test
```

The command logs `mIoU`, per-class IoU, and pixel accuracy, and writes indexed
prediction masks under the experiment `results/predictions/` directory.

## Code Organization

```text
config/                 Reproducible experiment templates
core/                   Logging and diffusion sampling utilities
data/                   Unlabeled and segmentation datasets
misc/                   Optimizer scheduler helper
model/                  Diffusion backbone and segmentation decoder
train_diffusion.py      Diffusion pre-training and sampling entry point
train_segmentation.py   Segmentation training and evaluation entry point
```

## Acknowledgements

This implementation was developed with reference to
[DDPM-CD](https://github.com/wgcban/ddpm-cd) and
[Image Super-Resolution via Iterative Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).

## Citation

```bibtex
@article{luo2025lfdm,
  title={Label-Efficient Fine-Tuning for Remote Sensing Imagery Segmentation with Diffusion Models},
  author={Luo, Yiyun and Wang, Jinnian and Sequeira, Jean and Yang, Xiankun and Wang, Dakang and Liu, Jiabin and Yao, Grekou and Mavromatis, S{\'e}bastien},
  journal={Remote Sensing},
  volume={17},
  number={15},
  pages={2579},
  year={2025},
  doi={10.3390/rs17152579}
}
```
