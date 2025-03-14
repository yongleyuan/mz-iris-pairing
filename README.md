# Monozygotic Iris Pairing

Implementation of the paper `A Siamese Network to Detect If Two Iris Images Are Monozygotic`

## Data Preparation

### MZ Dataset Acquisition

Our dataset is separated into two parts, synthetic MZ/NMZ pairs and natural MZ/NMZ pairs.

#### Synthetic MZ/NMZ Pairs

This part of the dataset will be available on `anonymized for now`.

#### Natural MZ/NMZ Pairs

We currently do not have the right to distribute the natural MZ/NMZ pairs, but we are working on the licensing.

### Data Organization

To use the code, your data must have metadata csv file with at least `sequenceid_left` and `sequenceid_right` columns, as well as a directory of all images. The images must be named `<sequenceid>.tiff`. The segmentation mask images must be in a separate directory and named `<sequenceid>_seg_mask.png`.

The code assumes the format of the iris images are in `tiff` and that of the segmentation masks are in `png`, since these are the formats used in our original dataset. Please feel free to change that in `dataset.py` if you are using your own data.

## Setup

For maximum reproducability, using Python 3.10 and install pip requirements with `pip install -r requirements.txt`.

> [!Note]
> If you are using a GPU(s), make sure to install the respective CUDA versions.

## Usage

### Training

Use `python train.py ...` with the following options:

```mardown
  -h,   --help                show help message and exit
  -b,   --backbone            options: "resnet18", "resnet34", "resnet50", "resnet101"; default: "resnet18"
  -w,   --init-weight-path    path to initial weights; required
  -d,   --data-path           path to data directory; required
  -i,   --image-dir           path to image directory; required
  -ts,  --train-split         propotion of data to use for training; default: 0.7
  -bs,  --batch-size          batch size; default: 32
  -e,   --epochs              number of epochs; default: 20
  -l,   --lr                  learning rate of Adam optimizer; default: 1e-4
  -t,   --thres               decision threshold; default: 0.5
  -s,   --suffix              suffix for stats and model saving names; optional
  -m,   --mask                flag to use segmentation mask, requires `--mask-dir`; optional
  -md,  --mask-dir            path to segmentation mask directory; required if using `--mask`
  -mi,  --mask-inverse        flag to inversely mask images; optional
  -rs,  --random-seed         integer randome seed; optional
  --save-all-models           flag to save all models instead of just the best one; optional
```

### Inference

Use `python infer.py ...` with the following options:

```mardown
  -h,   --help                show help message and exit
  -b,   --backbone            options: "resnet18", "resnet34", "resnet50", "resnet101"; default: "resnet18"
  -w,   --weight-path         path to model weights; required
  -d,   --data-path           path to data directory; required
  -i,   --image-dir           path to image directory; required
  -bs,  --batch-size          batch size; default: 1
  -t,   --thres               decision threshold; default: 0.5
  -s,   --suffix              suffix for stats and model saving names; optional
  -m,   --mask                flag to use segmentation mask, requires `--mask-dir`; optional
  -md,  --mask-dir            path to segmentation mask directory; required if using `--mask`
  -mi,  --mask-inverse        flag to inversely mask images; optional
```
