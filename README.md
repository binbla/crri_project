# Bidirectional Source-Only Calibration for Single-Source Generalized Object Detection

>   Single-source generalized object detection seeks to generalize a detector trained on a single source domain to unseen domains without access to target-domain data during training. Its central challenge is source-domain bias: source-only training encourages reliance on domain-specific patterns that fail to transfer. Recent approaches improve generalization by augmenting the training pipeline with diffusion or foundation models, but these gains often require additional models, training stages, and computation. This trend raises a question that warrants closer examination: how much improvement in domain-generalized detection performance can be achieved by explicitly managing source-domain bias within a source-trained detector, without augmenting it with auxiliary generative or language-model components? We reconsider source-domain bias not only as a nuisance to remove but also as a reference for calibrating unseen-domain features. Because the domain gap is difficult to close from either side alone, we introduce a bidirectional calibration strategy: parameter debiasing during training and feature rebiasing at inference. Parameter averaging attenuates the source-specific drift accumulated during prolonged source-only training, whereas rank-preserving feature-statistics calibration trains the downstream detector on source-referenced features and maps each channel marginal to the source-bank quantile prototype while preserving spatial rank at inference. The whole framework requires neither target-domain supervision nor auxiliary generative or language-model branches and stores only 0.13-0.26 M additional statistics. Experiments on the Diverse Weather and Artistic benchmarks demonstrate substantial and consistent improvements in out-of-domain detection performance.

---

## Highlights

-   No target-domain supervision required
-   Lightweight design (no extra heavy modules)
-   Plug-and-play components for MMDetection
-   Strong generalization across domains

## Overview

This repository provides the official implementation of our method based on the MMDetection framework.

- **Task**: Single-Source Generalized Object Detection
- **Framework**: MMDetection
- **Includes**:
    -   Training / evaluation / inference pipelines
    -   Config files for reproduction
    -   Pretrained models

---

## Installation

### Environment Setup

```shell
conda create -n sdgod python=3.10
conda activate sdgod

# pytorch: follow the 'Get Startted' page from "https://pytorch.org/get-started/locally/"
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# MMDetection: follow the guide from "https://mmdetection.readthedocs.io/en/latest/get_started.html"

export CXX=/usr/bin/g++-10
export CC=/usr/bin/gcc-10

pip install -U openmim
mim install mmengine
mim install "mmcv==2.2.0"

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e . --no-build-isolation
```

### Install This Project

```shell
# Make sure you are in the mmdetection directory.

cd projects

# Clone or Download this repo.
git clone <THIS_ANONYMOUS_REPO_URL>

cd ..

cp -r projects/crri_project/tools_extra/* .
```

#### Expected Structure

```
mmdetection
├── configs
├── mmdet
├── ...
├── projects
|   ├── ...
|   └── crri_project 
|       ├── components
|       ├── config
|       ├── hooks
|       ├── models
|       ├── README.md
|       └── tools_extra
├── run_inference.py
├── run_test.sh
├── run_train_1.sh
├── run_train_2.sh
├── tools
└── ...
```

### Dataset Preparation

#### Diverse Weather Datasets

##### Acquisition

Download dataset from: 

https://github.com/AmingWu/Single-DGOD

We recommend using the conversion script provided by MMDetection to convert standard annotation files into **COCO** format:

 `tools/dataset_converters/pascal_voc.py`

##### Datasets Configuration files

-   COCO format (**recommended**):

Set `data_root` to your dataset directory (e.g., `/path/to/DWD/`) and ensure the standard COCO annotation files (`daytime_clear_annotations_train.json`, `daytime_clear_annotations_test.json`, `daytime_foggy_annotations`, `dusk_rainy_annotations`, `night_rainy_annotations`, `night_sunny_annotations`) are correctly specified in.

`projects/crri_project/config/__base__/datasets/diverse_weather.py`

-   XML format:

`projects/crri_project/config/__base__/datasets/diverse_weather_xml.py` (***IF XML format annotation***)

##### Main configuration

-   `projects/crri_project/config/FRCNN_R101_FPN_DWD.py`

#### Reality to Art

Datasets Configuration files:

-   `projects/crri_project/config/__base__/datasets/voc2cwc_coco.py` (IF COCO format)
-   `projects/crri_project/config/__base__/datasets/voc2cwc.py` (IF VOC format)

## Training

`bash run_train1.sh` edit it before exec.

## Notes

-   Make sure dataset paths are correctly configured
-   Adjust batch size based on GPU memory (32 is recommended)
-   Recommended to use COCO-format annotations
