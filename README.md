# Revisiting Single-Source Generalized Object Detection

>   Single-domain generalized object detection (SDGOD) aims to train a robust object detector that performs well across multiple target domains using only a single source domain, without any supervision from the target domains. The core challenge lies in learning unbiased representations from a single, inherently biased source domain. Existing methods typically introduce additional prior information to simulate the data distribution of unknown domains in order to enhance the model's representational capacity, but this also increases the model's parameter count and computational burden. We revisit the single-domain generalization problem and propose a more concise solution that requires no knowledge of any target domain. Specifically, we construct a framework consisting of two key components: CR$^2$I, a memory bank that stores channel-wise statistics from the source domain to mitigate the impact of domain bias, and PTS, a parameter smoothing strategy that suppresses source-specific biases by dynamically updating model weights during training. Extensive experiments on benchmarks demonstrate that our method outperforms state-of-the-art significantly by coupling these two components.

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
