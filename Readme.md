# Self-Supervised Pretraining Script

This repository provides a Python script for **self-supervised pretraining of object-detection backbones** using the [LightlyTrain](https://docs.lightly.ai/) framework.  

The script wraps the `lightly_train.train` API in a command-line interface, so you can specify:

- Input images  
- Output directory  
- Model configuration  
- Training method  
- Hyperparameters  

—all without editing the code. Paths are resolved at runtime to avoid hard-coded, platform-specific values.

---

## Requirements

This script supports **Python 3.8–3.12**, the officially supported range for [LightlyTrain](https://docs.lightly.ai/).  

To train Ultralytics YOLO models with LightlyTrain, install the package with the **ultralytics extras**:

- `lightly-train[ultralytics]` – core training functionality with YOLO integration  
- `torch`, `torchvision`, and `pytorch-lightning` – pinned to recommended versions:  
  ```text
  torch==2.5
  torchvision==0.21
  pytorch-lightning==2.5

### Dependencies are listed in requirements.txt. Install them in a fresh virtual environment:

python3 -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt


## Usage

### Run the script from the command line after activating your environment. At minimum, provide paths to your image dataset and an output directory. You can also customise the model, method, epochs, batch size, crop scale, and image size.

python pretrain_production.py \
  --data /path/to/images \
  --out ./output/pretrain \
  --model ultralytics/yolo11l.yaml \
  --method distillation \
  --epochs 100 \
  --batch-size 4 \
  --min-scale 0.1 \
  --image-size 640 640

## Notes

--data accepts either a single directory or multiple directories separated by your OS path separator (: on Unix, ; on Windows).

The script validates all input directories before training.

The output directory is created automatically if missing.

After training, logs, checkpoints, and metrics are saved in the output folder.

## Customisation

You can adjust the following parameters to suit your dataset and hardware:

## Model

Supports any Ultralytics YAML or PT file, e.g. ultralytics/yolov8s.yaml or ultralytics/yolo11l.pt.

.yaml → train from scratch

.pt → start from pretrained weights

Method

## Choose the self-supervised method: distillation (default), DINOv2, etc.

Epochs & Batch Size

Balance training duration and memory usage.

Image Transforms

Use --min-scale for the minimum crop scale.

Use --image-size to set final resolution (e.g. 640 640).

## License

This project depends on:

LightlyTrain (AGPL-3.0 licence)

Ultralytics YOLO models (may require a commercial licence for certain use cases)
