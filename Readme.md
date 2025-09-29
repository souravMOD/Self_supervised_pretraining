# Self-Supervised Pretraining Script

This repository provides a Python script for self-supervised pretraining of object-detection backbones using the [LightlyTrain](https://docs.lightly.ai/) framework. The script wraps the `lightly_train.train` API in a command-line interface, allowing you to specify input images, output directory, model configuration, training method, and hyper-parameters without editing the code. All file paths are resolved at runtime to avoid hard-coded platform-specific values.

---

## Requirements

The application targets **Python 3.8–3.12**, which is the supported range for [LightlyTrain](https://docs.lightly.ai/).  

To use Ultralytics YOLO models with LightlyTrain you must install the package with the **ultralytics extras**:

- `lightly-train[ultralytics]` – provides the core training functionality and pulls in Ultralytics support ([docs](https://docs.lightly.ai/)).
- `torch`, `torchvision`, and `pytorch-lightning` – pinned to compatible versions as recommended by the LightlyTrain documentation ([docs](https://docs.lightly.ai/)):  
  ```text
  torch==2.5
  torchvision==0.21
  pytorch-lightning==2.5

These dependencies are listed in the accompanying requirements.txt. Install them in a fresh virtual environment with:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


⚠️ Note: Using Ultralytics models may require a commercial license. Consult the Ultralytics website
 for licensing details.

Usage

Run the script from the command line after activating your environment. At a minimum, provide paths to your image data and an output directory. You can customise the model, training method, number of epochs, batch size, minimum crop scale, and image size:

python pretrain_production.py \
  --data /path/to/images \
  --out ./output/pretrain \
  --model ultralytics/yolo11l.yaml \
  --method distillation \
  --epochs 100 \
  --batch-size 4 \
  --min-scale 0.1 \
  --image-size 640 640


The --data argument accepts either a single directory or a list of directories separated by your operating system’s path separator (: on Unix and ; on Windows).

The script verifies that each directory exists before starting training.

The output directory is created automatically.

After training completes, logs, checkpoints, and metrics are saved under the specified output folder.

Customisation

You can adjust the following parameters to suit your hardware and dataset:

Model – choose any supported Ultralytics YAML or PT model file, such as ultralytics/yolov8s.yaml or ultralytics/yolo11l.pt.

Models ending with .pt start from pretrained weights.

Models ending with .yaml start from scratch.
(docs
)

Method – select the self-supervised method (distillation, DINOv2, etc.); the default is distillation.

Epochs and batch size – control training time and memory usage.

Image transforms – specify --min-scale and --image-size to configure the random resize crop and the final resolution.

License

This code depends on LightlyTrain, which is available under the AGPL-3.0 licence, and Ultralytics YOLO models, which may require a commercial licence for certain use cases (docs
).

Please review the licences of all dependencies before deploying this script in a production environment.