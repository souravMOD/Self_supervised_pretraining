#!/usr/bin/env python3
"""
Self‑supervised pretraining script for object detection models using LightlyTrain.

This script wraps the :func:`lightly_train.train` function with a command‑line
interface and sensible defaults for production environments. It allows you to
specify the input image directory, output directory, model configuration,
pretraining method, number of epochs, batch size and image transformation
arguments. All paths are resolved at runtime so that hard‑coded platform
specific paths such as ``D:\GenV2\train\images`` are avoided.

Example::

    python pretrain_.py \
        --data /path/to/images \
        --out ./output/pretrain \
        --model ultralytics/yolo11l.yaml \
        --method distillation \
        --epochs 100 \
        --batch-size 4 \
        --min-scale 0.1 \
        --image-size 640 640

The script will create the output directory if it does not exist and will
propagate any exceptions raised by the underlying training call so that
calling processes can handle failures appropriately.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import lightly_train  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The 'lightly_train' package is required. Install it with"
        " 'pip install \"lightly-train[ultralytics]\"' and ensure your"
        " environment includes the necessary dependencies."
    ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the pretraining script.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run self‑supervised pretraining with LightlyTrain."
            " Specify the data directory, output directory, model, method,"
            " epochs, batch size and transform parameters via command‑line"
            " arguments."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help=(
            "Path to the directory containing training images. This can be a"
            " single folder or a list of folders separated by os.pathsep"
            " (e.g. ':' on Unix or ';' on Windows)."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help=(
            "Directory where training outputs (logs, checkpoints, metrics) will"
            " be saved. If the directory does not exist it will be created."
        ),
    )
    parser.add_argument(
        "--model",
        default="ultralytics/yolo11l.yaml",
        type=str,
        help=(
            "Model configuration or checkpoint identifier. See the LightlyTrain"
            " documentation for a list of supported models, e.g."
            " 'ultralytics/yolov8s.yaml' or 'ultralytics/yolo11l.pt'."
        ),
    )
    parser.add_argument(
        "--method",
        default="distillation",
        type=str,
        help=(
            "Self‑supervised pretraining method to use. The default"
            " 'distillation' is recommended."
        ),
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs to pretrain for."
    )
    parser.add_argument(
        "--batch-size",
        default=4,
        type=int,
        dest="batch_size",
        help=(
            "Batch size used during training. Adjust this based on your"
            " hardware capabilities."
        ),
    )
    parser.add_argument(
        "--min-scale",
        default=0.1,
        type=float,
        help=(
            "Minimum scale for the random resize transform. Should be a value"
            " between 0 and 1."
        ),
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=(640, 640),
        metavar=("HEIGHT", "WIDTH"),
        help="Output image size used for training, specified as HEIGHT WIDTH."
    )
    return parser.parse_args()


def build_transform_args(min_scale: float, image_size: Tuple[int, int]) -> Dict[str, Any]:
    """Construct a transformation configuration dictionary for LightlyTrain.

    Args:
        min_scale: Minimum random resize scale.
        image_size: Output (height, width) tuple.

    Returns:
        dict: Dictionary accepted by LightlyTrain's ``transform_args`` parameter.
    """
    return {
        "random_resize": {
            "min_scale": min_scale,
        },
        "image_size": image_size,
        "color_jitter": None,
    }


def main() -> None:
    """Entry point for the pretraining script."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    data_paths = args.data.split(os.pathsep)
    # Verify that all data directories exist
    for path in data_paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Data directory not found: {path}")

    # Ensure output directory exists
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform_args = build_transform_args(args.min_scale, tuple(args.image_size))

    logging.info(
        "Starting pretraining with model=%s, method=%s, epochs=%d, batch_size=%d",
        args.model,
        args.method,
        args.epochs,
        args.batch_size,
    )

    try:
        lightly_train.train(
            out=str(out_dir),
            model=args.model,
            data=data_paths if len(data_paths) > 1 else data_paths[0],
            method=args.method,
            epochs=args.epochs,
            batch_size=args.batch_size,
            transform_args=transform_args,
        )
    except Exception:  # pragma: no cover - we rethrow after logging
        logging.exception("Pretraining failed.")
        raise
    else:
        logging.info("Pretraining completed successfully. Results saved to %s", out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()