"""
Evaluation entry point for trained YOLO weights.

Loads a .pt checkpoint and runs validation metrics against a requested split.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on a dataset split.")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to a trained YOLO .pt weights file.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image size.",
    )
    return parser.parse_args()


def evaluate() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    model = YOLO(str(args.weights))
    logging.info("Evaluating %s on %s split", args.weights.name, args.split)

    results = model.val(
        data=str(args.data_config),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    logging.info("Evaluation metrics: %s", results)


if __name__ == "__main__":
    evaluate()

