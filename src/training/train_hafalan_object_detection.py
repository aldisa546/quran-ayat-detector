"""
High-level training entry point for YOLO-based ayat detection.

Reads dataset/model configuration from YAML files in `configs/` and launches
Ultralytics YOLO training with consistent logging/output layout.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import sys

import cv2
import numpy as np
import yaml
import wandb
import torch
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

try:
    import albumentations
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for ayat detection.")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_hafalan.yaml"),
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model_hafalan.yaml"),
        help="Path to model/training configuration YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where best/last weights will be copied.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in the project/name folder.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_names(data_cfg: Dict[str, Any]) -> Dict[int, str]:
    names = data_cfg.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    raise ValueError("`names` must be a list or mapping in configs/data.yaml")


def log_version_info() -> None:
    """Log version information for all key libraries."""
    logging.info("=" * 60)
    logging.info("Library Versions:")
    logging.info("=" * 60)
    
    # Core libraries
    logging.info(f"Python: {sys.version.split()[0]}")
    
    try:
        import ultralytics
        logging.info(f"ultralytics: {ultralytics.__version__}")
    except (ImportError, AttributeError):
        logging.info("ultralytics: version unknown")
    
    try:
        logging.info(f"wandb: {wandb.__version__}")
    except (ImportError, AttributeError):
        logging.info("wandb: not available")
    
    try:
        logging.info(f"numpy: {np.__version__}")
    except (ImportError, AttributeError):
        logging.info("numpy: not available")
    
    try:
        logging.info(f"opencv-python: {cv2.__version__}")
    except (ImportError, AttributeError):
        logging.info("opencv-python: not available")
    
    try:
        logging.info(f"PyYAML: {yaml.__version__}")
    except (ImportError, AttributeError):
        logging.info("PyYAML: not available")
    
    try:
        logging.info(f"PyTorch: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except (ImportError, AttributeError):
        logging.info("PyTorch: not available")
    
    if TQDM_AVAILABLE:
        try:
            logging.info(f"tqdm: {tqdm.__version__}")
        except (ImportError, AttributeError):
            logging.info("tqdm: installed but version unknown")
    else:
        logging.info("tqdm: not installed")
    
    if ALBUMENTATIONS_AVAILABLE:
        try:
            logging.info(f"albumentations: {albumentations.__version__}")
        except (ImportError, AttributeError):
            logging.info("albumentations: installed but version unknown")
    else:
        logging.info("albumentations: not installed")
    
    logging.info("=" * 60)


def train() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    # Log version information
    log_version_info()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    dataset_paths = data_cfg["paths"]
    names = resolve_names(data_cfg)

    # Initialize wandb if API key is provided
    wandb_token = os.getenv("WANDB_API_KEY")
    if wandb_token:
        wandb.login(key=wandb_token)
        logging.info("Logged in to wandb")
        
        # Initialize wandb run with project configuration
        wandb.init(
            project=model_cfg["logging"]["project"],
            name=model_cfg["logging"]["name"],
            config={
                "model": model_cfg["model"],
                "dataset": data_cfg.get("dataset", {}),
                "names": names,
                "data_config": str(args.data_config),
                "model_config": str(args.model_config),
            },
        )
        logging.info("Initialized wandb run: %s", wandb.run.url if wandb.run else "N/A")
    else:
        logging.warning("WANDB_API_KEY not set. Skipping wandb initialization.")

    yolov8_args = dict(model_cfg["model"])
    weights_path = yolov8_args.pop("weights")
    logging.info("Loading model from %s", weights_path)

    model = YOLO(weights_path)
    add_wandb_callback(model, enable_model_checkpointing=True)
    
    # Determine device: use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    else:
        logging.info("CUDA not available, using CPU")
    
    overrides = {
        "data": str(args.data_config),
        "imgsz": data_cfg["dataset"]["img_size"],
        "batch": data_cfg["dataset"]["batch_size"],
        "epochs": model_cfg["model"].get("epochs", 100),
        "workers": data_cfg["dataset"].get("num_workers", 8),
        "project": model_cfg["logging"]["project"],
        "name": model_cfg["logging"]["name"],
        "exist_ok": model_cfg["logging"].get("exist_ok", True),
        "device": device,
        **yolov8_args,
    }

    logging.info("Training overrides: %s", overrides)
    # YOLO automatically logs metrics to wandb (including mAP@0.5 and mAP@0.5:0.95)
    # when wandb is initialized and project is set. Metrics are logged per epoch as:
    # - metrics/mAP50(B) for mAP@0.5
    # - metrics/mAP50-95(B) for mAP@0.5:0.95
    results = model.train(resume=args.resume, **overrides)
    logging.info("Training finished: %s", results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = model.trainer
    if not trainer:
        logging.warning("Trainer handle missing; skipping weight copy.")
        return

    for weight_name in ("best", "last"):
        src = trainer.save_dir / f"{weight_name}.pt"
        if src.exists():
            dest = args.output_dir / f"{model_cfg['logging']['name']}_{weight_name}.pt"
            shutil.copy2(src, dest)
            logging.info("Copied %s to %s", src, dest)
        else:
            logging.warning("Missing weight file: %s", src)
    
    # Finish wandb run if it was initialized
    if wandb_token and wandb.run:
        wandb.finish()
        logging.info("Finished wandb run")


if __name__ == "__main__":
    train()

