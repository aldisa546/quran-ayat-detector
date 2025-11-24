#!/usr/bin/env python3
"""
Train a YOLOv8 image classifier to identify ayah markers.

The script expects a directory of cropped ayah marker images (default:
`datasets/data/processed/cropped_ayah_markers`). Each image file name is parsed to infer
the ayah label (e.g. `..._ayah_001.webp` -> `ayah_001`). The data is converted
into a YOLO classification-friendly folder structure with train/val/test
splits, after which the Ultralytics YOLO classifier is trained.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ultralytics import YOLO

LABEL_PATTERN = re.compile(r"ayah_(\d+)", re.IGNORECASE)
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 classifier on cropped ayah markers."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/data/processed/cropped_ayah_markers"),
        help="Directory containing cropped ayah marker images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/data/processed/cropped_ayah_markers_cls"),
        help=(
            "Directory where the YOLO classification dataset structure will be written."
        ),
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("datasets/data/processed/ayah_classifier_train"),
        help=(
            "Directory containing pre-split training data organized by class."
        ),
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("datasets/data/processed/ayah_classifier_test"),
        help=(
            "Directory containing validation data organized by class. "
            "Also used for test split if --test-dir is not specified."
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("datasets/data/processed/ayah_classifier_test"),
        help=(
            "Directory containing test data organized by class. "
            "Defaults to the same directory as --val-dir."
        ),
    )
    parser.add_argument(
        "--model",
        default="yolov8n-cls.pt",
        help="Base YOLO classification checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size (pixels) used during training.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device identifier for training (e.g., 'cpu', 'cuda:0', 'mps').",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("experiments"),
        help="Directory where YOLO training runs will be logged.",
    )
    parser.add_argument(
        "--run-name",
        default="ayah-classifier",
        help="Name of the training run inside the project directory.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the classification dataset structure even if it exists.",
    )
    return parser.parse_args()


def infer_label(image_path: Path) -> str | None:
    match = LABEL_PATTERN.search(image_path.stem)
    if not match:
        logging.warning(
            "Skipping %s because no ayah label could be inferred", image_path.name
        )
        return None
    return f"ayah_{int(match.group(1)):03d}"


def collect_images(source_dir: Path) -> List[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    images = sorted(source_dir.glob("*.webp"))
    if not images:
        raise FileNotFoundError(
            f"No .webp images found inside source directory: {source_dir}"
        )
    return images


def stratified_split(
    images: Iterable[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, str]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    test_ratio = 1.0 - train_ratio - val_ratio
    rng = random.Random(seed)
    grouped: Dict[str, List[Path]] = defaultdict(list)

    skipped: List[Path] = []

    for image_path in images:
        label = infer_label(image_path)
        if label is None:
            skipped.append(image_path)
            continue
        grouped[label].append(image_path)

    if skipped:
        sample = ", ".join(p.name for p in skipped[:5])
        logging.warning(
            "Skipped %d image(s) without ayah labels (e.g., %s). "
            "Ensure --source-dir points to cropped ayah markers.",
            len(skipped),
            sample,
        )

    if not grouped:
        raise RuntimeError(
            "No ayah-labelled images found. Verify that the source directory "
            "contains crops named like '*_ayah_001.webp'."
        )

    splits: Dict[str, List[Tuple[Path, str]]] = {name: [] for name in SPLIT_NAMES}

    for label, paths in grouped.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        splits["train"].extend((path, label) for path in paths[:n_train])
        splits["val"].extend((path, label) for path in paths[n_train : n_train + n_val])
        splits["test"].extend((path, label) for path in paths[n_train + n_val :])

    for name in SPLIT_NAMES:
        if not splits[name]:
            raise RuntimeError(
                f"Split '{name}' is empty. Adjust the split ratios or ensure "
                "sufficient data per class."
            )

    return splits


def link_or_copy(src: Path, dst: Path) -> None:
    """
    Create a symlink (or copy) from src to dst using absolute source paths.

    Using absolute paths avoids broken symlinks when the output directory is a
    child of the project root, where a relative symlink like "data/..." would be
    interpreted relative to the label directory instead of the repo root.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    absolute_src = src.resolve()
    try:
        os.symlink(absolute_src, dst)
    except OSError:
        shutil.copy2(absolute_src, dst)


def build_classification_dataset(
    splits: Dict[str, List[Tuple[Path, str]]],
    output_dir: Path,
    force_rebuild: bool,
) -> Path:
    if output_dir.exists() and force_rebuild:
        shutil.rmtree(output_dir)

    for split_name in SPLIT_NAMES:
        split_root = output_dir / split_name
        split_root.mkdir(parents=True, exist_ok=True)

    for split_name, entries in splits.items():
        for image_path, label in entries:
            label_dir = output_dir / split_name / label
            label_dir.mkdir(parents=True, exist_ok=True)
            target_path = label_dir / image_path.name
            link_or_copy(image_path, target_path)

    return output_dir


def prepare_existing_dataset(
    train_dir: Path,
    val_dir: Path | None,
    test_dir: Path | None,
    output_dir: Path,
    force_rebuild: bool,
) -> Path:
    def _ensure_dir(path: Path, label: str) -> Path:
        if not path:
            raise ValueError(f"{label} directory is required when --train-dir is set.")
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")
        return path.resolve()

    train_src = _ensure_dir(train_dir, "Training")
    val_src = _ensure_dir(val_dir or test_dir, "Validation/Test")
    test_src = _ensure_dir(test_dir or val_dir, "Test/Validation")

    if output_dir.exists() and force_rebuild:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Organize images by class for each split
    for split_name, src in {
        "train": train_src,
        "val": val_src,
        "test": test_src,
    }.items():
        split_output = output_dir / split_name
        split_output.mkdir(parents=True, exist_ok=True)
        
        # Find all image files in the source directory
        image_extensions = {".webp", ".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in src.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not images:
            logging.warning(f"No images found in {src}")
            continue
        
        skipped = []
        for image_path in images:
            label = infer_label(image_path)
            if label is None:
                skipped.append(image_path)
                continue
            
            # Create class directory and link/copy the image
            label_dir = split_output / label
            label_dir.mkdir(parents=True, exist_ok=True)
            target_path = label_dir / image_path.name
            link_or_copy(image_path, target_path)
        
        if skipped:
            sample = ", ".join(p.name for p in skipped[:5])
            logging.warning(
                f"Skipped {len(skipped)} image(s) in {split_name} without ayah labels (e.g., {sample})"
            )
        
        logging.info(f"Organized {len(images) - len(skipped)} images into {split_name} split")

    return output_dir


def train_classifier(
    dataset_dir: Path,
    model_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    project: Path,
    run_name: str,
    device: str | None,
) -> None:
    model = YOLO(model_name)
    model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=run_name,
        device=device,
        task="classify",
    )


def main() -> None:
    args = parse_args()
    dataset_dir = prepare_existing_dataset(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
    )
    train_classifier(
        dataset_dir=dataset_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        run_name=args.run_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()

