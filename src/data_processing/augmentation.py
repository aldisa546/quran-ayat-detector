"""
Apply Albumentations transforms to YOLO-format datasets.

This script reads images from `data/processed/images` and their corresponding
YOLO txt labels from `data/processed/labels`, applies configurable augmentations,
and writes augmented samples back to disk (with suffixes) while keeping labels
in sync.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment YOLO datasets.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/processed/images"),
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/processed/labels"),
        help="Directory containing YOLO txt label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/augmented"),
        help="Directory where augmented samples will be stored.",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=2,
        help="Number of augmented variants to generate per input image.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="aug",
        help="Suffix appended to the base filename for augmented samples.",
    )
    parser.add_argument(
        "--img-ext",
        type=str,
        default=".webp",
        help="Image extension to use when writing augmented images.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def build_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),
            A.RandomBrightnessContrast(p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=0.6,
            ),
            A.MotionBlur(p=0.2),
            A.CLAHE(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def read_labels(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    boxes: List[List[float]] = []
    class_ids: List[int] = []

    if not label_path.exists():
        return boxes, class_ids

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, *bbox = parts
            class_ids.append(int(cls))
            boxes.append([float(x) for x in bbox])

    return boxes, class_ids


def write_labels(label_path: Path, class_ids: List[int], boxes: List[List[float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cls, bbox in zip(class_ids, boxes):
            f.write(f\"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\\n\")


def augment_dataset(args: argparse.Namespace) -> None:
    transform = build_pipeline()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / \"labels\").mkdir(parents=True, exist_ok=True)
    (args.output_dir / \"images\").mkdir(parents=True, exist_ok=True)

    for image_path in sorted(args.images_dir.glob(\"*\")):
        if not image_path.is_file():
            continue
        label_path = args.labels_dir / f\"{image_path.stem}.txt\"
        boxes, class_ids = read_labels(label_path)
        if not boxes:
            logging.debug(\"Skipping %s (no boxes)\", image_path.name)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(\"Failed to read %s\", image_path)
            continue

        for idx in range(args.samples_per_image):
            transformed = transform(
                image=image, bboxes=boxes, class_labels=class_ids
            )
            aug_image = transformed[\"image\"]
            aug_boxes = transformed[\"bboxes\"]
            aug_labels = transformed[\"class_labels\"]

            if not aug_boxes:
                logging.debug(\"Skipping augmented sample (no boxes remain)\")
                continue

            aug_name = f\"{image_path.stem}_{args.suffix}{idx}\"
            img_out = args.output_dir / \"images\" / f\"{aug_name}{args.img_ext}\"
            lbl_out = args.output_dir / \"labels\" / f\"{aug_name}.txt\"

            cv2.imwrite(str(img_out), aug_image)
            write_labels(lbl_out, aug_labels, aug_boxes)
            logging.debug(\"Wrote %s\", img_out.name)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=\"%(levelname)s - %(message)s\",
    )
    augment_dataset(args)


if __name__ == \"__main__\":
    main()

