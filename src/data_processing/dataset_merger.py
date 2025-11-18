"""
Utility to merge multiple VOC-style datasets into a single destination folder.

Usage:
    python src/data_processing/dataset_merger.py \
        --sources data/raw/dataset_v1 data/raw/dataset_v2 \
        --images-subdir images \
        --labels-subdir annotations \
        --dest data/raw/merged
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple VOC datasets into a single destination directory."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        type=Path,
        help="One or more dataset directories to merge.",
    )
    parser.add_argument(
        "--images-subdir",
        default="images",
        type=str,
        help="Sub-directory name containing images within each dataset.",
    )
    parser.add_argument(
        "--labels-subdir",
        default="annotations",
        type=str,
        help="Sub-directory name containing XML annotations within each dataset.",
    )
    parser.add_argument(
        "--dest",
        required=True,
        type=Path,
        help="Destination directory that will hold merged images and annotations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the destination if they already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def resolve_pairs(
    dataset_dir: Path, images_subdir: str, labels_subdir: str
) -> Iterable[Tuple[Path, Path]]:
    images_dir = dataset_dir / images_subdir
    labels_dir = dataset_dir / labels_subdir

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_dir}' missing '{images_subdir}' or '{labels_subdir}'."
        )

    for image_path in images_dir.iterdir():
        if not image_path.is_file():
            continue
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.xml"
        if not label_path.exists():
            logging.warning("Skipping %s (missing annotation)", image_path.name)
            continue
        yield image_path, label_path


def copy_file(src: Path, dest: Path, overwrite: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        logging.debug("Skipping %s (exists)", dest.name)
        return
    shutil.copy2(src, dest)
    logging.debug("Copied %s -> %s", src.name, dest)


def merge_datasets(args: argparse.Namespace) -> None:
    logging.info("Merging %d dataset(s)", len(args.sources))
    dest_images = args.dest / "images"
    dest_annotations = args.dest / "annotations"

    total_copies = 0
    for dataset_dir in args.sources:
        if not dataset_dir.exists():
            logging.warning("Skipping %s (not found)", dataset_dir)
            continue

        for image_path, label_path in resolve_pairs(
            dataset_dir, args.images_subdir, args.labels_subdir
        ):
            copy_file(
                image_path,
                dest_images / image_path.name,
                overwrite=args.overwrite,
            )
            copy_file(
                label_path,
                dest_annotations / label_path.name,
                overwrite=args.overwrite,
            )
            total_copies += 1

    logging.info("Merged %d image/annotation pairs into %s", total_copies, args.dest)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    merge_datasets(args)


if __name__ == "__main__":
    main()

