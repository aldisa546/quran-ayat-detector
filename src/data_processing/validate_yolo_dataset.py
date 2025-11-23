"""Validate and fix YOLO dataset structure.

This script checks:
1. If label files exist for all images in split files
2. If label files are correctly formatted
3. If image paths in split files are correct
4. Creates missing label files or fixes path issues
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and fix YOLO dataset structure."
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues found (create missing label files, etc.).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def validate_label_file(label_path: Path) -> Tuple[bool, str]:
    """Validate a YOLO format label file.
    
    Returns:
        (is_valid, error_message)
    """
    if not label_path.exists():
        return False, "Label file does not exist"
    
    try:
        with label_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            # Empty label file is valid (no objects)
            return True, ""
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                return False, f"Line {line_num}: Expected 5 values, got {len(parts)}"
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Validate ranges
                if not (0 <= x_center <= 1):
                    return False, f"Line {line_num}: x_center {x_center} not in [0, 1]"
                if not (0 <= y_center <= 1):
                    return False, f"Line {line_num}: y_center {y_center} not in [0, 1]"
                if not (0 < width <= 1):
                    return False, f"Line {line_num}: width {width} not in (0, 1]"
                if not (0 < height <= 1):
                    return False, f"Line {line_num}: height {height} not in (0, 1]"
                
            except ValueError as e:
                return False, f"Line {line_num}: Invalid number format - {e}"
        
        return True, ""
    
    except Exception as e:
        return False, f"Error reading label file: {e}"


def find_label_file(image_path: Path, labels_dir: Path) -> Path | None:
    """Find the corresponding label file for an image."""
    base_name = image_path.stem
    label_path = labels_dir / f"{base_name}.txt"
    
    if label_path.exists():
        return label_path
    
    # Try with different extensions
    for ext in [".txt"]:
        label_path = labels_dir / f"{base_name}{ext}"
        if label_path.exists():
            return label_path
    
    return None


def validate_split_file(
    split_file: Path,
    images_dir: Path,
    labels_dir: Path,
    fix: bool = False,
) -> Tuple[int, int, int]:
    """Validate a split file (train.txt, val.txt, test.txt).
    
    Returns:
        (total_images, valid_images, invalid_images)
    """
    if not split_file.exists():
        logging.error("Split file does not exist: %s", split_file)
        return 0, 0, 0
    
    total = 0
    valid = 0
    invalid = 0
    missing_labels = []
    invalid_labels = []
    
    with split_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total += 1
            
            # Resolve image path
            image_path = Path(line)
            if not image_path.is_absolute():
                # Try relative to project root
                image_path = Path.cwd() / image_path
            
            # Check if image exists
            if not image_path.exists():
                logging.warning(
                    "Line %d: Image not found: %s", line_num, image_path
                )
                invalid += 1
                continue
            
            # Find corresponding label file
            label_path = find_label_file(image_path, labels_dir)
            
            if label_path is None:
                logging.warning(
                    "Line %d: No label file found for %s (expected: %s)",
                    line_num,
                    image_path.name,
                    labels_dir / f"{image_path.stem}.txt",
                )
                missing_labels.append((image_path, labels_dir / f"{image_path.stem}.txt"))
                invalid += 1
                continue
            
            # Validate label file
            is_valid, error_msg = validate_label_file(label_path)
            if not is_valid:
                logging.warning(
                    "Line %d: Invalid label file %s: %s",
                    line_num,
                    label_path,
                    error_msg,
                )
                invalid_labels.append((label_path, error_msg))
                invalid += 1
                continue
            
            valid += 1
    
    # Attempt to fix issues
    if fix:
        if missing_labels:
            logging.info("Creating %d empty label files...", len(missing_labels))
            for image_path, label_path in missing_labels:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                # Create empty label file (image with no objects)
                label_path.write_text("", encoding="utf-8")
                logging.info("Created empty label file: %s", label_path)
        
        if invalid_labels:
            logging.warning(
                "Found %d invalid label files. Please fix them manually.",
                len(invalid_labels),
            )
    
    return total, valid, invalid


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    
    import yaml
    
    # Load data config
    with args.data_config.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    
    # Get paths
    base_path = Path(data_cfg.get("path", "data"))
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    
    train_split = base_path / data_cfg["train"]
    val_split = base_path / data_cfg["val"]
    test_split = base_path / data_cfg["test"]
    
    paths = data_cfg.get("paths", {})
    images_dir = Path(paths.get("images", "data/processed/images"))
    labels_dir = Path(paths.get("labels", "data/processed/labels"))
    
    if not images_dir.is_absolute():
        images_dir = Path.cwd() / images_dir
    if not labels_dir.is_absolute():
        labels_dir = Path.cwd() / labels_dir
    
    logging.info("Dataset configuration:")
    logging.info("  Base path: %s", base_path)
    logging.info("  Images dir: %s", images_dir)
    logging.info("  Labels dir: %s", labels_dir)
    logging.info("  Train split: %s", train_split)
    logging.info("  Val split: %s", val_split)
    logging.info("  Test split: %s", test_split)
    
    # Check directories exist
    if not images_dir.exists():
        logging.error("Images directory does not exist: %s", images_dir)
        return
    
    if not labels_dir.exists():
        logging.warning("Labels directory does not exist: %s", labels_dir)
        if args.fix:
            labels_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Created labels directory: %s", labels_dir)
        else:
            logging.error("Labels directory does not exist. Use --fix to create it.")
            return
    
    # Validate each split
    splits = [
        ("train", train_split),
        ("val", val_split),
        ("test", test_split),
    ]
    
    total_images = 0
    total_valid = 0
    total_invalid = 0
    
    for split_name, split_file in splits:
        logging.info("\n" + "=" * 60)
        logging.info("Validating %s split...", split_name)
        logging.info("=" * 60)
        
        total, valid, invalid = validate_split_file(
            split_file, images_dir, labels_dir, fix=args.fix
        )
        
        total_images += total
        total_valid += valid
        total_invalid += invalid
        
        logging.info(
            "%s split: %d total, %d valid, %d invalid",
            split_name,
            total,
            valid,
            invalid,
        )
    
    logging.info("\n" + "=" * 60)
    logging.info("Summary:")
    logging.info("  Total images: %d", total_images)
    logging.info("  Valid: %d", total_valid)
    logging.info("  Invalid: %d", total_invalid)
    logging.info("=" * 60)
    
    if total_invalid > 0 and not args.fix:
        logging.warning(
            "\nFound %d invalid images. Run with --fix to attempt automatic fixes.",
            total_invalid,
        )
    elif total_invalid == 0:
        logging.info("\n✓ All images have valid labels!")


if __name__ == "__main__":
    main()

