"""Fix split files by removing entries for missing images.

This script:
1. Reads train.txt, val.txt, test.txt
2. Checks if each image exists
3. Checks if corresponding label exists
4. Writes back only valid entries
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix split files by removing entries for missing images."
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def find_label_file(image_path: Path, labels_dir: Path) -> Path | None:
    """Find the corresponding label file for an image."""
    base_name = image_path.stem
    label_path = labels_dir / f"{base_name}.txt"
    
    if label_path.exists():
        return label_path
    
    return None


def fix_split_file(
    split_file: Path,
    images_dir: Path,
    labels_dir: Path,
    base_path: Path,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Fix a split file by removing entries for missing images.
    
    Returns:
        (original_count, valid_count)
    """
    if not split_file.exists():
        logging.warning("Split file does not exist: %s", split_file)
        return 0, 0
    
    valid_entries = []
    removed_count = 0
    
    with split_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    
    original_count = len([l for l in lines if l.strip()])
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # Resolve image path - try multiple strategies
        image_path = None
        original_path = Path(line)
        
        # Strategy 1: Try as-is relative to base_path
        if not original_path.is_absolute():
            candidate = base_path / original_path
            if candidate.exists():
                image_path = candidate
        
        # Strategy 2: Try by filename in images_dir
        if image_path is None or not image_path.exists():
            filename = original_path.name
            candidate = images_dir / filename
            if candidate.exists():
                image_path = candidate
                # Update the line to use the correct relative path
                # Calculate relative path from base_path to image_path
                try:
                    rel_path = image_path.relative_to(base_path)
                    line = str(rel_path)
                except ValueError:
                    # If can't make relative, use the filename
                    line = f"datasets/data/processed/images/{filename}"
        
        # Strategy 3: Try resolving the path components
        if image_path is None or not image_path.exists():
            # Extract filename and try in images_dir
            filename = original_path.name
            candidate = images_dir / filename
            if candidate.exists():
                image_path = candidate
                # Update line to use correct path
                try:
                    rel_path = image_path.relative_to(base_path)
                    line = str(rel_path)
                except ValueError:
                    line = f"datasets/data/processed/images/{filename}"
        
        # Check if image exists
        if image_path is None or not image_path.exists():
            logging.debug(
                "Line %d: Removing entry - image not found: %s", line_num, original_path
            )
            removed_count += 1
            continue
        
        # Check if label exists
        label_path = find_label_file(image_path, labels_dir)
        if label_path is None:
            logging.debug(
                "Line %d: Removing entry - label not found for: %s", line_num, image_path.name
            )
            removed_count += 1
            continue
        
        # Entry is valid
        valid_entries.append(line)
    
    if not dry_run:
        # Write back valid entries
        with split_file.open("w", encoding="utf-8") as f:
            for entry in valid_entries:
                f.write(entry + "\n")
        logging.info("Updated %s: %d valid entries (removed %d)", split_file.name, len(valid_entries), removed_count)
    else:
        logging.info("Would update %s: %d valid entries (would remove %d)", split_file.name, len(valid_entries), removed_count)
    
    return original_count, len(valid_entries)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    
    # Load data config
    with args.data_config.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    
    # Get paths
    base_path = Path(data_cfg.get("path", "."))
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
    
    # Check directories exist
    if not images_dir.exists():
        logging.error("Images directory does not exist: %s", images_dir)
        return
    
    if not labels_dir.exists():
        logging.error("Labels directory does not exist: %s", labels_dir)
        return
    
    # Fix each split
    splits = [
        ("train", train_split),
        ("val", val_split),
        ("test", test_split),
    ]
    
    total_original = 0
    total_valid = 0
    
    for split_name, split_file in splits:
        logging.info("\n" + "=" * 60)
        logging.info("Fixing %s split...", split_name)
        logging.info("=" * 60)
        
        original, valid = fix_split_file(
            split_file, images_dir, labels_dir, base_path, dry_run=args.dry_run
        )
        
        total_original += original
        total_valid += valid
    
    logging.info("\n" + "=" * 60)
    logging.info("Summary:")
    logging.info("  Total original entries: %d", total_original)
    logging.info("  Total valid entries: %d", total_valid)
    logging.info("  Removed entries: %d", total_original - total_valid)
    logging.info("=" * 60)
    
    if args.dry_run:
        logging.info("\nThis was a dry run. Run without --dry-run to apply changes.")
    else:
        logging.info("\n✓ Split files fixed!")


if __name__ == "__main__":
    main()

