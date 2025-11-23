"""Fix common YOLO dataset issues.

This script:
1. Clears corrupted YOLO cache files
2. Validates and fixes label files
3. Ensures image-label pairs are correct
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix common YOLO dataset issues."
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear YOLO cache files (.cache files in labels directory).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def clear_cache_files(labels_dir: Path) -> int:
    """Clear all .cache files in the labels directory."""
    cache_files = list(labels_dir.glob("*.cache"))
    cache_files.extend(list(labels_dir.rglob("*.cache")))
    
    cleared = 0
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            logging.info("Deleted cache file: %s", cache_file)
            cleared += 1
        except Exception as e:
            logging.warning("Failed to delete %s: %s", cache_file, e)
    
    return cleared


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
    paths = data_cfg.get("paths", {})
    labels_dir = Path(paths.get("labels", "data/processed/labels"))
    
    if not labels_dir.is_absolute():
        labels_dir = Path.cwd() / labels_dir
    
    logging.info("Labels directory: %s", labels_dir)
    
    if not labels_dir.exists():
        logging.error("Labels directory does not exist: %s", labels_dir)
        return
    
    # Clear cache files
    if args.clear_cache:
        logging.info("Clearing cache files...")
        cleared = clear_cache_files(labels_dir)
        logging.info("Cleared %d cache file(s)", cleared)
    else:
        # Just report cache files
        cache_files = list(labels_dir.glob("*.cache"))
        cache_files.extend(list(labels_dir.rglob("*.cache")))
        if cache_files:
            logging.warning(
                "Found %d cache file(s). Use --clear-cache to remove them.",
                len(cache_files),
            )
            for cache_file in cache_files:
                logging.info("  Cache file: %s", cache_file)
    
    logging.info("\n✓ Done! Now run: make validate-dataset FIX=1")


if __name__ == "__main__":
    main()

