#!/usr/bin/env python3
"""
Crop ayah_marker bounding boxes (class '0') from images based on YOLO labels.
Crops are sorted top to bottom and named based on ayat numbers from page_ayat_meta.csv.

The CSV file should have columns: page, ayat_number
Each row represents one ayat on a page, listed from top to bottom.

Usage example:
    python src/data_processing/crop_ayah_markers.py \
        --labels-dir data/processed/labels \
        --images-dir data/processed/images \
        --output-dir data/processed/cropped_ayah_markers \
        --meta-csv src/data_processing/page_ayat_meta.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AYAH_MARKER_CLASS = 0
PAGE_PATTERN = re.compile(r"(?:page[_-]?)(\d{1,4})", re.IGNORECASE)
TRAILING_DIGITS_PATTERN = re.compile(r"(\d{1,4})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop ayah_marker bounding boxes from images based on YOLO labels. "
        "Crops are sorted top to bottom and named based on ayat numbers from CSV."
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory containing YOLO format label files (.txt).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cropped images will be saved.",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=Path(__file__).parent / "page_ayat_meta.csv",
        help="Path to CSV file with page, ayat_number columns.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Padding in pixels to add around each crop (default: 0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def extract_page_number(filename: str) -> Optional[int]:
    """Extract page number from filename if possible."""
    stem = Path(filename).stem
    match = PAGE_PATTERN.search(stem)
    if match:
        return int(match.group(1))
    trailing = TRAILING_DIGITS_PATTERN.search(stem)
    if trailing:
        return int(trailing.group(1))
    return None


def load_page_ayat_meta(csv_path: Path) -> Dict[int, List[int]]:
    """Load page ayat metadata from CSV.
    
    The CSV should have columns: page, ayat_number
    Each row represents one ayat on a page, listed from top to bottom.
    
    Returns:
        Dictionary mapping page number to list of ayat numbers (in order from top to bottom).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    page_meta: Dict[int, List[int]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                page = int(row["page"])
                ayat_number = int(row["ayat_number"])
                
                if page not in page_meta:
                    page_meta[page] = []
                page_meta[page].append(ayat_number)
            except (KeyError, ValueError) as exc:
                logging.warning("Invalid row in %s: %s - %s", csv_path, row, exc)
                continue
    
    return page_meta


def yolo_to_pixel_coords(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized coordinates to pixel coordinates (x_min, y_min, x_max, y_max)."""
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    x_min = int(x_center_px - width_px / 2.0)
    y_min = int(y_center_px - height_px / 2.0)
    x_max = int(x_center_px + width_px / 2.0)
    y_max = int(y_center_px + height_px / 2.0)

    return x_min, y_min, x_max, y_max


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO format label file and return list of (class_id, x_center, y_center, width, height)."""
    if not label_path.exists():
        return []

    labels = []
    with label_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                labels.append((class_id, x_center, y_center, width, height))
            except (ValueError, IndexError):
                continue

    return labels


def crop_ayah_markers(
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    page_meta: Dict[int, List[int]],
    padding: int = 0,
) -> int:
    """Crop all ayah_marker bounding boxes from an image and save them.
    
    Markers are sorted top to bottom and named based on ayat numbers from CSV.

    Returns:
        Number of crops saved.
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error("Failed to read image: %s", image_path)
        return 0

    img_height, img_width = image.shape[:2]

    # Extract page number and filename prefix from filename
    page_num = extract_page_number(image_path.name)
    if page_num is None:
        return 0

    # Extract filename prefix (e.g., "non-tajwid" from "non-tajwid_page_322")
    base_name = image_path.stem
    # Remove page pattern to get prefix
    filename_prefix = PAGE_PATTERN.sub("", base_name).rstrip("_-")
    if not filename_prefix:
        filename_prefix = base_name  # Fallback to full stem if pattern removal fails

    # Get ayat numbers for this page (in order from top to bottom)
    if page_num not in page_meta:
        return 0

    ayat_numbers = page_meta[page_num]
    expected_count = len(ayat_numbers)

    # Read labels
    labels = read_yolo_labels(label_path)
    if not labels:
        logging.debug("No labels found in %s", label_path)
        return 0

    # Filter for ayah_marker class (class 0) and store with coordinates for sorting
    ayah_marker_labels = []
    for class_id, x_center, y_center, width, height in labels:
        if class_id == AYAH_MARKER_CLASS:
            # Convert to pixel coordinates for sorting
            y_center_px = y_center * img_height
            x_center_px = x_center * img_width
            ayah_marker_labels.append((y_center_px, x_center_px, x_center, y_center, width, height))

    if not ayah_marker_labels:
        logging.debug("No ayah_marker (class 0) found in %s", label_path)
        return 0

    # Group markers by line (same y-coordinate within tolerance)
    # Tolerance: markers within 5% of image height are considered on the same line
    y_tolerance = img_height * 0.05
    
    # Sort all markers by y first to group them
    ayah_marker_labels.sort(key=lambda x: x[0])
    
    # Group into lines
    lines = []
    current_line = []
    current_y = None
    
    for marker in ayah_marker_labels:
        y_px = marker[0]
        if current_y is None or abs(y_px - current_y) <= y_tolerance:
            # Same line
            current_line.append(marker)
            if current_y is None:
                current_y = y_px
        else:
            # New line - sort current line by x descending (right to left) and add to lines
            current_line.sort(key=lambda x: -x[1])  # Sort by x descending
            lines.append(current_line)
            # Start new line
            current_line = [marker]
            current_y = y_px
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda x: -x[1])  # Sort by x descending
        lines.append(current_line)
    
    # Flatten lines back to single list (already sorted top to bottom, right to left within each line)
    ayah_marker_labels = [marker for line in lines for marker in line]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crop each ayah_marker sequentially
    saved_count = 0
    last_ayat_num = ayat_numbers[-1] if ayat_numbers else 0  # Get last ayat number for incrementing

    for seq_idx, (_, _, x_center, y_center, width, height) in enumerate(ayah_marker_labels):
        # Get ayat number from the list (in order from top to bottom)
        # If we run out of ayat numbers, increment from the last one
        if seq_idx < len(ayat_numbers):
            ayat_num = ayat_numbers[seq_idx]
        else:
            # Continue incrementing from the last ayat number
            increment = seq_idx - len(ayat_numbers) + 1
            ayat_num = last_ayat_num + increment

        # Track if this specific marker has issues (invalid bounding box)
        has_issues = False

        # Convert to pixel coordinates
        x_min, y_min, x_max, y_max = yolo_to_pixel_coords(
            x_center, y_center, width, height, img_width, img_height
        )

        # Apply padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_width, x_max + padding)
        y_max = min(img_height, y_max + padding)

        # Ensure valid coordinates
        if x_max <= x_min or y_max <= y_min:
            has_issues = True
            continue

        # Crop the image
        crop = image[y_min:y_max, x_min:x_max]

        # Save the cropped image with ayat number in filename
        # Add _CHECK suffix only for invalid bounding boxes
        suffix = "_CHECK" if has_issues else ""
        output_filename = f"{base_name}_ayah_{ayat_num:03d}{suffix}.webp"
        output_path = output_dir / output_filename

        success = cv2.imwrite(str(output_path), crop)
        if success:
            saved_count += 1
            logging.debug(
                "Saved crop from %s: %s (ayat %d, %dx%d)",
                image_path.name,
                output_filename,
                ayat_num,
                crop.shape[1],
                crop.shape[0],
            )
        else:
            logging.error("Failed to save crop to %s", output_path)

    return saved_count


def find_matching_pairs(
    labels_dir: Path, images_dir: Path
) -> List[Tuple[Path, Path]]:
    """Find matching label and image file pairs."""
    label_files = sorted(labels_dir.glob("*.txt"))
    pairs = []

    for label_path in label_files:
        # Try to find matching image with various extensions
        base_name = label_path.stem
        image_path = None

        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{base_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        pairs.append((label_path, image_path))

    return pairs


def main() -> None:
    args = parse_args()
    
    # Create output directory early to store log file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with console handler only
    log_format = "%(levelname)s - %(message)s"
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)

    if not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    # Load page ayat metadata
    logging.info("Loading page ayat metadata from %s", args.meta_csv)
    page_meta = load_page_ayat_meta(args.meta_csv)
    logging.info("Loaded metadata for %d page(s)", len(page_meta))

    # Find matching label-image pairs
    pairs = find_matching_pairs(args.labels_dir, args.images_dir)
    if not pairs:
        logging.info(
            "No matching label-image pairs found in %s and %s",
            args.labels_dir,
            args.images_dir,
        )
        return

    logging.info(
        "Found %d matching label-image pair(s). Processing...", len(pairs)
    )

    total_crops = 0
    for idx, (label_path, image_path) in enumerate(pairs, start=1):
        logging.info(
            "[%d/%d] Processing %s", idx, len(pairs), image_path.name
        )
        crops_saved = crop_ayah_markers(
            image_path,
            label_path,
            args.output_dir,
            page_meta,
            padding=args.padding,
        )
        total_crops += crops_saved

    logging.info(
        "Completed! Saved %d ayah_marker crop(s) to %s",
        total_crops,
        args.output_dir.resolve(),
    )
    logging.info("Files with issues have been marked with _CHECK suffix")


if __name__ == "__main__":
    main()

