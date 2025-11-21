#!/usr/bin/env python3
"""
Run inference on ayah classifier model.

This script loads a trained YOLOv8 classifier model and runs predictions
on all images in a test directory, outputting results to a text file.
"""

import argparse
from pathlib import Path
from typing import List

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on ayah classifier model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/ayah-classifier-v2.pt"),
        help="Path to the trained classifier model.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/processed/ayah_classifier_test"),
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/ayah_classifier_predictions.txt"),
        help="Output file path for predictions.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device identifier for inference (e.g., 'cpu', 'cuda:0', 'mps').",
    )
    return parser.parse_args()


def get_image_files(test_dir: Path) -> List[Path]:
    """Get all image files from the test directory."""
    image_extensions = {".webp", ".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f
        for f in test_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    return sorted(image_files)


def main() -> None:
    args = parse_args()

    # Validate inputs
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    # Get image files
    image_files = get_image_files(args.test_dir)
    if not image_files:
        print(f"⚠️  No images found in {args.test_dir}")
        return

    print(f"Loading model from {args.model}...")
    model = YOLO(str(args.model))

    print(f"Found {len(image_files)} images to process")
    print(f"Running inference...")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run predictions and write results
    results_lines = []
    for image_path in image_files:
        # Run prediction
        results = model.predict(
            source=str(image_path),
            save=False,
            verbose=False,
            device=args.device,
        )

        # Get the top prediction
        result = results[0]
        if result.probs is not None:
            top1_idx = int(result.probs.top1)
            top1_conf = float(result.probs.top1conf)
            class_name = result.names[top1_idx]

            # Format: filename\tpredicted_class\tclass_index\tconfidence
            results_lines.append(
                f"{image_path.name}\t{class_name}\t{top1_idx}\t{top1_conf:.6f}\n"
            )
        else:
            # Fallback if no predictions
            results_lines.append(
                f"{image_path.name}\tunknown\t-1\t0.0\n"
            )

    # Write results to file
    with open(args.output, "w") as f:
        f.writelines(results_lines)

    print(f"✅ Predictions saved to {args.output}")
    print(f"   Processed {len(results_lines)} images")


if __name__ == "__main__":
    main()

