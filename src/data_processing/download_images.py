"""
Download Quran page images from Google Cloud Storage based on variant configuration.

This script reads variant configurations from configs/variants.yaml and downloads
all pages (1-604) for each variant listed in the configuration.

Usage:
    python src/data_processing/download_images.py \
        --config configs/variants.yaml \
        --output-dir data/processed/images
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Dict

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Quran page images from configured variants."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/variants.yaml"),
        help="Path to variants configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/images"),
        help="Directory where downloaded images will be saved.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Starting page number (default: 1).",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=604,
        help="Ending page number (default: 604).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files. Skips files by default.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def load_variants(config_path: Path) -> Dict[str, Dict[str, str]]:
    """Load variant configurations from YAML file.
    
    Supports two formats:
    1. Object format: {url: "...", extension: "webp"}
    2. String format (backward compatible): "..." (defaults to 'webp' extension)
    
    Returns:
        Dictionary mapping variant names to {url: str, extension: str}
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "variants" not in config:
        raise ValueError("Configuration file must contain a 'variants' key")

    variants_raw = config["variants"]
    if not isinstance(variants_raw, dict):
        raise ValueError("'variants' must be a dictionary")

    # Process variants - support both object and string formats
    variants: Dict[str, Dict[str, str]] = {}
    default_extension = "webp"
    
    for variant_name, variant_config in variants_raw.items():
        if isinstance(variant_config, str):
            # Backward compatible: simple string URL, default to 'webp'
            variants[variant_name] = {
                "url": variant_config,
                "extension": default_extension
            }
        elif isinstance(variant_config, dict):
            # New format: object with url and optional extension
            if "url" not in variant_config:
                raise ValueError(f"Variant '{variant_name}' must have a 'url' field")
            
            extension = variant_config.get("extension", default_extension)
            if not isinstance(extension, str):
                raise ValueError(
                    f"Variant '{variant_name}' extension must be a string "
                    "(e.g., 'webp', 'jpeg', 'png')"
                )
            
            variants[variant_name] = {
                "url": variant_config["url"],
                "extension": extension
            }
        else:
            raise ValueError(
                f"Variant '{variant_name}' must be either a string URL or "
                "an object with 'url' and optional 'extension' fields"
            )

    return variants


def download_page(
    base_url: str,
    variant_name: str,
    page_num: int,
    output_dir: Path,
    extension: str,
    overwrite: bool = False,
) -> bool:
    """Download a single page image."""
    # URL format: {base_url}/page_{page_num:03d}.{extension}
    url = f"{base_url}/page_{page_num:03d}.{extension}"
    
    # Output filename: {variant_name}_page_{page_num}.{extension}
    output_file = output_dir / f"{variant_name}_page_{page_num:03d}.{extension}"

    # Skip if file exists and not overwriting
    if output_file.exists() and not overwrite:
        logging.info("Skipping %s (already exists)", output_file.name)
        return True

    try:
        logging.info("Downloading %s...", url)
        urllib.request.urlretrieve(url, output_file)
        logging.info("Downloaded %s", output_file.name)
        return True
    except urllib.error.HTTPError as e:
        logging.error("Failed to download page %d for %s: HTTP %d", page_num, variant_name, e.code)
        return False
    except Exception as e:
        logging.error("Failed to download page %d for %s: %s", page_num, variant_name, e)
        return False


def download_variant(
    variant_name: str,
    base_url: str,
    output_dir: Path,
    start_page: int,
    end_page: int,
    extension: str,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Download all pages for a variant."""
    logging.info("Downloading variant: %s (extension: %s)", variant_name, extension)
    
    success_count = 0
    fail_count = 0

    for page_num in range(start_page, end_page + 1):
        if download_page(base_url, variant_name, page_num, output_dir, extension, overwrite):
            success_count += 1
        else:
            fail_count += 1

    logging.info(
        "Completed %s: %d successful, %d failed",
        variant_name,
        success_count,
        fail_count,
    )
    return success_count, fail_count


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    # Load variants configuration
    try:
        variants = load_variants(args.config)
    except Exception as e:
        logging.error("Failed to load configuration: %s", e)
        return

    if not variants:
        logging.warning("No variants found in configuration file")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download all variants
    total_success = 0
    total_fail = 0

    for variant_name, variant_config in variants.items():
        base_url = variant_config["url"]
        extension = variant_config["extension"]
        
        success, fail = download_variant(
            variant_name,
            base_url,
            args.output_dir,
            args.start_page,
            args.end_page,
            extension,
            args.overwrite,
        )
        total_success += success
        total_fail += fail

    logging.info(
        "Download complete! Total: %d successful, %d failed",
        total_success,
        total_fail,
    )


if __name__ == "__main__":
    main()

