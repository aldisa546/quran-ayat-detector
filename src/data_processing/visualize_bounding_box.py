#!/usr/bin/env python3
"""
Visualize YOLO predictions for an image folder and export Pascal VOC XML.

Example:
    python src/data_processing/visualize_bounding_box.py \
        --model models/yolo-ayat-detector_best.pt \
        --target-folder data/processed/images \
        --output-dir experiments/visualizations/run1
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence
import xml.etree.ElementTree as ET

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize detections and export VOC XML files for an image folder."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the fine-tuned YOLO .pt weights file.",
    )
    parser.add_argument(
        "--target-folder",
        type=Path,
        required=True,
        help="Folder containing images to run inference on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/visualizations"),
        help="Base directory to store annotated images and XML files.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for filtering predictions.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. 'cuda:0' or 'cpu'). Uses YOLO default when omitted.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images recursively inside the target folder.",
    )
    return parser.parse_args()


def iter_images(folder: Path, recursive: bool) -> List[Path]:
    if recursive:
        candidates: Iterable[Path] = folder.rglob("*")
    else:
        candidates = folder.iterdir()
    images = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def clamp(value: float, min_value: int, max_value: int) -> int:
    return int(max(min_value, min(round(value), max_value)))


def to_voc_annotation(
    image_path: Path,
    image_shape: Sequence[int],
    detections,
    xml_path: Path,
) -> None:
    height, width = image_shape[:2]
    depth = image_shape[2] if len(image_shape) > 2 else 3

    annotation = ET.Element("annotation", verified="no")
    folder = ET.SubElement(annotation, "folder")
    folder.text = image_path.parent.name

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_path.name

    path_elem = ET.SubElement(annotation, "path")
    path_elem.text = str(image_path.resolve())

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = str(depth)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for det in detections:
        obj = ET.SubElement(annotation, "object")
        name_elem = ET.SubElement(obj, "name")
        name_elem.text = det["name"]

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bbox, "xmin")
        xmin.text = str(det["bbox"][0])
        ymin = ET.SubElement(bbox, "ymin")
        ymin.text = str(det["bbox"][1])
        xmax = ET.SubElement(bbox, "xmax")
        xmax.text = str(det["bbox"][2])
        ymax = ET.SubElement(bbox, "ymax")
        ymax.text = str(det["bbox"][3])

        confidence = det.get("confidence")
        if confidence is not None:
            conf_elem = ET.SubElement(obj, "confidence")
            conf_elem.text = f"{confidence:.4f}"

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def visualize_predictions(
    model: YOLO,
    image_path: Path,
    conf: float,
    device,
):
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=False,
        verbose=False,
        device=device,
    )
    result = results[0]
    boxes = result.boxes
    detections = []

    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        names = model.names

        for coords, class_id, score in zip(xyxy, cls, confs):
            xmin, ymin, xmax, ymax = coords.tolist()
            xmin = clamp(xmin, 0, result.orig_shape[1])
            ymin = clamp(ymin, 0, result.orig_shape[0])
            xmax = clamp(xmax, 0, result.orig_shape[1])
            ymax = clamp(ymax, 0, result.orig_shape[0])
            if isinstance(names, dict):
                label = names.get(class_id, str(class_id))
            else:
                label = names[class_id]
            detections.append(
                {
                    "name": label,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": float(score),
                }
            )

    return detections


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not args.target_folder.exists():
        raise FileNotFoundError(f"Target folder not found: {args.target_folder}")

    images = iter_images(args.target_folder, args.recursive)
    if not images:
        raise FileNotFoundError(
            f"No images found in {args.target_folder} "
            f"(recursive={args.recursive}) with extensions {sorted(IMAGE_EXTENSIONS)}"
        )

    logging.info("Loading model from %s", model_path)
    model = YOLO(str(model_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Running inference on %d image(s); results -> %s",
        len(images),
        args.output_dir,
    )

    for idx, image_path in enumerate(images, start=1):
        logging.info("[%d/%d] Processing %s", idx, len(images), image_path.name)
        detections = visualize_predictions(
            model=model,
            image_path=image_path,
            conf=args.conf,
            device=args.device,
        )
        output_image_path = args.output_dir / image_path.name
        shutil.copy2(image_path, output_image_path)
        orig = cv2.imread(str(image_path))
        if orig is None:
            raise ValueError(f"Unable to read image: {image_path}")
        xml_path = args.output_dir / f"{image_path.stem}.xml"
        to_voc_annotation(
            image_path=image_path,
            image_shape=orig.shape,
            detections=detections,
            xml_path=xml_path,
        )

    logging.info("Visualization complete. Outputs saved to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()

