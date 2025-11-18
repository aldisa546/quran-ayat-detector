"""Utility to convert Pascal VOC XML annotations into YOLO txt labels.

Usage example:
    python src/data_processing/xml_to_yolo.py \
        --xml-dir data/annotations/xml \
        --output-dir data/labels \
        --classes configs/classes.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML annotations into YOLO txt labels."
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        required=True,
        help="Directory containing Pascal VOC XML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where YOLO txt label files will be written.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        required=False,
        help="Path to a text file with one class name per line, defining YOLO class ids.",
    )
    parser.add_argument(
        "--save-classes",
        type=Path,
        required=False,
        help="Optional path to write discovered classes if --classes is not provided.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing txt files. Skips files by default.",
    )
    parser.add_argument(
        "--skip-unknown",
        action="store_true",
        help="Skip objects whose class name is not present in the classes file. "
        "Raises an error otherwise.",
    )
    parser.add_argument(
        "--keep-difficult",
        action="store_true",
        help="Include objects marked as difficult in the VOC annotations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def load_class_mapping(classes_file: Path) -> Dict[str, int]:
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")

    class_map: Dict[str, int] = {}
    with classes_file.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            class_map[line] = idx

    if not class_map:
        raise ValueError(f"No classes found in {classes_file}")
    return class_map


def build_class_map_from_xml(xml_files: Sequence[Path]) -> Dict[str, int]:
    discovered: List[str] = []
    seen: Set[str] = set()

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            raise ValueError(f"Failed to parse {xml_path}: {exc}") from exc
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.findtext("name")
            if not name or name in seen:
                continue
            seen.add(name)
            discovered.append(name)

    if not discovered:
        raise ValueError("No object classes discovered across XML files.")

    discovered.sort()
    return {name: idx for idx, name in enumerate(discovered)}


def voc_bbox_to_yolo(
    image_width: int, image_height: int, bbox: Sequence[float]
) -> Tuple[float, float, float, float]:
    """Convert (xmin, ymin, xmax, ymax) into normalized YOLO format."""
    xmin, ymin, xmax, ymax = bbox

    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + box_width / 2.0
    y_center = ymin + box_height / 2.0

    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def extract_objects(
    root: ET.Element,
    class_map: Dict[str, int],
    keep_difficult: bool = False,
    skip_unknown: bool = False,
) -> Iterable[Tuple[int, Tuple[float, float, float, float]]]:
    size_node = root.find("size")
    if size_node is None:
        raise ValueError("Missing <size> element in XML.")

    width = int(size_node.findtext("width", default="0"))
    height = int(size_node.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions in XML.")

    for obj in root.findall("object"):
        difficult = int(obj.findtext("difficult", default="0"))
        if difficult and not keep_difficult:
            continue

        name = obj.findtext("name")
        if not name:
            raise ValueError("Object missing class name.")

        if name not in class_map:
            if skip_unknown:
                logging.debug("Skipping unknown class %s", name)
                continue
            raise KeyError(f"Class '{name}' not found in classes file.")

        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise ValueError("Object missing <bndbox> element.")

        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        yolo_bbox = voc_bbox_to_yolo(width, height, (xmin, ymin, xmax, ymax))
        yield class_map[name], yolo_bbox


def convert_xml_file(
    xml_path: Path,
    output_path: Path,
    class_map: Dict[str, int],
    overwrite: bool = False,
    keep_difficult: bool = False,
    skip_unknown: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        logging.info("Skipping %s (exists)", output_path.name)
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines: List[str] = []
    for class_id, (xc, yc, w, h) in extract_objects(
        root, class_map, keep_difficult=keep_difficult, skip_unknown=skip_unknown
    ):
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logging.info("Wrote %s", output_path.name)


def iter_xml_files(xml_dir: Path) -> Iterable[Path]:
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    return sorted(xml_dir.glob("*.xml"))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    xml_files = list(iter_xml_files(args.xml_dir))
    if not xml_files:
        logging.warning("No XML files found in %s", args.xml_dir)
        return

    if args.classes:
        class_map = load_class_mapping(args.classes)
    else:
        class_map = build_class_map_from_xml(xml_files)
        if args.save_classes:
            args.save_classes.parent.mkdir(parents=True, exist_ok=True)
            with args.save_classes.open("w", encoding="utf-8") as fh:
                for class_name in sorted(class_map, key=class_map.get):
                    fh.write(f"{class_name}\n")
            logging.info("Discovered classes written to %s", args.save_classes)
        else:
            logging.info(
                "Discovered classes (auto): %s",
                ", ".join(sorted(class_map, key=class_map.get)),
            )

    for xml_file in xml_files:
        output_file = args.output_dir / f"{xml_file.stem}.txt"
        try:
            convert_xml_file(
                xml_file,
                output_file,
                class_map,
                overwrite=args.overwrite,
                keep_difficult=args.keep_difficult,
                skip_unknown=args.skip_unknown,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.error("Failed to convert %s: %s", xml_file.name, exc)


if __name__ == "__main__":
    main()
