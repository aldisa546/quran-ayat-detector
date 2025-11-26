import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from xml.etree.ElementTree import Element, SubElement, tostring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert bounding boxes stored in a CSV file to Pascal VOC XML/JSON."
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        type=Path,
        help="Path to the CSV file containing the bounding boxes.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where Pascal VOC annotations will be written.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        help="Optional directory where Pascal-style JSON annotations will be written. Defaults to <output-dir>/json.",
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        help="Extension used when building image file names. Default: .jpg",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=100,
        help="Image width used for the Pascal <size> block. Default: 100",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Optional fixed image height. "
        "If omitted, height is derived as image_width * aspect_ratio.",
    )
    parser.add_argument(
        "--image-depth",
        type=int,
        default=3,
        help="Image depth used for the Pascal <size> block. Default: 3",
    )
    parser.add_argument(
        "--page-column",
        default="coord_page",
        help="Name of the column that identifies the page or image id.",
    )
    parser.add_argument(
        "--x-min-column",
        default="coord_left",
        help="Name of the column that stores the left coordinate.",
    )
    parser.add_argument(
        "--x-max-column",
        default="coord_right",
        help="Name of the column that stores the right coordinate.",
    )
    parser.add_argument(
        "--y-min-column",
        default="coord_top",
        help="Name of the column that stores the top coordinate.",
    )
    parser.add_argument(
        "--y-max-column",
        default="coord_bottom",
        help="Name of the column that stores the bottom coordinate.",
    )
    parser.add_argument(
        "--class-column",
        default="coord_type",
        help="Name of the column that stores the class label.",
    )
    parser.add_argument(
        "--aspect-ratio-column",
        default="coord_aspect_ratio",
        help="Column containing the aspect ratio used in Flutter exports.",
    )
    parser.add_argument(
        "--default-aspect-ratio",
        type=float,
        default=1.0,
        help="Fallback aspect ratio when the column is missing/blank.",
    )
    return parser.parse_args()


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def pretty_xml(element: Element) -> str:
    """Return a pretty printed XML string for the Element."""
    from xml.dom import minidom

    rough_string = tostring(element, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def build_pascal_xml(
    filename: str,
    size: Dict[str, int],
    objects: List[Dict[str, float]],
) -> Element:
    annotation = Element("annotation")

    folder = SubElement(annotation, "folder")
    folder.text = "dataset"

    filename_node = SubElement(annotation, "filename")
    filename_node.text = filename

    size_node = SubElement(annotation, "size")
    for key in ("width", "height", "depth"):
        child = SubElement(size_node, key)
        child.text = str(size[key])

    segmented = SubElement(annotation, "segmented")
    segmented.text = "0"

    for obj in objects:
        object_node = SubElement(annotation, "object")
        name = SubElement(object_node, "name")
        name.text = obj["name"]

        pose = SubElement(object_node, "pose")
        pose.text = "Unspecified"

        truncated = SubElement(object_node, "truncated")
        truncated.text = "0"

        difficult = SubElement(object_node, "difficult")
        difficult.text = "0"

        bndbox = SubElement(object_node, "bndbox")
        xmin = SubElement(bndbox, "xmin")
        xmin.text = f"{obj['xmin']:.6f}"

        ymin = SubElement(bndbox, "ymin")
        ymin.text = f"{obj['ymin']:.6f}"

        xmax = SubElement(bndbox, "xmax")
        xmax.text = f"{obj['xmax']:.6f}"

        ymax = SubElement(bndbox, "ymax")
        ymax.text = f"{obj['ymax']:.6f}"

    return annotation


def annotation_to_json_dict(filename: str, size: Dict[str, int], objects: List[Dict[str, float]]) -> Dict:
    return {
        "filename": filename,
        "size": size,
        "objects": objects,
    }


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    json_dir = args.json_dir or (output_dir / "json")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.csv_path)

    required_columns = {
        args.page_column,
        args.x_min_column,
        args.x_max_column,
        args.y_min_column,
        args.y_max_column,
        args.class_column,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    grouped = df.groupby(args.page_column, sort=True)
    width_px = args.image_width

    for page_id, group in grouped:
        image_name = f"page_{int(page_id):03d}{args.image_ext}"

        page_ratio = args.default_aspect_ratio
        if args.aspect_ratio_column and args.aspect_ratio_column in group:
            first = group[args.aspect_ratio_column].iloc[0]
            try:
                page_ratio = float(first)
            except (TypeError, ValueError):
                page_ratio = args.default_aspect_ratio
        height_px = args.image_height or int(round(width_px * page_ratio))
        size = {"width": width_px, "height": height_px, "depth": args.image_depth}

        objects = []
        for _, row in group.iterrows():
            class_name = str(row[args.class_column])
            left = (float(row[args.x_min_column]) * width_px) / 100.0
            right = (float(row[args.x_max_column]) * width_px) / 100.0
            top = (float(row[args.y_min_column]) * width_px * page_ratio) / 100.0
            bottom = (float(row[args.y_max_column]) * width_px * page_ratio) / 100.0

            objects.append(
                {
                    "name": class_name,
                    "xmin": min(left, right),
                    "xmax": max(left, right),
                    "ymin": min(top, bottom),
                    "ymax": max(top, bottom),
                }
            )

        xml_element = build_pascal_xml(image_name, size, objects)
        xml_path = output_dir / f"{Path(image_name).stem}.xml"
        xml_path.write_text(pretty_xml(xml_element), encoding="utf-8")

        json_dict = annotation_to_json_dict(image_name, size, objects)
        json_path = json_dir / f"{Path(image_name).stem}.json"
        json_path.write_text(json.dumps(json_dict, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

