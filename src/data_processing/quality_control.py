"""Quality control utility to compare ayah_marker counts with page metadata."""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

PAGE_PATTERN = re.compile(r"(?:page[_-]?)(\d{1,4})", re.IGNORECASE)
TRAILING_DIGITS_PATTERN = re.compile(r"(\d{1,4})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that ayah_marker counts in XML annotations "
        "match the ayat_total values declared in page_meta.csv."
    )
    default_csv = Path(__file__).resolve().parent / "page_meta.csv"
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Path to CSV containing page metadata (default: {default_csv})",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        required=True,
        help="Directory containing Pascal VOC XML files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional debugging information.",
    )
    return parser.parse_args()


def load_page_meta(csv_path: Path) -> Tuple[Dict[int, int], List[int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    page_to_totals: Dict[int, List[int]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                page = int(row["page"])
                total = int(row["ayat_total"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid row in {csv_path}: {row}") from exc
            page_to_totals[page].append(total)

    duplicates: List[int] = []
    canonical: Dict[int, int] = {}
    for page, totals in page_to_totals.items():
        if len(totals) > 1:
            duplicates.append(page)
        canonical[page] = totals[0]

    return canonical, duplicates


def extract_page_number(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    match = PAGE_PATTERN.search(stem)
    if match:
        return int(match.group(1))
    trailing = TRAILING_DIGITS_PATTERN.search(stem)
    if trailing:
        return int(trailing.group(1))
    return None


def index_xml_files(xml_dir: Path) -> Tuple[Dict[int, List[Path]], List[Path]]:
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    page_to_files: Dict[int, List[Path]] = defaultdict(list)
    unparsed: List[Path] = []
    for xml_path in sorted(xml_dir.glob("*.xml")):
        page = extract_page_number(xml_path.name)
        if page is None:
            unparsed.append(xml_path)
            continue
        page_to_files[page].append(xml_path)
    return page_to_files, unparsed


def count_ayah_markers(xml_path: Path) -> int:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse {xml_path}: {exc}") from exc

    root = tree.getroot()
    count = 0
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            continue
        if name.strip().lower() == "ayah_marker":
            count += 1
    return count


@dataclass
class Issue:
    page: int
    expected: int
    actual: Optional[int]
    xml_files: List[Path]
    message: str


def run_quality_control(
    csv_path: Path,
    xml_dir: Path,
    verbose: bool = False,
) -> Tuple[List[Issue], List[int], List[Path]]:
    meta, duplicate_pages = load_page_meta(csv_path)
    index, unparsed_files = index_xml_files(xml_dir)

    issues: List[Issue] = []

    for page, expected in sorted(meta.items()):
        files = index.get(page)
        if not files:
            issues.append(
                Issue(
                    page=page,
                    expected=expected,
                    actual=None,
                    xml_files=[],
                    message="Missing XML annotation",
                )
            )
            continue

        # Evaluate the first file, but include names of duplicates for context.
        primary_file = files[0]
        actual = count_ayah_markers(primary_file)
        if actual != expected:
            issues.append(
                Issue(
                    page=page,
                    expected=expected,
                    actual=actual,
                    xml_files=files,
                    message="Count mismatch",
                )
            )
        elif verbose and len(files) > 1:
            print(
                f"[warn] Multiple XML files detected for page {page}: "
                + ", ".join(str(f.name) for f in files)
            )

    if verbose and unparsed_files:
        print("[warn] Could not infer page numbers for:")
        for xml_path in unparsed_files:
            print(f"   - {xml_path}")

    # Identify XML pages absent from CSV.
    extra_pages = sorted(set(index.keys()) - set(meta.keys()))
    if verbose and extra_pages:
        print(f"[warn] {len(extra_pages)} XML files do not have CSV metadata: {extra_pages}")

    return issues, duplicate_pages, unparsed_files


def print_report(
    issues: Iterable[Issue],
    duplicate_pages: Iterable[int],
    unparsed_files: Iterable[Path],
) -> None:
    issues = list(issues)
    duplicates = list(duplicate_pages)
    unparsed = list(unparsed_files)

    print("=" * 72)
    print("QURAN BOUNDING • AYAH MARKER QUALITY CONTROL")
    print("=" * 72)
    print()

    if duplicates:
        print(f"Warning: duplicate page entries found in CSV: {sorted(duplicates)}")
        print()

    if unparsed:
        print("Warning: could not infer page numbers for the following XML files:")
        for xml_path in unparsed:
            print(f"  - {xml_path}")
        print()

    if not issues:
        print("✓ All ayah_marker counts match ayat_total for the provided dataset.")
        print()
        return

    print(f"Found {len(issues)} issue(s):")
    for issue in issues:
        files = ", ".join(f.name for f in issue.xml_files) if issue.xml_files else "—"
        actual_repr = "—" if issue.actual is None else str(issue.actual)
        print(
            f"  • Page {issue.page:>4}: {issue.message} "
            f"(expected={issue.expected}, actual={actual_repr}, files={files})"
        )
    print()


def main() -> int:
    args = parse_args()
    issues, duplicate_pages, unparsed_files = run_quality_control(
        csv_path=args.csv,
        xml_dir=args.xml_dir,
        verbose=args.verbose,
    )
    print_report(issues, duplicate_pages, unparsed_files)
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())

