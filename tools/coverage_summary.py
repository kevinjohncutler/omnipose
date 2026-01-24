#!/usr/bin/env python3
"""Summarize coverage.xml with lowest-coverage files first."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize coverage.xml")
    parser.add_argument(
        "--xml",
        default="coverage.xml",
        help="Path to coverage XML (default: coverage.xml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of files to show (default: 25)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise SystemExit(f"missing coverage xml: {xml_path}")

    root = ET.parse(xml_path).getroot()
    rows = []
    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        if not filename:
            continue
        line_nodes = cls.findall("./lines/line")
        if not line_nodes:
            continue
        lines_valid = len(line_nodes)
        lines_covered = sum(1 for ln in line_nodes if int(ln.attrib.get("hits", "0")) > 0)
        pct = 100.0 * lines_covered / lines_valid
        rows.append((pct, lines_valid - lines_covered, lines_valid, filename))

    rows.sort(key=lambda r: (r[0], -r[1], r[3]))
    print("coverage.xml summary (lowest coverage first):")
    for pct, miss, total, filename in rows[: args.limit]:
        print(f"{pct:6.1f}%  miss {miss:4d}/{total:<4d}  {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
