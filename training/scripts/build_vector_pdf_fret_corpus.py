from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pymupdf


CLASSES = [f"fret_{value}" for value in range(23)] + ["dead_note"]
TOKEN_PATTERN = re.compile(r"^[<(\[]?(X|x|\d{1,2})[>)\]]?$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract exact fret boxes from vector Guitar Pro PDFs")
    parser.add_argument("--source", type=Path, default=Path(r"D:\document\谱子"))
    parser.add_argument("--output", type=Path, default=Path("training/data/pdf-frets"))
    parser.add_argument("--scale", type=float, default=2.0)
    return parser.parse_args()


def union_length(segments: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted((min(a, b), max(a, b)) for a, b in segments):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def horizontal_rows(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    by_y: dict[float, list[tuple[float, float]]] = {}
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            start, end = item[1], item[2]
            if abs(start.y - end.y) > 0.3 or abs(start.x - end.x) < 2:
                continue
            key = round((start.y + end.y) * 2) / 4
            by_y.setdefault(key, []).append((start.x, end.x))
    rows = []
    for y, segments in by_y.items():
        coverage = union_length(segments)
        if coverage < 70:
            continue
        rows.append((y, min(min(a, b) for a, b in segments), max(max(a, b) for a, b in segments)))
    return sorted(rows)


def staff_groups(rows: list[tuple[float, float, float]]) -> list[dict]:
    groups = []
    used: set[int] = set()
    for index in range(len(rows) - 5):
        if any(position in used for position in range(index, index + 6)):
            continue
        window = rows[index:index + 6]
        gaps = [window[position + 1][0] - window[position][0] for position in range(5)]
        gap = float(np.median(gaps))
        if not 4 <= gap <= 12 or max(abs(value - gap) for value in gaps) > 0.8:
            continue
        left = min(row[1] for row in window)
        right = max(row[2] for row in window)
        if right - left < 90:
            continue
        groups.append({"ys": [row[0] for row in window], "left": left, "right": right, "gap": gap})
        used.update(range(index, index + 6))
    return groups


def token_class(text: str) -> str | None:
    compact = text.strip().replace(" ", "")
    match = TOKEN_PATTERN.match(compact)
    if not match:
        return None
    value = match.group(1)
    if value.lower() == "x":
        return "dead_note"
    fret = int(value)
    return f"fret_{fret}" if 0 <= fret <= 22 else None


def split_for(path: Path) -> str:
    name = path.name.lower()
    if "藍二乗" in name or "滑滑蛋" in name:
        return "test"
    digest = int(hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8], 16) % 10
    return "val" if digest == 0 else "train"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
    (args.output / "annotations").mkdir(parents=True, exist_ok=True)
    categories = [
        {"id": index + 1, "name": name, "supercategory": "tab_token"}
        for index, name in enumerate(CLASSES)
    ]
    category_ids = {item["name"]: item["id"] for item in categories}
    payloads = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "val", "test")
    }
    source_counts = Counter()
    image_id = annotation_id = 1

    for pdf_path in sorted(args.source.rglob("*.pdf")):
        split = split_for(pdf_path)
        source_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
        document = pymupdf.open(pdf_path)
        source_images = 0
        for page_index, page in enumerate(document):
            words = page.get_text("words")
            groups = staff_groups(horizontal_rows(page))
            if not groups:
                continue
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(args.scale, args.scale), alpha=False)
            page_image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)
            for group_index, group in enumerate(groups):
                top = max(0.0, group["ys"][0] - group["gap"] * 2.6)
                bottom = min(page.rect.height, group["ys"][-1] + group["gap"] * 4.2)
                left = max(0.0, group["left"] - 8)
                right = min(page.rect.width, group["right"] + 8)
                crop = page_image[
                    int(top * args.scale):int(np.ceil(bottom * args.scale)),
                    int(left * args.scale):int(np.ceil(right * args.scale)),
                ]
                if crop.size == 0:
                    continue
                annotations = []
                for word in words:
                    x1, y1, x2, y2, text = word[:5]
                    class_name = token_class(text)
                    if class_name is None or x2 < left or x1 > right:
                        continue
                    center_y = (y1 + y2) / 2
                    if min(abs(center_y - row_y) for row_y in group["ys"]) > max(1.8, group["gap"] * 0.30):
                        continue
                    string = min(range(6), key=lambda index: abs(center_y - group["ys"][index])) + 1
                    bbox = [
                        (x1 - left) * args.scale,
                        (y1 - top) * args.scale,
                        max(2.0, (x2 - x1) * args.scale),
                        max(2.0, (y2 - y1) * args.scale),
                    ]
                    annotations.append((category_ids[class_name], string, bbox))
                if not annotations:
                    continue
                file_name = f"{source_id}-p{page_index:03d}-s{group_index:02d}.jpg"
                cv2.imwrite(str(args.output / "images" / split / file_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                height, width = crop.shape[:2]
                payloads[split]["images"].append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                        "staff_lines": [round((value - top) * args.scale, 3) for value in group["ys"]],
                    }
                )
                for category_id, string, bbox in annotations:
                    payloads[split]["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "string": string,
                            "bbox": [round(value, 3) for value in bbox],
                            "area": round(bbox[2] * bbox[3], 3),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
                image_id += 1
                source_images += 1
        if source_images:
            source_counts[split] += 1
            print(f"{split}: {pdf_path.name} -> {source_images} systems")

    for split, payload in payloads.items():
        (args.output / "annotations" / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    summary = {
        "schema_version": 1,
        "label_source": "embedded vector text boxes intersecting detected six-line TAB staves",
        "test_policy": "paired named GP/PDF sources held out: 藍二乗 and 滑滑蛋",
        "sources": dict(source_counts),
        "images": {split: len(payload["images"]) for split, payload in payloads.items()},
        "annotations": {split: len(payload["annotations"]) for split, payload in payloads.items()},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
