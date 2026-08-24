from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 alphaTab 坐标真值生成品位数字 OCR 小图。")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("training/data/private-gp"),
        help="build_gp_corpus.mjs 的输出目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/fret-ocr"),
        help="OCR 数据集输出目录",
    )
    parser.add_argument("--padding", type=int, default=3, help="裁切框四周扩展像素")
    parser.add_argument("--qa-limit", type=int, default=200, help="每张 QA 图最多画多少个框")
    return parser.parse_args()


def split_for_source(source_id: str) -> str:
    bucket = int(source_id[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def iter_notes(label_data: dict):
    for system in label_data.get("systems", []):
        for master_bar in system.get("master_bars", []):
            for bar in master_bar.get("bars", []):
                for beat in bar.get("beats", []):
                    for note in beat.get("notes", []):
                        yield master_bar["master_bar_index"], beat, note


def referenced_label_files(corpus: Path) -> list[Path]:
    manifest_path = corpus / "manifest.jsonl"
    if not manifest_path.is_file():
        return sorted(corpus.glob("*/*.labels.json"))
    label_files: list[Path] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") != "ok":
            continue
        for render in entry.get("renders", []):
            label_files.append(corpus / Path(render["labels"]))
    return label_files


def is_plain_fret(note: dict) -> bool:
    return (
        note.get("is_stringed")
        and isinstance(note.get("fret"), int)
        and note["fret"] >= 0
        and not note.get("is_dead")
        and not note.get("is_ghost")
        and not note.get("is_tie_destination")
        and note.get("harmonic_type", 0) == 0
    )


def clamp_crop(box: dict, width: int, height: int, padding: int) -> tuple[int, int, int, int] | None:
    left = max(0, int(box["x"] // 1) - padding)
    top = max(0, int(box["y"] // 1) - padding)
    right = min(width, int(-(-float(box["x"] + box["w"]) // 1)) + padding)
    bottom = min(height, int(-(-float(box["y"] + box["h"]) // 1)) + padding)
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    qa_dir = args.output / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    label_rows: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    source_counts: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    crop_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    label_files = referenced_label_files(args.corpus)
    for label_path in label_files:
        data = json.loads(label_path.read_text(encoding="utf-8"))
        if data.get("profile") != "tab":
            continue
        source_id = data["source_id"]
        split = split_for_source(source_id)
        image_path = label_path.with_name(data["image"]["file"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取渲染图: {image_path}")
        height, width = image.shape[:2]
        split_dir = args.output / "images" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        qa_image = image.copy()
        qa_drawn = 0
        accepted = 0

        for master_bar_index, beat, note in iter_notes(data):
            if not is_plain_fret(note):
                continue
            crop_box = clamp_crop(note["box"], width, height, args.padding)
            if crop_box is None:
                continue
            left, top, right, bottom = crop_box
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue
            name = (
                f"{source_id}-t{data['track_index']:02d}-m{master_bar_index:04d}"
                f"-b{beat['beat_id']}-n{note['note_id']}.png"
            )
            crop_path = split_dir / name
            if not cv2.imwrite(str(crop_path), crop):
                raise RuntimeError(f"无法写入裁图: {crop_path}")
            relative = crop_path.relative_to(args.output).as_posix()
            label_rows[split].append(f"{relative}\t{note['fret']}")
            crop_counts[split] += 1
            accepted += 1

            if qa_drawn < args.qa_limit:
                cv2.rectangle(qa_image, (left, top), (right, bottom), (168, 79, 255), 1)
                qa_drawn += 1

        if accepted:
            source_counts[split].add(source_id)
            qa_path = qa_dir / f"{source_id}-t{data['track_index']:02d}.jpg"
            cv2.imwrite(str(qa_path), qa_image, [cv2.IMWRITE_JPEG_QUALITY, 92])

    for split, rows in label_rows.items():
        rows.sort()
        (args.output / f"{split}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    summary = {
        "schema_version": 1,
        "split_policy": "sha256(source) first 8 hex modulo 100: train 0-79, val 80-89, test 90-99",
        "filters": ["stringed", "not dead", "not ghost", "not tie destination", "not harmonic"],
        "sources": {split: len(values) for split, values in source_counts.items()},
        "crops": crop_counts,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
