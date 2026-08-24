from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


FRET_CLASSES = [f"fret_{value}" for value in range(23)] + ["dead_note"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-isolated full TAB event data from alphaTab renders."
    )
    parser.add_argument("--corpus", type=Path, default=Path("training/data/private-gp"))
    parser.add_argument("--output", type=Path, default=Path("training/data/tab-events"))
    parser.add_argument("--train-copies", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def load_manifest(corpus: Path) -> list[dict]:
    path = corpus / "manifest.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_splits(entries: list[dict]) -> dict[str, str]:
    source_ids = sorted(
        {entry["source_id"] for entry in entries if entry.get("status") == "ok"},
        key=lambda value: hashlib.sha256(value.encode("ascii")).hexdigest(),
    )
    test_count = max(1, round(len(source_ids) * 0.12))
    val_count = max(1, round(len(source_ids) * 0.12))
    result: dict[str, str] = {}
    for index, source_id in enumerate(source_ids):
        if index < test_count:
            result[source_id] = "test"
        elif index < test_count + val_count:
            result[source_id] = "val"
        else:
            result[source_id] = "train"
    return result


def clip_box(box: dict, crop_x: int, crop_y: int, width: int, height: int) -> list[float] | None:
    x1 = max(0.0, float(box["x"]) - crop_x)
    y1 = max(0.0, float(box["y"]) - crop_y)
    x2 = min(float(width), float(box["x"] + box["w"]) - crop_x)
    y2 = min(float(height), float(box["y"] + box["h"]) - crop_y)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]


def augment(image: np.ndarray, copy_index: int, rng: random.Random) -> np.ndarray:
    if copy_index == 0:
        return image
    result = image.astype(np.float32)
    if copy_index % 3 == 1:
        result = 255.0 - result
        tint = np.array(
            [rng.uniform(0.72, 1.0), rng.uniform(0.72, 1.0), rng.uniform(0.72, 1.0)],
            dtype=np.float32,
        )
        result *= tint
    else:
        contrast = rng.uniform(0.62, 1.22)
        brightness = rng.uniform(-32, 24)
        result = (result - 127.5) * contrast + 127.5 + brightness
    if rng.random() < 0.75:
        sigma = rng.uniform(1.5, 8.0)
        result += rng.normalvariate(0, sigma) * np.random.default_rng(
            rng.randrange(2**32)
        ).standard_normal(result.shape)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if rng.random() < 0.55:
        result = cv2.GaussianBlur(result, (3, 3), rng.uniform(0.15, 0.85))
    return result


def duration_label(beat: dict) -> str:
    prefix = "rest" if beat.get("is_rest") else "beat"
    return f"{prefix}_{beat['duration']}_dots_{beat['dots']}"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    entries = load_manifest(args.corpus)
    splits = source_splits(entries)
    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "beat-images" / split).mkdir(parents=True, exist_ok=True)
    (args.output / "annotations").mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": index + 1, "name": name, "supercategory": "tab_token"}
        for index, name in enumerate(FRET_CLASSES)
    ]
    category_ids = {item["name"]: item["id"] for item in categories}
    coco = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "val", "test")
    }
    events = {split: [] for split in ("train", "val", "test")}
    beat_rows = {split: [] for split in ("train", "val", "test")}
    source_counts = Counter()
    annotation_id = 1
    image_id = 1

    for entry in entries:
        if entry.get("status") != "ok":
            continue
        source_id = entry["source_id"]
        split = splits[source_id]
        source_counts[split] += 1
        for render in entry.get("renders", []):
            image_path = args.corpus / render["image"]
            label_path = args.corpus / render["labels"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or not label_path.exists():
                continue
            labels = json.loads(label_path.read_text(encoding="utf-8"))
            copies = args.train_copies if split == "train" else 1
            for system in labels.get("systems", []):
                real = system["real_box"]
                crop_x = max(0, int(np.floor(real["x"] - 8)))
                crop_y = max(0, int(np.floor(real["y"] - 8)))
                crop_x2 = min(image.shape[1], int(np.ceil(real["x"] + real["w"] + 8)))
                crop_y2 = min(image.shape[0], int(np.ceil(real["y"] + real["h"] + 8)))
                base_crop = image[crop_y:crop_y2, crop_x:crop_x2]
                height, width = base_crop.shape[:2]
                if width < 32 or height < 32:
                    continue

                system_events: list[dict] = []
                note_annotations: list[tuple[int, list[float]]] = []
                for master_bar in system.get("master_bars", []):
                    for bar in master_bar.get("bars", []):
                        for beat in bar.get("beats", []):
                            notes = []
                            for note in beat.get("notes", []):
                                if not note.get("is_stringed"):
                                    continue
                                fret = int(note.get("fret", -1))
                                if note.get("is_dead"):
                                    class_name = "dead_note"
                                elif 0 <= fret <= 22:
                                    class_name = f"fret_{fret}"
                                else:
                                    continue
                                bbox = clip_box(note["box"], crop_x, crop_y, width, height)
                                if bbox is None:
                                    continue
                                note_annotations.append((category_ids[class_name], bbox))
                                notes.append(
                                    {
                                        "string": int(note["string"]),
                                        "fret": fret,
                                        "dead": bool(note.get("is_dead")),
                                        "bbox": bbox,
                                    }
                                )
                            event = {
                                "measure": int(master_bar["master_bar_index"]) + 1,
                                "bar_index": int(bar["bar_index"]),
                                "voice": beat.get("voice_index"),
                                "beat_index": int(beat["beat_index"]),
                                "x": round(float(beat["on_notes_x"]) - crop_x, 3),
                                "duration": int(beat["duration"]),
                                "dots": int(beat["dots"]),
                                "rest": bool(beat.get("is_rest")),
                                "full_bar_rest": bool(beat.get("is_full_bar_rest")),
                                "display_start_ticks": beat.get("display_start_ticks"),
                                "display_duration_ticks": beat.get("display_duration_ticks"),
                                "absolute_display_start_ticks": beat.get("absolute_display_start_ticks"),
                                "notes": notes,
                            }
                            system_events.append(event)

                            beat_x = int(round(float(beat["on_notes_x"])))
                            half_width = 42
                            bx1 = max(crop_x, beat_x - half_width)
                            bx2 = min(crop_x2, beat_x + half_width)
                            if bx2 - bx1 >= 24:
                                beat_crop = image[crop_y:crop_y2, bx1:bx2]
                                beat_name = (
                                    f"{source_id}-t{render['track_index']:02d}-"
                                    f"s{system['system_index']:04d}-b{beat['beat_id']}.jpg"
                                )
                                cv2.imwrite(
                                    str(args.output / "beat-images" / split / beat_name),
                                    beat_crop,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                                )
                                beat_rows[split].append(
                                    {
                                        "image": f"beat-images/{split}/{beat_name}",
                                        "source_id": source_id,
                                        "label": duration_label(beat),
                                        "duration": int(beat["duration"]),
                                        "dots": int(beat["dots"]),
                                        "rest": bool(beat.get("is_rest")),
                                    }
                                )

                for copy_index in range(copies):
                    file_name = (
                        f"{source_id}-t{render['track_index']:02d}-"
                        f"s{system['system_index']:04d}-a{copy_index}.jpg"
                    )
                    output_image = args.output / "images" / split / file_name
                    transformed = augment(base_crop, copy_index, rng)
                    cv2.imwrite(str(output_image), transformed, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    coco[split]["images"].append(
                        {"id": image_id, "file_name": file_name, "width": width, "height": height}
                    )
                    for category_id, bbox in note_annotations:
                        coco[split]["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": bbox,
                                "area": round(bbox[2] * bbox[3], 3),
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1
                    events[split].append(
                        {
                            "image": f"images/{split}/{file_name}",
                            "source_id": source_id,
                            "source_path": entry.get("source_path"),
                            "track_index": render["track_index"],
                            "system_index": system["system_index"],
                            "augmentation": copy_index,
                            "crop": {"x": crop_x, "y": crop_y, "width": width, "height": height},
                            "events": system_events,
                        }
                    )
                    image_id += 1

    for split in ("train", "val", "test"):
        (args.output / "annotations" / f"{split}.json").write_text(
            json.dumps(coco[split], ensure_ascii=False), encoding="utf-8"
        )
        (args.output / f"events-{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in events[split]) + "\n",
            encoding="utf-8",
        )
        (args.output / f"beats-{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in beat_rows[split]) + "\n",
            encoding="utf-8",
        )

    summary = {
        "schema_version": 1,
        "split_policy": "source-isolated deterministic 76/12/12 by source hash order",
        "sources": dict(source_counts),
        "images": {split: len(coco[split]["images"]) for split in coco},
        "note_annotations": {split: len(coco[split]["annotations"]) for split in coco},
        "beat_samples": {split: len(beat_rows[split]) for split in beat_rows},
        "duration_classes": {
            split: dict(Counter(row["label"] for row in beat_rows[split])) for split in beat_rows
        },
        "classes": FRET_CLASSES,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
