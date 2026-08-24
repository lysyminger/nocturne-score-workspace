from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the music-symbol COCO dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def validate_split(dataset: Path, split: str, expected_classes: list[str]) -> tuple[int, int]:
    annotation_path = dataset / "annotations" / f"{split}.json"
    image_root = dataset / "images" / split
    if not annotation_path.is_file():
        raise ValueError(f"Missing annotation file: {annotation_path}")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(payload.get("categories", []), key=lambda item: int(item["id"]))
    category_names = [str(item["name"]) for item in categories]
    if category_names != expected_classes:
        raise ValueError(f"{split}.json categories do not match symbol_classes.yaml")
    category_ids = {int(item["id"]) for item in categories}
    images = {int(item["id"]): item for item in payload.get("images", [])}
    for image_id, image_info in images.items():
        image_path = image_root / str(image_info["file_name"])
        if not image_path.is_file():
            raise ValueError(f"Missing image: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        if width < 8 or height < 8:
            raise ValueError(f"Invalid image size: {image_path}")
    valid_annotations = 0
    for annotation in payload.get("annotations", []):
        if int(annotation["image_id"]) not in images:
            raise ValueError(f"Annotation references an unknown image: {annotation['id']}")
        if int(annotation["category_id"]) not in category_ids:
            raise ValueError(f"Annotation references an unknown category: {annotation['id']}")
        _x, _y, width, height = (float(value) for value in annotation["bbox"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Annotation has an invalid bbox: {annotation['id']}")
        valid_annotations += 1
    if not images or not valid_annotations:
        raise ValueError(f"{split} split must contain images and annotations")
    return len(images), valid_annotations


def main() -> None:
    args = parse_args()
    training_root = Path(__file__).resolve().parents[1]
    classes = yaml.safe_load((training_root / "configs" / "symbol_classes.yaml").read_text(encoding="utf-8"))["classes"]
    train_count = validate_split(args.dataset.resolve(), "train", classes)
    val_count = validate_split(args.dataset.resolve(), "val", classes)
    print(f"COCO dataset valid: train={train_count[0]} images/{train_count[1]} boxes, val={val_count[0]} images/{val_count[1]} boxes")


if __name__ == "__main__":
    main()
