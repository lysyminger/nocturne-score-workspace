from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collapse fret classes into one position class")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    for split in ("train", "val", "test"):
        source = args.dataset / "annotations" / f"{split}.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["categories"] = [{"id": 1, "name": "fret_token", "supercategory": "tab_token"}]
        for annotation in payload["annotations"]:
            annotation["category_id"] = 1
        target = args.dataset / "annotations" / f"{split}.location.json"
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"{split}: {len(payload['images'])} images / {len(payload['annotations'])} locations")


if __name__ == "__main__":
    main()
