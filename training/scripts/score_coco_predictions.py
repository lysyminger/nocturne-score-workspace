from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score exact-class token detections with a fixed IoU threshold")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    truth_payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    truth: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for annotation in truth_payload["annotations"]:
        truth[(int(annotation["image_id"]), int(annotation["category_id"]))].append(annotation)
    selected = [row for row in predictions if float(row["score"]) >= args.score_threshold]
    selected.sort(key=lambda row: float(row["score"]), reverse=True)
    matched: dict[tuple[int, int], set[int]] = defaultdict(set)
    true_positive = false_positive = 0
    for prediction in selected:
        key = (int(prediction["image_id"]), int(prediction["category_id"]))
        candidates = truth.get(key, [])
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(candidates):
            if index in matched[key]:
                continue
            overlap = iou(prediction["bbox"], target["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= args.iou_threshold:
            matched[key].add(best_index)
            true_positive += 1
        else:
            false_positive += 1
    false_negative = len(truth_payload["annotations"]) - true_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    result = {
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ground_truth_tokens": len(truth_payload["annotations"]),
    }
    image_info = {int(row["id"]): row for row in truth_payload["images"]}
    if image_info and all("staff_lines" in row for row in image_info.values()) and all(
        "string" in row for row in truth_payload["annotations"]
    ):
        def event_groups(rows: list[dict], staff_lines: list[float], predicted: bool) -> list[tuple[float, frozenset[tuple[int, int]]]]:
            gap = sum(b - a for a, b in zip(staff_lines, staff_lines[1:])) / 5
            positioned = []
            for row in rows:
                x, y, width, height = row["bbox"]
                center_x, center_y = x + width / 2, y + height / 2
                if predicted:
                    nearest = min(range(6), key=lambda index: abs(center_y - staff_lines[index]))
                    if abs(center_y - staff_lines[nearest]) > gap * 0.55:
                        continue
                    string = nearest + 1
                else:
                    string = int(row["string"])
                positioned.append((center_x, string, int(row["category_id"])))
            positioned.sort()
            groups: list[list[tuple[float, int, int]]] = []
            for item in positioned:
                if groups and item[0] - sum(value[0] for value in groups[-1]) / len(groups[-1]) <= gap * 0.48:
                    groups[-1].append(item)
                else:
                    groups.append([item])
            return [
                (
                    sum(item[0] for item in group) / len(group),
                    frozenset((item[1], item[2]) for item in group),
                )
                for group in groups
            ]

        truth_by_image: dict[int, list[dict]] = defaultdict(list)
        predictions_by_image: dict[int, list[dict]] = defaultdict(list)
        for row in truth_payload["annotations"]:
            truth_by_image[int(row["image_id"])].append(row)
        for row in selected:
            predictions_by_image[int(row["image_id"])].append(row)
        event_tp = event_fp = event_fn = 0
        for image_id, info in image_info.items():
            lines = [float(value) for value in info["staff_lines"]]
            gap = sum(b - a for a, b in zip(lines, lines[1:])) / 5
            true_events = event_groups(truth_by_image[image_id], lines, False)
            predicted_events = event_groups(predictions_by_image[image_id], lines, True)
            used: set[int] = set()
            for predicted_x, predicted_notes in predicted_events:
                candidates = [
                    (abs(predicted_x - true_x), index)
                    for index, (true_x, true_notes) in enumerate(true_events)
                    if index not in used and predicted_notes == true_notes and abs(predicted_x - true_x) <= gap
                ]
                if candidates:
                    used.add(min(candidates)[1])
                    event_tp += 1
                else:
                    event_fp += 1
            event_fn += len(true_events) - len(used)
        event_precision = event_tp / max(1, event_tp + event_fp)
        event_recall = event_tp / max(1, event_tp + event_fn)
        result.update(
            {
                "event_true_positive": event_tp,
                "event_false_positive": event_fp,
                "event_false_negative": event_fn,
                "event_precision": event_precision,
                "event_recall": event_recall,
                "event_f1": 2 * event_precision * event_recall / max(1e-12, event_precision + event_recall),
            }
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
