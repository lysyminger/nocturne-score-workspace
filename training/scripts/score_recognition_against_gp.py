from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def dotted_units(duration: int, dots: int) -> float:
    units = 8.0 / duration
    return units * (1.0 if dots == 0 else 1.5 if dots == 1 else 1.75)


def truth_events(payload: dict) -> dict[int, list[dict]]:
    grouped: dict[tuple[int, int], dict] = {}
    for system in payload["systems"]:
        for master_bar in system["master_bars"]:
            measure = int(master_bar["master_bar_index"]) + 1
            for bar in master_bar["bars"]:
                for beat in bar["beats"]:
                    if beat.get("is_rest") or not beat.get("notes"):
                        continue
                    start_ticks = int(beat.get("display_start_ticks") or 0)
                    key = (measure, start_ticks)
                    event = grouped.setdefault(
                        key,
                        {
                            "measure": measure,
                            "onset": start_ticks / 480.0,
                            "duration": dotted_units(int(beat["duration"]), int(beat.get("dots") or 0)),
                            "notes": set(),
                        },
                    )
                    event["duration"] = max(
                        event["duration"],
                        dotted_units(int(beat["duration"]), int(beat.get("dots") or 0)),
                    )
                    for note in beat["notes"]:
                        rendered = "X" if note.get("is_dead") else int(note["fret"])
                        # AlphaTab numbers guitar strings from low to high;
                        # the editor and MusicXML TAB use high to low.
                        event["notes"].add((7 - int(note["string"]), rendered))
    by_measure: dict[int, list[dict]] = defaultdict(list)
    for event in grouped.values():
        event["notes"] = frozenset(event["notes"])
        by_measure[event["measure"]].append(event)
    for events in by_measure.values():
        events.sort(key=lambda event: event["onset"])
    return by_measure


def predicted_events(payload: dict) -> dict[int, list[dict]]:
    by_measure: dict[int, list[dict]] = defaultdict(list)
    for measure in payload["measures"]:
        for event in measure.get("ocr_events") or measure.get("events") or []:
            notes = frozenset(
                (
                    int(note["string"]),
                    "X" if note.get("technique") == "dead_note" else int(note["fret"]),
                )
                for note in event.get("notes") or []
            )
            if notes:
                by_measure[int(measure["number"])].append(
                    {
                        "onset": float(event["onset_eighths"]),
                        "duration": float(event["duration_eighths"]),
                        "notes": notes,
                    }
                )
    return by_measure


def main() -> None:
    parser = argparse.ArgumentParser(description="Score recognized TAB events against paired GP truth")
    parser.add_argument("--gp-labels", type=Path, required=True)
    parser.add_argument("--recognition", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--onset-tolerance", type=float, default=0.26)
    args = parser.parse_args()

    truth = truth_events(json.loads(args.gp_labels.read_text(encoding="utf-8")))
    predicted = predicted_events(json.loads(args.recognition.read_text(encoding="utf-8")))
    location_tp = full_tp = rhythm_matches = 0
    truth_count = sum(len(events) for events in truth.values())
    predicted_count = sum(len(events) for events in predicted.values())
    for measure_number in sorted(set(truth) | set(predicted)):
        targets = truth.get(measure_number, [])
        used: set[int] = set()
        for event in predicted.get(measure_number, []):
            candidates = [
                (abs(event["onset"] - target["onset"]), index)
                for index, target in enumerate(targets)
                if index not in used
                and event["notes"] == target["notes"]
                and abs(event["onset"] - target["onset"]) <= args.onset_tolerance
            ]
            if not candidates:
                continue
            _, index = min(candidates)
            used.add(index)
            location_tp += 1
            if abs(event["duration"] - targets[index]["duration"]) <= 0.01:
                rhythm_matches += 1
                full_tp += 1
    location_precision = location_tp / max(1, predicted_count)
    location_recall = location_tp / max(1, truth_count)
    full_precision = full_tp / max(1, predicted_count)
    full_recall = full_tp / max(1, truth_count)
    result = {
        "truth_note_events": truth_count,
        "predicted_note_events": predicted_count,
        "location_chord_event_precision": location_precision,
        "location_chord_event_recall": location_recall,
        "location_chord_event_f1": 2 * location_precision * location_recall / max(1e-12, location_precision + location_recall),
        "duration_accuracy_on_matched_events": rhythm_matches / max(1, location_tp),
        "full_event_precision": full_precision,
        "full_event_recall": full_recall,
        "full_event_f1": 2 * full_precision * full_recall / max(1e-12, full_precision + full_recall),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
