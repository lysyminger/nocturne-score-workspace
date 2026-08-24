from __future__ import annotations

import base64
import csv
import io
import json
import math
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import httpx
import numpy as np

from backend.score_pdf import build_recognized_score_pdf


STANDARD_TUNING_MIDI = (64, 59, 55, 50, 45, 40)
EIGHTH_UNITS_PER_MEASURE = 8
TAB_TECHNIQUES = frozenset(
    {
        "legato",
        "slide",
        "hammer_on",
        "pull_off",
        "bend",
        "vibrato",
        "harmonic",
        "palm_mute",
        "let_ring",
        "dead_note",
    }
)


@dataclass
class FrameInput:
    path: Path
    time_seconds: float
    source_frame: int = 0


@dataclass
class DigitGlyph:
    image: np.ndarray
    feature: np.ndarray
    label: str | None = None
    confidence: float = 0.0
    raw_label: str | None = None
    raw_confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None


@dataclass
class NoteToken:
    x: float
    string: int
    glyphs: list[DigitGlyph]
    duration_units: float | None = None
    raw_text: str | None = None
    raw_confidence: float = 0.0

    @property
    def text(self) -> str | None:
        labels = [glyph.label for glyph in self.glyphs]
        return "".join(labels) if labels and all(labels) else None

    @property
    def confidence(self) -> float:
        return min((glyph.confidence for glyph in self.glyphs), default=0.0)


@dataclass
class MeasureGeometry:
    left: float
    right: float
    label_image: np.ndarray | None
    raw_label: str | None = None
    raw_label_confidence: float = 0.0
    label_alternates: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class TechniqueMark:
    x: float
    technique: str
    raw_text: str
    confidence: float
    end_x: float | None = None


@dataclass
class ParsedFrame:
    source: FrameInput
    gray: np.ndarray
    staff_lines: list[int]
    measures: list[MeasureGeometry]
    tokens: list[NoteToken]
    highlighted_index: int | None
    start_measure: int | None = None
    start_measure_confidence: float = 0.0
    layout: str = "tab_only"
    polarity: str = "dark_on_light"
    notation_staff_lines: list[int] | None = None
    technique_marks: list[TechniqueMark] = field(default_factory=list)
    tempo_candidates: list[tuple[float, float]] = field(default_factory=list)
    ai_recognized_tokens: int = 0
    ai_error: str | None = None


@dataclass(frozen=True)
class TabNote:
    string: int
    fret: int
    technique: str | None = None


@dataclass(frozen=True)
class TabEvent:
    # Values are measured in eighth-note units. Half a unit is a sixteenth note.
    onset: float
    duration: float
    notes: tuple[TabNote, ...]


@dataclass
class MeasureCandidate:
    number: int
    events: tuple[TabEvent, ...]
    quality: float
    source_time: float
    signature: tuple[tuple[float, int, int], ...]
    source_path: Path | None = None
    crop_box: tuple[int, int, int, int] | None = None
    time_signature: tuple[int, int] = (4, 4)


@dataclass
class TabRecognitionResult:
    score_path: Path
    diagnostics_path: Path
    summary: dict
    sync_suggestions: list[dict] = field(default_factory=list)
    pdf_path: Path | None = None


def _group_runs(indices: np.ndarray) -> list[list[int]]:
    if indices.size == 0:
        return []
    groups = [[int(indices[0])]]
    for value in indices[1:]:
        integer = int(value)
        if integer == groups[-1][-1] + 1:
            groups[-1].append(integer)
        else:
            groups.append([integer])
    return groups


@dataclass(frozen=True)
class ScoreLayout:
    gray: np.ndarray
    staff_lines: list[int]
    notation_staff_lines: list[int] | None
    layout: str
    polarity: str


def _local_foreground(gray: np.ndarray, polarity: str) -> tuple[np.ndarray, np.ndarray]:
    source = gray.astype(np.float32)
    background = cv2.GaussianBlur(source, (0, 0), sigmaX=3, sigmaY=3)
    signal = source - background if polarity == "light_on_dark" else background - source
    signal = np.maximum(signal, 0)
    foreground = signal >= 4
    return signal, foreground


def _line_centers(signal: np.ndarray, foreground: np.ndarray) -> tuple[list[int], np.ndarray]:
    width = foreground.shape[1]
    coverage = np.count_nonzero(foreground, axis=1) / max(width, 1)
    strength = np.mean(np.minimum(signal, 48), axis=1) + coverage * 12
    groups = _group_runs(np.flatnonzero(coverage >= 0.45))
    centers = [max(group, key=lambda y: float(strength[y])) for group in groups]
    return centers, strength


def _best_line_group(
    centers: list[int], strength: np.ndarray, count: int, height: int
) -> tuple[float, list[int]] | None:
    best: tuple[float, list[int]] | None = None
    for first_index, first in enumerate(centers):
        for second in centers[first_index + 1 :]:
            gap = second - first
            if gap < 5 or gap > height / 4:
                continue
            candidate = [first, second]
            last_index = centers.index(second)
            for offset in range(2, count):
                expected = first + gap * offset
                options = [
                    value
                    for value in centers[last_index + 1 :]
                    if abs(value - expected) <= max(2.0, gap * 0.28)
                ]
                if not options:
                    break
                value = min(options, key=lambda item: abs(item - expected))
                candidate.append(value)
                last_index = centers.index(value)
            if len(candidate) != count:
                continue
            gaps = np.diff(candidate)
            median_gap = float(np.median(gaps))
            irregularity = float(np.mean(np.abs(gaps - median_gap)) / median_gap)
            if irregularity > 0.22:
                continue
            average_strength = float(statistics.mean(float(strength[y]) for y in candidate))
            score = average_strength - irregularity * 80
            if best is None or score > best[0]:
                best = (score, candidate)
    return best


def detect_score_layout(gray: np.ndarray) -> ScoreLayout:
    height, _width = gray.shape
    choices: list[tuple[float, str, list[int], np.ndarray, np.ndarray, list[int]]] = []
    for polarity in ("dark_on_light", "light_on_dark"):
        signal, foreground = _local_foreground(gray, polarity)
        centers, strength = _line_centers(signal, foreground)
        group = _best_line_group(centers, strength, 6, height)
        if group:
            choices.append((group[0], polarity, group[1], signal, foreground, centers))
    if not choices:
        raise ValueError("没有检测到完整的六线 TAB，请重新框选六根弦线")

    background_level = float(np.median(gray))
    preferred_polarity = "dark_on_light" if background_level >= 160 else "light_on_dark" if background_level <= 96 else None
    preferred = [item for item in choices if item[1] == preferred_polarity]
    _score, polarity, staff_lines, signal, foreground, centers = max(
        preferred or choices, key=lambda item: item[0]
    )
    if preferred_polarity and polarity != preferred_polarity:
        polarity = preferred_polarity
        signal, foreground = _local_foreground(gray, polarity)
        centers, _strength = _line_centers(signal, foreground)
    spacing = float(statistics.median(np.diff(staff_lines)))
    notation_candidates = [value for value in centers if value < staff_lines[0] - spacing * 1.5]
    notation_group = _best_line_group(notation_candidates, np.mean(np.minimum(signal, 48), axis=1) + np.mean(foreground, axis=1) * 12, 5, height)
    notation_lines = notation_group[1] if notation_group else None
    if notation_lines:
        notation_spacing = float(statistics.median(np.diff(notation_lines)))
        separation = staff_lines[0] - notation_lines[-1]
        if not (spacing * 0.5 <= notation_spacing <= spacing * 1.6 and separation <= spacing * 12):
            notation_lines = None

    canonical = (
        np.where(foreground, 0, 255).astype(np.uint8)
        if polarity == "light_on_dark"
        else gray.copy()
    )
    return ScoreLayout(
        gray=canonical,
        staff_lines=staff_lines,
        notation_staff_lines=notation_lines,
        layout="staff_tab_pair" if notation_lines else "tab_only",
        polarity=polarity,
    )


def detect_staff_lines(gray: np.ndarray) -> list[int]:
    return detect_score_layout(gray).staff_lines


def _merge_nearby(values: list[int], distance: float) -> list[float]:
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= distance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [statistics.mean(group) for group in groups]


def detect_measure_boundaries(gray: np.ndarray, staff_lines: list[int]) -> list[float]:
    spacing = statistics.median(np.diff(staff_lines))
    top, bottom = staff_lines[0], staff_lines[-1]
    band = gray[top : bottom + 1] < 215
    column_counts = np.count_nonzero(band, axis=0)
    longest_runs = np.zeros(band.shape[1], dtype=np.int32)
    for x in range(band.shape[1]):
        runs = _group_runs(np.flatnonzero(band[:, x]))
        longest_runs[x] = max((len(run) for run in runs), default=0)
    below_end = min(gray.shape[0], round(bottom + spacing * 2.2) + 1)
    below = gray[bottom + 1 : below_end] < 215
    below_counts = np.count_nonzero(below, axis=0) if below.size else np.zeros(gray.shape[1])
    # A note stem can cross most or all TAB lines, but it continues well below
    # the sixth line. A barline stops at the staff edge. This distinction is
    # essential for dense printed PDF tablature where every note has a stem.
    candidates = (column_counts >= band.shape[0] * 0.86) & (
        longest_runs >= band.shape[0] * 0.9
    ) & (
        below_counts <= max(2, spacing * 0.35)
    )
    groups = _group_runs(np.flatnonzero(candidates))
    if len(groups) < 3:
        relaxed = (column_counts >= band.shape[0] * 0.86) & (
            below_counts <= max(2, spacing * 0.35)
        )
        groups = _group_runs(np.flatnonzero(relaxed))
    centers = [round(statistics.mean(group)) for group in groups]
    centers = _merge_nearby(centers, max(5, spacing * 0.75))
    if len(centers) < 3:
        raise ValueError("没有检测到连续小节线，请扩大左右框选范围")

    large_gaps = [gap for gap in np.diff(centers) if gap >= spacing * 4]
    if not large_gaps:
        raise ValueError("小节线间距无效")
    expected = statistics.median(large_gaps)

    filtered = [centers[0]]
    for center in centers[1:]:
        gap = center - filtered[-1]
        if gap < expected * 0.3:
            filtered[-1] = statistics.mean((filtered[-1], center))
        else:
            filtered.append(center)
    return filtered


def _glyph_feature(image: np.ndarray) -> np.ndarray:
    points = cv2.findNonZero(image)
    if points is None:
        return np.zeros((28, 20), dtype=np.uint8)
    x, y, width, height = cv2.boundingRect(points)
    crop = image[y : y + height, x : x + width]
    scale = min(24 / max(height, 1), 16 / max(width, 1))
    resized = cv2.resize(
        crop,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    resized = (resized >= 96).astype(np.uint8)
    canvas = np.zeros((28, 20), dtype=np.uint8)
    top = (canvas.shape[0] - resized.shape[0]) // 2
    left = (canvas.shape[1] - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def extract_note_tokens(
    gray: np.ndarray, staff_lines: list[int]
) -> list[NoteToken]:
    spacing = float(statistics.median(np.diff(staff_lines)))
    crop_top = max(0, round(staff_lines[0] - spacing * 0.7))
    crop_bottom = min(gray.shape[0], round(staff_lines[-1] + spacing * 0.7) + 1)
    symbols = ((gray[crop_top:crop_bottom] < 225) * 255).astype(np.uint8)
    horizontal = cv2.morphologyEx(
        symbols,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(8, round(spacing * 2.2)), 1),
        ),
    )
    symbols = cv2.subtract(symbols, horizontal)
    symbols = cv2.morphologyEx(
        symbols,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(symbols)

    by_string: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for index in range(1, count):
        x, local_y, width, height, area = map(int, stats[index])
        center_y = crop_top + local_y + height / 2
        string_index = min(range(6), key=lambda value: abs(staff_lines[value] - center_y))
        if abs(staff_lines[string_index] - center_y) > spacing * 0.5:
            continue
        if not (
            spacing * 0.1 <= width <= spacing * 0.95
            and spacing * 0.4 <= height <= spacing * 1.25
            and area >= spacing * spacing * 0.04
        ):
            continue
        by_string[string_index + 1].append((x, crop_top + local_y, width, height, area))

    tokens: list[NoteToken] = []
    for string, components in by_string.items():
        components.sort(key=lambda item: item[0])
        groups: list[list[tuple[int, int, int, int, int]]] = []
        for component in components:
            if groups:
                previous = groups[-1][-1]
                gap = component[0] - (previous[0] + previous[2])
                if gap <= spacing * 0.3:
                    groups[-1].append(component)
                    continue
            groups.append([component])

        for group in groups:
            glyphs: list[DigitGlyph] = []
            for x, y, width, height, _area in group:
                glyph_image = symbols[
                    max(0, y - crop_top) : max(0, y - crop_top) + height,
                    x : x + width,
                ]
                glyphs.append(
                    DigitGlyph(
                        glyph_image,
                        _glyph_feature(glyph_image),
                        box=(x, y, width, height),
                    )
                )
            left = min(item[0] for item in group)
            right = max(item[0] + item[2] for item in group)
            tokens.append(NoteToken(x=(left + right) / 2, string=string, glyphs=glyphs))
    return sorted(tokens, key=lambda token: (token.x, token.string))


def assign_rhythm_units(
    gray: np.ndarray, staff_lines: list[int], tokens: list[NoteToken]
) -> None:
    """Read the compact rhythm stems rendered below this video's TAB staff.

    A standalone stem is a quarter note, a nearby augmentation dot makes it a
    dotted quarter, one connecting beam is an eighth note, and two separated
    beam bands are a sixteenth note.
    """
    spacing = float(statistics.median(np.diff(staff_lines)))
    top = min(gray.shape[0], round(staff_lines[-1] + spacing * 0.2))
    bottom = min(gray.shape[0], round(staff_lines[-1] + spacing * 3.4))
    if bottom - top < spacing:
        return
    rhythm = ((gray[top:bottom] < 210) * 255).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(rhythm)
    components = [
        tuple(map(int, stats[index]))
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= spacing * 1.2
    ]
    small_components = [
        tuple(map(int, stats[index]))
        for index in range(1, count)
        if 3 <= stats[index, cv2.CC_STAT_AREA] <= spacing * spacing * 0.2
        and stats[index, cv2.CC_STAT_WIDTH] <= spacing * 0.6
        and stats[index, cv2.CC_STAT_HEIGHT] <= spacing * 0.6
    ]

    for token in tokens:
        stem_component: tuple[int, int, int, int, int] | None = None
        for component in components:
            x, y, width, height, _area = component
            if height < spacing * 1.25 or y > spacing * 0.9:
                continue
            if x - spacing * 0.25 <= token.x <= x + width + spacing * 0.25:
                if stem_component is None or height > stem_component[3]:
                    stem_component = component
        if stem_component is None:
            continue
        x, y, width, height, _area = stem_component
        if width >= spacing * 1.2:
            component_pixels = rhythm[y : y + height, x : x + width]
            beam_rows = np.flatnonzero(
                np.count_nonzero(component_pixels, axis=1) >= spacing * 1.2
            )
            beam_bands = _group_runs(beam_rows)
            token.duration_units = 0.5 if len(beam_bands) >= 2 else 1
            continue
        stem_right = x + width
        stem_bottom = y + height
        dotted = any(
            stem_right <= dot_x <= stem_right + spacing * 0.65
            and stem_bottom - spacing * 0.55 <= dot_y + dot_height / 2 <= stem_bottom + spacing * 0.35
            for dot_x, dot_y, _dot_width, dot_height, _dot_area in small_components
        )
        token.duration_units = 3 if dotted else 2


def _longest_dark_run(row: np.ndarray) -> tuple[int, int]:
    indices = np.flatnonzero(row < 90)
    groups = _group_runs(indices)
    if not groups:
        return 0, 0
    longest = max(groups, key=len)
    return longest[0], longest[-1] + 1


def detect_highlighted_measure(
    gray: np.ndarray, staff_lines: list[int], boundaries: list[float]
) -> int | None:
    if len(boundaries) < 2:
        return None
    measure_width = statistics.median(np.diff(boundaries))
    runs: list[tuple[int, int, int]] = []
    for y, row in enumerate(gray):
        left, right = _longest_dark_run(row)
        if measure_width * 0.65 <= right - left <= measure_width * 1.15:
            runs.append((y, left, right))
    if not runs:
        return None

    grouped: list[list[tuple[int, int, int]]] = [[runs[0]]]
    for run in runs[1:]:
        if run[0] == grouped[-1][-1][0] + 1:
            grouped[-1].append(run)
        else:
            grouped.append([run])
    representatives = [
        (
            round(statistics.mean(item[0] for item in group)),
            round(statistics.median(item[1] for item in group)),
            round(statistics.median(item[2] for item in group)),
        )
        for group in grouped
    ]
    top_candidates = [item for item in representatives if item[0] < staff_lines[0]]
    bottom_candidates = [item for item in representatives if item[0] > staff_lines[-1]]
    for top in top_candidates:
        for bottom in bottom_candidates:
            if (
                abs(top[1] - bottom[1]) <= measure_width * 0.08
                and abs(top[2] - bottom[2]) <= measure_width * 0.08
            ):
                center = statistics.mean((top[1], top[2], bottom[1], bottom[2]))
                for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
                    if left <= center <= right:
                        return index
    return None


def _clean_measure_labels(
    color: np.ndarray,
    canonical_gray: np.ndarray,
    left: float,
    staff_top: int,
    spacing: float,
    polarity: str,
) -> list[np.ndarray]:
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    x0 = max(0, round(left - spacing * 1.2))
    x1 = min(gray.shape[1], round(left + spacing * 2.8))
    y0 = max(0, round(staff_top - spacing * 1.35))
    y1 = max(y0 + 1, round(staff_top - spacing * 0.05))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    color_crop = color[y0:y1, x0:x1]
    blue, green, red = cv2.split(color_crop)
    red_pixels = (
        (red > 45)
        & (red.astype(np.int16) > green.astype(np.int16) * 1.25 + 10)
        & (red.astype(np.int16) > blue.astype(np.int16) * 1.25 + 10)
    )
    thresholds = (120, 140, 160, 180, 200) if polarity == "dark_on_light" else (210,)
    results: list[np.ndarray] = []
    for threshold in thresholds:
        base_foreground = (
            crop < threshold
            if polarity == "dark_on_light"
            else canonical_gray[y0:y1, x0:x1] < threshold
        )
        foreground = ((base_foreground | red_pixels) * 255).astype(np.uint8)
        horizontal = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, foreground.shape[1] // 3), 1)),
        )
        foreground = cv2.subtract(foreground, horizontal)

        # A measure number may sit just to either side of the bar line. Keep the
        # generous horizontal crop, but remove full-height bar fragments. Dark
        # video backgrounds also need tiny texture specks removed; light score
        # paper keeps them because they can be real parts of an 11 px digit.
        component_count, component_labels, component_stats, _centroids = cv2.connectedComponentsWithStats(
            (foreground > 0).astype(np.uint8),
            connectivity=8,
        )
        cleaned = np.zeros_like(foreground)
        minimum_area = max(5, round(spacing * 0.25)) if polarity == "light_on_dark" else 1
        for component_index in range(1, component_count):
            _x, _y, component_width, component_height, area = component_stats[component_index]
            is_bar_fragment = component_height >= spacing * 1.05 and component_width <= spacing * 0.5
            if area < minimum_area or is_bar_fragment:
                continue
            cleaned[component_labels == component_index] = 255
        if np.count_nonzero(cleaned) >= 8:
            foreground = cleaned

        points = cv2.findNonZero(foreground)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        if width < 2 or height < 3:
            continue
        candidate = foreground[y : y + height, x : x + width]
        if not any(candidate.shape == item.shape and np.array_equal(candidate, item) for item in results):
            results.append(candidate)
    return results


def parse_frame(source: FrameInput) -> ParsedFrame:
    color = cv2.imread(str(source.path), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"无法读取切片：{source.path.name}")
    source_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    layout = detect_score_layout(source_gray)
    gray = layout.gray.copy()
    channel_range = np.max(color, axis=2).astype(np.int16) - np.min(color, axis=2).astype(np.int16)
    saturated_overlay = (channel_range >= 36) & (np.max(color, axis=2) >= 70)
    gray[saturated_overlay] = 255
    staff_lines = layout.staff_lines
    boundaries = detect_measure_boundaries(gray, staff_lines)
    spacing = float(statistics.median(np.diff(staff_lines)))
    label_staff_top = layout.notation_staff_lines[0] if layout.notation_staff_lines else staff_lines[0]
    label_spacing = (
        float(statistics.median(np.diff(layout.notation_staff_lines)))
        if layout.notation_staff_lines
        else spacing
    )
    measures: list[MeasureGeometry] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < spacing * 4:
            continue
        label_images = _clean_measure_labels(
            color,
            gray,
            left,
            label_staff_top,
            label_spacing,
            layout.polarity,
        )
        measures.append(
            MeasureGeometry(
                left=left,
                right=right,
                label_image=label_images[0] if label_images else None,
                label_alternates=label_images[1:],
            )
        )
    if len(measures) < 2:
        raise ValueError("切片内可用的完整小节少于 2 个")
    effective_boundaries = [measures[0].left, *(measure.right for measure in measures)]
    tokens = extract_note_tokens(gray, staff_lines)
    assign_rhythm_units(gray, staff_lines, tokens)
    return ParsedFrame(
        source=source,
        gray=gray,
        staff_lines=staff_lines,
        measures=measures,
        tokens=tokens,
        highlighted_index=detect_highlighted_measure(gray, staff_lines, effective_boundaries),
        layout=layout.layout,
        polarity=layout.polarity,
        notation_staff_lines=layout.notation_staff_lines,
    )


def _normalise_ocr_cell(foreground: np.ndarray, width: int = 200, height: int = 72) -> np.ndarray:
    points = cv2.findNonZero(foreground)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    if points is None:
        return canvas
    x, y, source_width, source_height = cv2.boundingRect(points)
    crop = foreground[y : y + source_height, x : x + source_width]
    scale = min((width - 24) / source_width, (height - 8) / source_height)
    resized = cv2.resize(
        crop,
        (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    resized = np.where(resized >= 96, 0, 255).astype(np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _run_ai_token_ocr(frame: ParsedFrame, vision_model_url: str) -> None:
    encoded_images: list[str] = []
    for token in frame.tokens:
        boxes = [glyph.box for glyph in token.glyphs if glyph.box is not None]
        if not boxes:
            encoded_images.append("")
            continue
        left = max(0, min(box[0] for box in boxes) - 4)
        top = max(0, min(box[1] for box in boxes) - 4)
        right = min(frame.gray.shape[1], max(box[0] + box[2] for box in boxes) + 4)
        bottom = min(frame.gray.shape[0], max(box[1] + box[3] for box in boxes) + 4)
        crop = frame.gray[top:bottom, left:right]
        success, payload = cv2.imencode(".png", crop)
        encoded_images.append(base64.b64encode(payload).decode("ascii") if success else "")

    if not encoded_images or any(not value for value in encoded_images):
        return
    endpoint = vision_model_url.rstrip("/")
    if not endpoint.endswith("/v1/fret-ocr"):
        endpoint += "/v1/fret-ocr"
    try:
        response = httpx.post(endpoint, json={"images": encoded_images}, timeout=45.0)
        response.raise_for_status()
        predictions = response.json().get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(frame.tokens):
            raise ValueError("本地 AI 返回的预测数量与品位候选不一致")
        for token, prediction in zip(frame.tokens, predictions, strict=True):
            text = str(prediction.get("text") or "") if isinstance(prediction, dict) else ""
            confidence = float(prediction.get("confidence") or 0) if isinstance(prediction, dict) else 0.0
            if not text.isdigit() or int(text) > 36 or len(text) != len(token.glyphs):
                continue
            token.raw_text = text
            token.raw_confidence = max(0.0, min(100.0, confidence * 100))
            for glyph, label in zip(token.glyphs, text):
                glyph.raw_label = label
                glyph.raw_confidence = token.raw_confidence
            frame.ai_recognized_tokens += 1
    except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
        frame.ai_error = str(exc)[:180]


def _run_frame_ocr(
    frame: ParsedFrame,
    work_dir: Path,
    tesseract_path: str,
    vision_model_url: str | None = None,
) -> None:
    targets: list[tuple[str, int, np.ndarray]] = []
    for index, token in enumerate(frame.tokens):
        gap = 4
        token_width = sum(glyph.image.shape[1] for glyph in token.glyphs) + gap * (len(token.glyphs) - 1)
        token_height = max(glyph.image.shape[0] for glyph in token.glyphs)
        foreground = np.zeros((token_height, token_width), dtype=np.uint8)
        cursor = 0
        for glyph in token.glyphs:
            y = (token_height - glyph.image.shape[0]) // 2
            foreground[y : y + glyph.image.shape[0], cursor : cursor + glyph.image.shape[1]] = glyph.image
            cursor += glyph.image.shape[1] + gap
        targets.append(("token", index, foreground))
    for index, measure in enumerate(frame.measures):
        if measure.label_image is not None:
            targets.append(("measure", index, measure.label_image))
            targets.extend(
                ("measure", index, alternate) for alternate in measure.label_alternates
            )
    if not targets:
        return

    row_height = 88
    sheet_width = 220
    sheet = np.full(((len(targets) + 2) * row_height, sheet_width), 255, dtype=np.uint8)
    for index, (_kind, _target_index, image) in enumerate(targets):
        cell = _normalise_ocr_cell(image, sheet_width, 72)
        top = (index + 1) * row_height + (row_height - cell.shape[0]) // 2
        sheet[top : top + cell.shape[0], :] = cell

    work_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = work_dir / f"ocr-{frame.source.source_frame or frame.source.path.stem}.png"
    cv2.imwrite(str(sheet_path), sheet)
    try:
        result = subprocess.run(
            [
                tesseract_path,
                str(sheet_path),
                "stdout",
                "--psm",
                "6",
                "-c",
                "tessedit_char_whitelist=0123456789",
                "-c",
                "user_defined_dpi=300",
                "tsv",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    finally:
        sheet_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "Tesseract OCR 执行失败")[-500:])

    words_by_row: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text.isdigit():
            continue
        try:
            center_y = float(row["top"]) + float(row["height"]) / 2
            target_row = math.floor(center_y / row_height) - 1
            confidence = max(0.0, float(row.get("conf") or 0))
            left = int(row.get("left") or 0)
        except (TypeError, ValueError):
            continue
        if 0 <= target_row < len(targets):
            words_by_row[target_row].append((left, text, confidence))

    measure_readings: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for row_index, words in words_by_row.items():
        words.sort(key=lambda item: item[0])
        text = "".join(item[1] for item in words)
        confidence = min(item[2] for item in words)
        kind, target_index, _image = targets[row_index]
        if kind == "token":
            token = frame.tokens[target_index]
            token.raw_text = text
            token.raw_confidence = confidence
            if len(text) == len(token.glyphs):
                for glyph, label in zip(token.glyphs, text):
                    glyph.raw_label = label
                    glyph.raw_confidence = confidence
        else:
            if int(text) <= 1000:
                measure_readings[target_index].append((text, confidence))

    unique_readings: dict[int, dict[str, float]] = {
        index: {
            text: max(confidence for candidate, confidence in readings if candidate == text)
            for text in {candidate for candidate, _confidence in readings}
        }
        for index, readings in measure_readings.items()
    }
    sequence_votes: dict[int, tuple[set[int], float]] = {}
    for index, readings in unique_readings.items():
        for text, confidence in readings.items():
            start = int(text) - index
            if start < 0:
                continue
            indices, weight = sequence_votes.setdefault(start, (set(), 0.0))
            indices.add(index)
            sequence_votes[start] = (indices, weight + max(10.0, confidence))

    winning_start: int | None = None
    if sequence_votes:
        candidate_start, (indices, _weight) = max(
            sequence_votes.items(),
            key=lambda item: (len(item[1][0]), item[1][1]),
        )
        if len(indices) >= 2:
            winning_start = candidate_start

    for index, readings in unique_readings.items():
        if winning_start is not None:
            text = str(winning_start + index)
            if text not in readings:
                continue
        else:
            text = max(readings, key=lambda candidate: readings[candidate])
        measure = frame.measures[index]
        measure.raw_label = text
        measure.raw_label_confidence = readings[text]

    if vision_model_url:
        _run_ai_token_ocr(frame, vision_model_url)


def _technique_from_text(text: str) -> str | None:
    normalized = re.sub(r"[^A-Z]", "", text.upper())
    if not normalized:
        return None
    if "PM" in normalized:
        return "palm_mute"
    if any(value in normalized for value in ("NH", "AH", "PH", "HARM")):
        return "harmonic"
    if "FULL" in normalized or "BEND" in normalized:
        return "bend"
    if normalized in {"SL", "SLIDE"}:
        return "slide"
    if normalized in {"LR", "LETRING"}:
        return "let_ring"
    if normalized == "H":
        return "hammer_on"
    if normalized == "P":
        return "pull_off"
    return None


def tempo_from_ocr_text(text: str) -> float | None:
    """Read an explicit tempo marking without mistaking measure or fret numbers for BPM."""
    normalized = text.upper().replace(",", ".")
    patterns = (
        r"(?:B\s*P\s*M|TEMPO)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)",
        r"(\d{2,3}(?:\.\d+)?)\s*(?:B\s*P\s*M)",
        r"(?:[♩♪]|\b(?:Q|J)\b)\s*=\s*(\d{2,3}(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        value = float(match.group(1))
        if 30 <= value <= 300:
            return value
    return None


def detect_tempo_from_images(
    image_paths: list[Path],
    *,
    tesseract_path: str = "tesseract",
    sample_limit: int = 8,
) -> float | None:
    """Quickly scan early score/video crops for an explicit BPM marking."""
    if not image_paths:
        return None
    sampled = image_paths[:sample_limit]
    candidates: list[float] = []
    for image_path in sampled:
        try:
            result = subprocess.run(
                [tesseract_path, str(image_path), "stdout", "--psm", "11", "-l", "eng"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        value = tempo_from_ocr_text(result.stdout)
        if value is not None:
            candidates.append(value)
    return round(float(statistics.median(candidates)), 1) if candidates else None


def _run_technique_ocr(frame: ParsedFrame, work_dir: Path, tesseract_path: str) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    image_path = work_dir / f"technique-{frame.source.source_frame or frame.source.path.stem}.png"
    cv2.imwrite(str(image_path), frame.gray)
    try:
        result = subprocess.run(
            [
                tesseract_path,
                str(image_path),
                "stdout",
                "--psm",
                "11",
                "-l",
                "eng",
                "-c",
                "user_defined_dpi=300",
                "tsv",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    finally:
        image_path.unlink(missing_ok=True)
    if result.returncode != 0:
        return

    spacing = float(statistics.median(np.diff(frame.staff_lines)))
    marks: list[TechniqueMark] = []
    rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
    tempo_lines: dict[tuple[str, str, str, str], list[tuple[float, str]]] = defaultdict(list)
    for row in rows:
        raw_text = (row.get("text") or "").strip()
        if raw_text:
            try:
                word_confidence = max(0.0, float(row.get("conf") or 0))
            except (TypeError, ValueError):
                word_confidence = 0.0
            line_key = tuple(str(row.get(name) or "") for name in ("page_num", "block_num", "par_num", "line_num"))
            tempo_lines[line_key].append((word_confidence, raw_text))
        technique = _technique_from_text(raw_text)
        if not technique:
            continue
        try:
            confidence = max(0.0, float(row.get("conf") or 0))
            left = float(row.get("left") or 0)
            top = float(row.get("top") or 0)
            width = float(row.get("width") or 0)
            height = float(row.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if confidence < 20 or top + height > frame.staff_lines[0] + spacing * 0.8:
            continue
        marks.append(
            TechniqueMark(
                x=left + width / 2,
                end_x=left + max(width, spacing * 2) if technique == "palm_mute" else None,
                technique=technique,
                raw_text=raw_text,
                confidence=confidence,
            )
        )
    frame.technique_marks = marks
    candidates: list[tuple[float, float]] = []
    for words in tempo_lines.values():
        value = tempo_from_ocr_text(" ".join(word for _confidence, word in words))
        if value is not None:
            candidates.append((value, statistics.mean(confidence for confidence, _word in words)))
    frame.tempo_candidates = candidates


@dataclass
class _GlyphCluster:
    members: list[DigitGlyph]
    prototype: np.ndarray
    label: str | None = None
    confidence: float = 0.0


def _cluster_and_label_glyphs(frames: list[ParsedFrame]) -> None:
    glyphs = [glyph for frame in frames for token in frame.tokens for glyph in token.glyphs]
    clusters: list[_GlyphCluster] = []
    for glyph in glyphs:
        best_index = -1
        best_distance = math.inf
        for index, cluster in enumerate(clusters):
            distance = float(np.mean(glyph.feature != cluster.prototype))
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index >= 0 and best_distance <= 0.075:
            cluster = clusters[best_index]
            cluster.members.append(glyph)
            cluster.prototype = (
                np.mean([member.feature for member in cluster.members], axis=0) >= 0.5
            ).astype(np.uint8)
        else:
            clusters.append(_GlyphCluster([glyph], glyph.feature.copy()))

    labelled_clusters: list[_GlyphCluster] = []
    for cluster in clusters:
        votes: dict[str, float] = defaultdict(float)
        for glyph in cluster.members:
            if glyph.raw_label and glyph.raw_label.isdigit():
                votes[glyph.raw_label] += max(10.0, glyph.raw_confidence)
        if votes:
            winner, winner_weight = max(votes.items(), key=lambda item: item[1])
            total = sum(votes.values())
            confidence = 100 * winner_weight / total
            cluster.label = winner
            cluster.confidence = confidence
            for glyph in cluster.members:
                if not glyph.raw_label:
                    glyph.label = winner
                    glyph.confidence = confidence
                elif glyph.raw_label == winner:
                    glyph.label = glyph.raw_label
                    glyph.confidence = max(glyph.raw_confidence, confidence)
                elif confidence >= 65 and len(cluster.members) >= 3:
                    glyph.label = winner
                    glyph.confidence = confidence
                else:
                    glyph.label = glyph.raw_label
                    glyph.confidence = max(10.0, glyph.raw_confidence)
            labelled_clusters.append(cluster)

    for cluster in clusters:
        if any(member.label for member in cluster.members) or not labelled_clusters:
            continue
        nearest = min(
            labelled_clusters,
            key=lambda candidate: float(np.mean(cluster.prototype != candidate.prototype)),
        )
        distance = float(np.mean(cluster.prototype != nearest.prototype))
        if distance <= 0.15 and nearest.label:
            label = nearest.label
            confidence = max(0.0, nearest.confidence * (1 - distance / 0.15))
            for glyph in cluster.members:
                glyph.label = label
                glyph.confidence = confidence


def _measure_start_votes(frame: ParsedFrame) -> dict[int, tuple[int, float]]:
    votes: dict[int, tuple[int, float]] = {}
    for index, measure in enumerate(frame.measures):
        if not measure.raw_label or not measure.raw_label.isdigit():
            continue
        value = int(measure.raw_label)
        start = value - index
        if start < 0 or start > 100000:
            continue
        count, weight = votes.get(start, (0, 0.0))
        votes[start] = (count + 1, weight + max(10.0, measure.raw_label_confidence))
    return votes


def _infer_frame_start(frame: ParsedFrame) -> None:
    votes = _measure_start_votes(frame)
    if not votes:
        return
    start, (count, weight) = max(votes.items(), key=lambda item: (item[1][0], item[1][1]))
    if count >= 2:
        frame.start_measure = start
        frame.start_measure_confidence = min(100.0, weight / max(count, 1))


def _smooth_frame_starts(frames: list[ParsedFrame]) -> None:
    """Resolve OCR anchors as one monotonic video sequence.

    Scrolling score videos advance roughly with musical time. This Viterbi pass
    prevents a truncated three-digit label from making a frame jump backwards
    or hundreds of measures forward while still allowing tempo variation.
    """
    if not frames:
        return
    frames.sort(key=lambda frame: frame.source.time_seconds)
    observed = [
        int(measure.raw_label)
        for frame in frames
        for measure in frame.measures
        if measure.raw_label and measure.raw_label.isdigit() and int(measure.raw_label) <= 1000
    ]
    if not observed:
        return
    max_state = min(1000, max(256, (max(observed) if observed else 200) + 12))
    negative_infinity = -1e18

    emissions: list[np.ndarray] = []
    for frame in frames:
        scores = np.zeros(max_state + 1, dtype=np.float64)
        for state in range(max_state + 1):
            score = 0.0
            for index, measure in enumerate(frame.measures):
                if not measure.raw_label or not measure.raw_label.isdigit():
                    continue
                if int(measure.raw_label) == state + index:
                    score += 35.0 + max(0.0, measure.raw_label_confidence)
            scores[state] = score
        emissions.append(scores)

    previous_scores = emissions[0] - np.arange(max_state + 1) * 0.25
    back_pointers: list[np.ndarray] = []
    for frame_index in range(1, len(frames)):
        time_delta = max(
            0.05,
            frames[frame_index].source.time_seconds - frames[frame_index - 1].source.time_seconds,
        )
        expected_delta = time_delta * 0.75
        max_jump = max(3, math.ceil(time_delta * 2.0) + 2)
        current_scores = np.full(max_state + 1, negative_infinity, dtype=np.float64)
        pointers = np.zeros(max_state + 1, dtype=np.int32)
        for state in range(max_state + 1):
            lower = max(0, state - max_jump)
            previous_states = np.arange(lower, state + 1)
            deltas = state - previous_states
            transition = previous_scores[lower : state + 1] - np.abs(deltas - expected_delta) * 7.0
            best_offset = int(np.argmax(transition))
            current_scores[state] = transition[best_offset] + emissions[frame_index][state]
            pointers[state] = lower + best_offset
        back_pointers.append(pointers)
        previous_scores = current_scores

    state = int(np.argmax(previous_scores))
    sequence = [state]
    for pointers in reversed(back_pointers):
        state = int(pointers[state])
        sequence.append(state)
    sequence.reverse()

    for frame, start, emission in zip(frames, sequence, emissions):
        frame.start_measure = start
        matching = [
            measure.raw_label_confidence
            for index, measure in enumerate(frame.measures)
            if measure.raw_label
            and measure.raw_label.isdigit()
            and int(measure.raw_label) == start + index
        ]
        frame.start_measure_confidence = (
            min(100.0, statistics.mean(matching)) if matching else min(45.0, emission[start] / 2)
        )


def _candidate_from_geometry(
    frame: ParsedFrame, index: int, number: int
) -> MeasureCandidate:
    geometry = frame.measures[index]
    width = geometry.right - geometry.left
    token_groups: list[list[tuple[NoteToken, TabNote]]] = []
    confidences: list[float] = []
    spacing = float(statistics.median(np.diff(frame.staff_lines)))
    for token in frame.tokens:
        if not (geometry.left <= token.x < geometry.right):
            continue
        text = token.text
        if not text or not text.isdigit():
            continue
        fret = int(text)
        if fret > 36:
            continue
        note = TabNote(string=token.string, fret=fret)
        if token_groups and abs(token.x - token_groups[-1][0][0].x) <= spacing * 0.45:
            token_groups[-1].append((token, note))
        else:
            token_groups.append([(token, note)])
        confidences.append(token.confidence)

    group_centers = [statistics.mean(token.x for token, _note in group) for group in token_groups]
    group_techniques: dict[int, str] = {}
    for mark in frame.technique_marks:
        if not (geometry.left - spacing * 2 <= mark.x < geometry.right):
            continue
        if mark.technique == "palm_mute":
            start_x = max(geometry.left, mark.x - spacing * 2)
            end_x = max(mark.end_x or start_x, start_x + spacing * 4)
            for group_index, center in enumerate(group_centers):
                if start_x <= center <= end_x:
                    group_techniques[group_index] = mark.technique
            continue
        before = [group_index for group_index, center in enumerate(group_centers) if center <= mark.x + spacing]
        if before:
            group_techniques[before[-1]] = mark.technique

    events: list[TabEvent] = []
    signature_items: list[tuple[float, int, int]] = []
    cursor = 0.0
    for group_index, group in enumerate(token_groups):
        durations = [token.duration_units for token, _note in group if token.duration_units]
        if durations:
            duration = max(0.5, round(statistics.median(durations) * 2) / 2)
        elif len(token_groups) >= 6:
            duration = 1
        else:
            duration = max(1, round(EIGHTH_UNITS_PER_MEASURE / max(1, len(token_groups))))
        if cursor >= EIGHTH_UNITS_PER_MEASURE:
            break
        duration = min(duration, EIGHTH_UNITS_PER_MEASURE - cursor)
        technique = group_techniques.get(group_index)
        notes = tuple(
            sorted(
                {TabNote(note.string, note.fret, technique) for _token, note in group},
                key=lambda note: note.string,
            )
        )
        events.append(TabEvent(onset=cursor, duration=duration, notes=notes))
        signature_items.extend((cursor, note.string, note.fret) for note in notes)
        cursor += duration
    average_confidence = statistics.mean(confidences) if confidences else 0.0
    quality = average_confidence + min(20, len(signature_items))
    if frame.highlighted_index == index:
        quality -= 12
    crop_box = (
        max(0, math.floor(geometry.left - spacing * 0.2)),
        max(
            0,
            math.floor(
                (frame.notation_staff_lines[0] if frame.notation_staff_lines else frame.staff_lines[0])
                - spacing * 2.5
            ),
        ),
        min(frame.gray.shape[1], math.ceil(geometry.right + spacing * 0.2)),
        min(frame.gray.shape[0], math.ceil(frame.staff_lines[-1] + spacing * 3.0)),
    )
    return MeasureCandidate(
        number=number,
        events=tuple(events),
        quality=quality,
        source_time=frame.source.time_seconds,
        signature=tuple(sorted(signature_items)),
        source_path=frame.source.path,
        crop_box=crop_box,
    )


def _collect_measures(frames: list[ParsedFrame]) -> tuple[dict[int, MeasureCandidate], list[ParsedFrame]]:
    candidates_by_number: dict[int, list[MeasureCandidate]] = defaultdict(list)

    def resolved_measures() -> dict[int, MeasureCandidate]:
        resolved: dict[int, MeasureCandidate] = {}
        for number, candidates in candidates_by_number.items():
            by_signature: dict[tuple[tuple[int, int, int], ...], list[MeasureCandidate]] = defaultdict(list)
            for candidate in candidates:
                by_signature[candidate.signature].append(candidate)
            winning_group = max(
                by_signature.values(),
                key=lambda group: (len(group), statistics.mean(item.quality for item in group)),
            )
            resolved[number] = max(winning_group, key=lambda item: item.quality)
        return resolved

    unresolved: list[ParsedFrame] = []
    for frame in frames:
        if frame.start_measure is None:
            unresolved.append(frame)
            continue
        for index in range(len(frame.measures)):
            number = frame.start_measure + index
            candidate = _candidate_from_geometry(frame, index, number)
            candidates_by_number[number].append(candidate)

    for _ in range(3):
        measures = resolved_measures()
        if not unresolved or not measures:
            break
        remaining: list[ParsedFrame] = []
        signature_index: dict[tuple[tuple[int, int, int], ...], list[int]] = defaultdict(list)
        for number, measure in measures.items():
            if measure.signature:
                signature_index[measure.signature].append(number)
        for frame in unresolved:
            start_votes: dict[int, int] = defaultdict(int)
            provisional = [
                _candidate_from_geometry(frame, index, 0)
                for index in range(len(frame.measures))
            ]
            for index, candidate in enumerate(provisional):
                for known_number in signature_index.get(candidate.signature, []):
                    start_votes[known_number - index] += 1
            if not start_votes:
                remaining.append(frame)
                continue
            start, matches = max(start_votes.items(), key=lambda item: item[1])
            if matches < 2 or start < 0:
                remaining.append(frame)
                continue
            frame.start_measure = start
            frame.start_measure_confidence = 45.0
            for index in range(len(frame.measures)):
                number = start + index
                candidate = _candidate_from_geometry(frame, index, number)
                candidates_by_number[number].append(candidate)
        unresolved = remaining
    return resolved_measures(), unresolved


def _pitch_xml(parent: ET.Element, midi: int) -> None:
    names = (("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0))
    step, alter = names[midi % 12]
    pitch = ET.SubElement(parent, "pitch")
    ET.SubElement(pitch, "step").text = step
    if alter:
        ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(midi // 12 - 1)


def _duration_notation(duration: float) -> tuple[str, int]:
    mapping = {
        1: ("64th", 0),
        2: ("32nd", 0),
        3: ("32nd", 1),
        4: ("16th", 0),
        6: ("16th", 1),
        7: ("16th", 2),
        8: ("eighth", 0),
        12: ("eighth", 1),
        14: ("eighth", 2),
        16: ("quarter", 0),
        24: ("quarter", 1),
        28: ("quarter", 2),
        32: ("half", 0),
        48: ("half", 1),
        56: ("half", 2),
        64: ("whole", 0),
    }
    return mapping.get(round(duration * 8), ("eighth", 0))


def _musicxml_duration(duration: float) -> str:
    return str(round(duration * 8))


def _append_rest(measure_xml: ET.Element, duration: float) -> None:
    remaining = duration
    for chunk in (8, 7, 6, 4, 3.5, 3, 2, 1.75, 1.5, 1, 0.875, 0.75, 0.5, 0.375, 0.25, 0.125):
        while remaining + 1e-9 >= chunk:
            note = ET.SubElement(measure_xml, "note")
            ET.SubElement(note, "rest")
            ET.SubElement(note, "duration").text = _musicxml_duration(chunk)
            note_type, dot_count = _duration_notation(chunk)
            ET.SubElement(note, "type").text = note_type
            for _ in range(dot_count):
                ET.SubElement(note, "dot")
            remaining -= chunk


def _measure_capacity_eighths(time_signature: tuple[int, int]) -> float:
    numerator, denominator = time_signature
    if numerator < 1 or denominator not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("拍号无效")
    return numerator * 8 / denominator


def _measure_validation(
    events: tuple[TabEvent, ...] | list[TabEvent],
    capacity: float,
) -> dict:
    ordered = sorted(events, key=lambda event: event.onset)
    cursor = 0.0
    committed = 0.0
    has_gaps = False
    overflow = 0.0
    for event in ordered:
        if event.onset > cursor + 1e-9:
            has_gaps = True
        if event.onset < cursor - 1e-9:
            overflow = max(overflow, cursor - event.onset)
        cursor = max(cursor, event.onset + event.duration)
        committed += event.duration
    if cursor < capacity - 1e-9:
        has_gaps = True
    overflow = max(overflow, cursor - capacity, committed - capacity, 0.0)
    missing = max(0.0, capacity - committed)
    complete = not has_gaps and overflow <= 1e-9 and abs(committed - capacity) <= 1e-9
    return {
        "capacity_eighths": round(capacity, 3),
        "committed_eighths": round(committed, 3),
        "missing_eighths": round(missing, 3),
        "overflow_eighths": round(overflow, 3),
        "has_gaps": has_gaps,
        "is_complete": complete,
    }


def _tab_note_from_dict(value: dict) -> TabNote:
    technique = value.get("technique") or None
    if technique is not None and technique not in TAB_TECHNIQUES:
        raise ValueError(f"不支持的演奏技巧：{technique}")
    return TabNote(int(value["string"]), int(value["fret"]), technique)


def _tab_note_to_dict(note: TabNote) -> dict:
    value = {"string": note.string, "fret": note.fret}
    if note.technique:
        value["technique"] = note.technique
    return value


def _technique_links(
    measures: dict[int, MeasureCandidate],
) -> tuple[dict[tuple[int, int, int], str], dict[tuple[int, int, int], str]]:
    linked = {"legato", "slide", "hammer_on", "pull_off"}
    starts: dict[tuple[int, int, int], str] = {}
    stops: dict[tuple[int, int, int], str] = {}
    by_string: dict[int, list[tuple[tuple[int, int, int], TabNote]]] = defaultdict(list)
    for measure_number, candidate in sorted(measures.items()):
        for event_index, event in enumerate(candidate.events):
            for note in event.notes:
                key = (measure_number, event_index, note.string)
                by_string[note.string].append((key, note))
    for notes in by_string.values():
        for (key, note), (next_key, _next_note) in zip(notes, notes[1:]):
            if note.technique in linked:
                starts[key] = note.technique
                stops[next_key] = note.technique
    return starts, stops


def _build_musicxml(
    title: str,
    measures: dict[int, MeasureCandidate],
    output_path: Path,
    tempo_bpm: float,
) -> None:
    score = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title
    identification = ET.SubElement(score, "identification")
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "Nocturne TAB Editor"

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Electric Guitar"
    score_instrument = ET.SubElement(score_part, "score-instrument", id="P1-I1")
    ET.SubElement(score_instrument, "instrument-name").text = "Electric Guitar"
    midi_instrument = ET.SubElement(score_part, "midi-instrument", id="P1-I1")
    ET.SubElement(midi_instrument, "midi-channel").text = "1"
    ET.SubElement(midi_instrument, "midi-program").text = "30"

    part = ET.SubElement(score, "part", id="P1")
    technique_starts, technique_stops = _technique_links(measures)
    start_number = min(measures)
    end_number = max(measures)
    previous_time_signature: tuple[int, int] | None = None
    for number in range(start_number, end_number + 1):
        measure_xml = ET.SubElement(part, "measure", number=str(number))
        candidate = measures.get(number)
        time_signature = candidate.time_signature if candidate else previous_time_signature or (4, 4)
        if number == start_number or time_signature != previous_time_signature:
            attributes = ET.SubElement(measure_xml, "attributes")
            if number == start_number:
                # Sixteen divisions per quarter preserve double-dotted sixteenth notes.
                ET.SubElement(attributes, "divisions").text = "16"
                key = ET.SubElement(attributes, "key")
                ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(time_signature[0])
            ET.SubElement(time, "beat-type").text = str(time_signature[1])
            if number == start_number:
                clef = ET.SubElement(attributes, "clef")
                ET.SubElement(clef, "sign").text = "TAB"
                ET.SubElement(clef, "line").text = "5"
                staff_details = ET.SubElement(attributes, "staff-details")
                ET.SubElement(staff_details, "staff-lines").text = "6"
                tuning = ((1, "E", 2), (2, "A", 2), (3, "D", 3), (4, "G", 3), (5, "B", 3), (6, "E", 4))
                for line, step, octave in tuning:
                    staff_tuning = ET.SubElement(staff_details, "staff-tuning", line=str(line))
                    ET.SubElement(staff_tuning, "tuning-step").text = step
                    ET.SubElement(staff_tuning, "tuning-octave").text = str(octave)
        if number == start_number:
            direction = ET.SubElement(measure_xml, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = str(round(tempo_bpm, 1))
            ET.SubElement(direction, "sound", tempo=str(round(tempo_bpm, 2)))

        previous_time_signature = time_signature
        capacity = _measure_capacity_eighths(time_signature)
        if candidate is None or not candidate.events:
            _append_rest(measure_xml, capacity)
            continue
        cursor = 0
        for event_index, event in enumerate(candidate.events):
            if event.onset > cursor:
                _append_rest(measure_xml, event.onset - cursor)
            if not event.notes:
                _append_rest(measure_xml, event.duration)
                cursor = max(cursor, event.onset + event.duration)
                continue
            for note_index, tab_note in enumerate(event.notes):
                note_key = (number, event_index, tab_note.string)
                outgoing = technique_starts.get(note_key)
                incoming = technique_stops.get(note_key)
                note = ET.SubElement(measure_xml, "note")
                if note_index:
                    ET.SubElement(note, "chord")
                midi = STANDARD_TUNING_MIDI[tab_note.string - 1] + tab_note.fret
                _pitch_xml(note, midi)
                ET.SubElement(note, "duration").text = _musicxml_duration(event.duration)
                ET.SubElement(note, "voice").text = "1"
                note_type, dot_count = _duration_notation(event.duration)
                ET.SubElement(note, "type").text = note_type
                for _ in range(dot_count):
                    ET.SubElement(note, "dot")
                if tab_note.technique == "dead_note":
                    ET.SubElement(note, "notehead").text = "x"
                ET.SubElement(note, "staff").text = "1"
                notations = ET.SubElement(note, "notations")
                if tab_note.technique == "let_ring":
                    ET.SubElement(notations, "tied", type="let-ring")
                if incoming in {"legato", "hammer_on", "pull_off"}:
                    ET.SubElement(notations, "slur", type="stop", number="1")
                if outgoing in {"legato", "hammer_on", "pull_off"}:
                    ET.SubElement(notations, "slur", type="start", number="1")
                if incoming == "slide":
                    ET.SubElement(notations, "slide", type="stop", number="1")
                if outgoing == "slide":
                    ET.SubElement(notations, "slide", type="start", number="1")
                technical = ET.SubElement(notations, "technical")
                if tab_note.technique == "harmonic":
                    harmonic = ET.SubElement(technical, "harmonic")
                    ET.SubElement(harmonic, "natural")
                ET.SubElement(technical, "string").text = str(tab_note.string)
                ET.SubElement(technical, "fret").text = str(tab_note.fret)
                if incoming == "hammer_on":
                    ET.SubElement(technical, "hammer-on", type="stop", number="1")
                if outgoing == "hammer_on":
                    ET.SubElement(technical, "hammer-on", type="start", number="1").text = "H"
                if incoming == "pull_off":
                    ET.SubElement(technical, "pull-off", type="stop", number="1")
                if outgoing == "pull_off":
                    ET.SubElement(technical, "pull-off", type="start", number="1").text = "P"
                if tab_note.technique == "bend":
                    bend = ET.SubElement(technical, "bend")
                    ET.SubElement(bend, "bend-alter").text = "2"
                if tab_note.technique == "vibrato":
                    ET.SubElement(technical, "other-technical").text = "vibrato"
                elif tab_note.technique == "palm_mute":
                    ET.SubElement(technical, "other-technical").text = "palm mute"
                elif tab_note.technique == "dead_note":
                    ET.SubElement(technical, "other-technical").text = "dead note"
                elif tab_note.technique in {"legato", "slide", "hammer_on", "pull_off"} and not outgoing:
                    ET.SubElement(technical, "other-technical").text = tab_note.technique.replace("_", " ")
            cursor = max(cursor, event.onset + event.duration)
        if cursor < capacity:
            _append_rest(measure_xml, capacity - cursor)

    ET.indent(score, space="  ")
    xml_body = ET.tostring(score, encoding="unicode")
    output_path.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE score-partwise PUBLIC \"-//Recordare//DTD MusicXML 4.0 Partwise//EN\" "
        "\"http://www.musicxml.org/dtds/partwise.dtd\">\n"
        f"{xml_body}\n",
        encoding="utf-8",
    )


def _sync_suggestions(frames: list[ParsedFrame]) -> tuple[list[dict], float | None]:
    earliest: dict[int, float] = {}
    for frame in frames:
        if frame.start_measure is None or frame.highlighted_index is None:
            continue
        measure = frame.start_measure + frame.highlighted_index
        current = earliest.get(measure)
        if current is None or frame.source.time_seconds < current:
            earliest[measure] = frame.source.time_seconds
    ordered = sorted(earliest.items())
    suggestions = [
        {"measure_number": measure, "time_seconds": round(seconds, 3)}
        for measure, seconds in ordered
    ]
    seconds_per_measure: list[float] = []
    for (first_measure, first_time), (second_measure, second_time) in zip(ordered, ordered[1:]):
        measure_delta = second_measure - first_measure
        time_delta = second_time - first_time
        if 1 <= measure_delta <= 8 and time_delta > 0:
            estimate = time_delta / measure_delta
            if 0.3 <= estimate <= 8:
                seconds_per_measure.append(estimate)
    if not seconds_per_measure:
        return suggestions, None
    tempo = 240 / statistics.median(seconds_per_measure)
    return suggestions, min(300.0, max(30.0, tempo))


def create_blank_tab_score(
    output_dir: Path,
    *,
    title: str,
    measure_count: int = 8,
    tempo_bpm: float = 120.0,
) -> TabRecognitionResult:
    """Create a persistent, playable six-string TAB document for manual entry."""
    if not 1 <= measure_count <= 128:
        raise ValueError("空白六线谱的小节数必须在 1～128 之间")
    if not 30 <= tempo_bpm <= 300:
        raise ValueError("速度必须在 30～300 BPM 之间")

    output_dir.mkdir(parents=True, exist_ok=True)
    measures = {
        number: MeasureCandidate(number, (), 100.0, 0.0, ())
        for number in range(1, measure_count + 1)
    }
    score_path = output_dir / "manual-tab.musicxml"
    _build_musicxml(title, measures, score_path, tempo_bpm)
    summary = {
        "engine": "tab_manual_editor",
        "engine_label": "六线谱手动编辑器",
        "measure_count": measure_count,
        "start_measure": 1,
        "end_measure": measure_count,
        "estimated_tempo_bpm": round(tempo_bpm, 1),
        "time_signature_numerator": 4,
        "time_signature_denominator": 4,
        "invalid_measures": list(range(1, measure_count + 1)),
        "confidence": 1.0,
        "low_confidence_glyphs": 0,
        "missing_measures": [],
        "warnings": [
            "当前手动谱使用标准六弦调弦、4/4 拍与十六分音符网格",
        ],
    }
    diagnostics = {
        "summary": summary,
        "sync_suggestions": [],
        "parse_errors": [],
        "frames": [],
        "measures": [
            {
                "number": number,
                "quality": 100.0,
                "source_time": 0.0,
                "events": [],
                "time_signature": {"numerator": 4, "denominator": 4},
                "validation": _measure_validation([], EIGHTH_UNITS_PER_MEASURE),
            }
            for number in measures
        ],
    }
    diagnostics_path = output_dir / "recognition.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return TabRecognitionResult(score_path, diagnostics_path, summary)


def recognize_tab_frames(
    frames: list[FrameInput],
    output_dir: Path,
    *,
    title: str,
    tesseract_path: str = "tesseract",
    tempo_bpm: float | None = None,
    recognition_mode: str = "ocr",
    vision_model_url: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TabRecognitionResult:
    if not frames:
        raise ValueError("没有视频切片可供 TAB 识别")
    if recognition_mode not in {"ocr", "ai"}:
        raise ValueError("识别模式必须是 ocr 或 ai")
    if recognition_mode == "ai" and not vision_model_url:
        raise ValueError("本地 AI 品位服务尚未连接")
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed: list[ParsedFrame] = []
    parse_errors: list[str] = []
    ordered_frames = sorted(frames, key=lambda item: item.time_seconds)
    for frame_index, source in enumerate(ordered_frames, start=1):
        try:
            frame = parse_frame(source)
            _run_frame_ocr(
                frame,
                output_dir / ".ocr",
                tesseract_path,
                vision_model_url if recognition_mode == "ai" else None,
            )
            _run_technique_ocr(frame, output_dir / ".ocr", tesseract_path)
            _infer_frame_start(frame)
            parsed.append(frame)
        except Exception as exc:
            parse_errors.append(f"{source.path.name}: {str(exc)[:140]}")
        finally:
            if progress_callback:
                try:
                    progress_callback(frame_index, len(ordered_frames))
                except Exception:
                    pass
    if not parsed:
        raise RuntimeError("所有切片都无法识别为六线 TAB")

    _smooth_frame_starts(parsed)
    _cluster_and_label_glyphs(parsed)
    measures, unresolved = _collect_measures(parsed)
    note_measure_count = sum(bool(measure.events) for measure in measures.values())
    if note_measure_count < 2:
        raise RuntimeError("识别出的有效小节少于 2 个，请重新框选更清晰的 TAB 区域")
    usable = measures

    suggestions, estimated_tempo = _sync_suggestions(parsed)
    visual_tempos = [value for frame in parsed for value, confidence in frame.tempo_candidates if confidence >= 20]
    detected_visual_tempo = statistics.median(visual_tempos) if visual_tempos else None
    if tempo_bpm is not None:
        tempo = min(300.0, max(30.0, float(tempo_bpm)))
        tempo_source = "user"
    elif detected_visual_tempo is not None:
        tempo = detected_visual_tempo
        tempo_source = "visual_ocr"
    elif estimated_tempo is not None:
        tempo = estimated_tempo
        tempo_source = "video_timing"
    else:
        tempo = 120.0
        tempo_source = "default"
    score_path = output_dir / "recognized-tab.musicxml"
    _build_musicxml(title, usable, score_path, tempo)
    recognized_pdf_path: Path | None = None
    pdf_error: str | None = None
    measure_sources = [
        (number, candidate.source_path, candidate.crop_box)
        for number, candidate in sorted(usable.items())
        if candidate.source_path is not None and candidate.crop_box is not None
    ]
    try:
        recognized_pdf_path = build_recognized_score_pdf(
            measure_sources,
            output_dir / "recognized-score.pdf",
        )
    except (OSError, ValueError) as exc:
        pdf_error = str(exc)[:180]

    start_measure, end_measure = min(usable), max(usable)
    gaps = [number for number in range(start_measure, end_measure + 1) if number not in usable]
    glyphs = [glyph for frame in parsed for token in frame.tokens for glyph in token.glyphs]
    glyph_confidences = [glyph.confidence for glyph in glyphs if glyph.label]
    recognized_glyphs = len(glyph_confidences)
    total_glyphs = len(glyphs)
    glyph_coverage = recognized_glyphs / total_glyphs if total_glyphs else 0.0
    mean_glyph_confidence = statistics.mean(glyph_confidences) if glyph_confidences else 0.0
    confidence = mean_glyph_confidence / 100 * glyph_coverage
    low_confidence = sum(value < 60 for value in glyph_confidences)
    technique_counts: dict[str, int] = defaultdict(int)
    for frame in parsed:
        for mark in frame.technique_marks:
            technique_counts[mark.technique] += 1
    warnings = [
        "当前按六弦标准调弦和 4/4 拍生成草稿；校对器支持细化到十六分音符网格",
        "已尝试识别 H/P、sl.、P.M.、Full/Bend 与泛音文字；复杂曲线、揉弦和连音仍需在校对器中确认",
    ]
    layout_counts = {
        name: sum(frame.layout == name for frame in parsed)
        for name in sorted({frame.layout for frame in parsed})
    }
    polarity_counts = {
        name: sum(frame.polarity == name for frame in parsed)
        for name in sorted({frame.polarity for frame in parsed})
    }
    if layout_counts.get("staff_tab_pair"):
        warnings.append("已检测到上五线谱、下六线谱的联合谱；小节号与 PDF 裁切会保留上方五线谱")
    if polarity_counts.get("light_on_dark"):
        warnings.append("已对深色或半透明背景上的浅色谱线做自动反相与背景抑制")
    if tempo_source == "visual_ocr":
        warnings.append(f"已从谱面或视频文字识别到 {tempo:.1f} BPM；手动锁定后自动分析不会再覆盖")
    if pdf_error:
        warnings.append(f"按小节合成 PDF 失败，仍保留切片预览 PDF：{pdf_error}")
    if gaps:
        warnings.append(f"有 {len(gaps)} 个小节未获得可靠结果，已在导出谱中留空")
    if unresolved:
        warnings.append(f"有 {len(unresolved)} 张切片无法确定起始小节号")
    if glyph_coverage < 0.75:
        warnings.append(
            f"仅有 {recognized_glyphs}/{total_glyphs} 个品位数字获得可用标签，置信度已按覆盖率折减"
        )
    ai_recognized_tokens = sum(frame.ai_recognized_tokens for frame in parsed)
    ai_failed_frames = sum(frame.ai_error is not None for frame in parsed)
    if recognition_mode == "ai" and ai_failed_frames:
        warnings.append(f"本地 AI 有 {ai_failed_frames} 张切片调用失败，已自动回退到传统 OCR")

    summary = {
        "engine": "tab_cv_ai" if recognition_mode == "ai" else "tab_cv_tesseract",
        "engine_label": "本地 AI + OCR 六线 TAB（实验）" if recognition_mode == "ai" else "六线 TAB 专用识别（Beta）",
        "measure_count": len(usable),
        "start_measure": start_measure,
        "end_measure": end_measure,
        "estimated_tempo_bpm": round(tempo, 1),
        "tempo_source": tempo_source,
        "tempo_locked": tempo_bpm is not None,
        "time_signature_numerator": 4,
        "time_signature_denominator": 4,
        "confidence": round(confidence, 3),
        "glyph_coverage": round(glyph_coverage, 3),
        "recognized_glyphs": recognized_glyphs,
        "total_glyphs": total_glyphs,
        "low_confidence_glyphs": low_confidence,
        "ai_recognized_tokens": ai_recognized_tokens,
        "ai_failed_frames": ai_failed_frames,
        "technique_counts": dict(sorted(technique_counts.items())),
        "parsed_frames": len(parsed),
        "failed_frames": len(parse_errors),
        "unresolved_frames": len(unresolved),
        "missing_measures": gaps,
        "layout_counts": layout_counts,
        "polarity_counts": polarity_counts,
        "warnings": warnings,
    }
    invalid_measures = [
        number
        for number, candidate in sorted(usable.items())
        if not _measure_validation(candidate.events, EIGHTH_UNITS_PER_MEASURE)["is_complete"]
    ]
    summary["invalid_measures"] = invalid_measures
    if invalid_measures:
        warnings.append(
            f"有 {len(invalid_measures)} 个小节尚未用音符或实体休止填满 4/4 拍，校对器会标红"
        )
    diagnostics = {
        "summary": summary,
        "sync_suggestions": suggestions,
        "parse_errors": parse_errors,
        "frames": [
            {
                "name": frame.source.path.name,
                "time_seconds": round(frame.source.time_seconds, 3),
                "start_measure": frame.start_measure,
                "start_measure_confidence": round(frame.start_measure_confidence, 1),
                "highlighted_index": frame.highlighted_index,
                "raw_measure_labels": [measure.raw_label for measure in frame.measures],
                "layout": frame.layout,
                "polarity": frame.polarity,
                "ai_recognized_tokens": frame.ai_recognized_tokens,
                "ai_error": frame.ai_error,
            }
            for frame in parsed
        ],
        "measures": [
            {
                "number": number,
                "quality": round(candidate.quality, 2),
                "source_time": round(candidate.source_time, 3),
                "events": [
                    {
                        "onset_eighths": event.onset,
                        "duration_eighths": event.duration,
                        "notes": [_tab_note_to_dict(note) for note in event.notes],
                    }
                    for event in candidate.events
                ],
                "ocr_events": [
                    {
                        "onset_eighths": event.onset,
                        "duration_eighths": event.duration,
                        "notes": [_tab_note_to_dict(note) for note in event.notes],
                    }
                    for event in candidate.events
                ],
                "human_verified": False,
                "time_signature": {"numerator": 4, "denominator": 4},
                "validation": _measure_validation(candidate.events, EIGHTH_UNITS_PER_MEASURE),
            }
            for number, candidate in sorted(usable.items())
        ],
    }
    diagnostics_path = output_dir / "recognition.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return TabRecognitionResult(score_path, diagnostics_path, summary, suggestions, recognized_pdf_path)


def recognize_tab_measure(
    frame: FrameInput,
    output_dir: Path,
    *,
    frame_start_measure: int,
    measure_number: int,
    tesseract_path: str = "tesseract",
    recognition_mode: str = "ocr",
    vision_model_url: str | None = None,
) -> dict:
    """Re-run OCR for one measure and return an unsaved review proposal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_frame(frame)
    target_index = measure_number - frame_start_measure
    if target_index < 0 or target_index >= len(parsed.measures):
        raise ValueError("所选切片不包含这个小节，请重新选择更接近的源帧")
    _run_frame_ocr(
        parsed,
        output_dir / ".ocr",
        tesseract_path,
        vision_model_url if recognition_mode == "ai" else None,
    )
    _run_technique_ocr(parsed, output_dir / ".ocr", tesseract_path)
    parsed.start_measure = frame_start_measure
    parsed.start_measure_confidence = 100.0
    _cluster_and_label_glyphs([parsed])
    candidate = _candidate_from_geometry(parsed, target_index, measure_number)
    return {
        "number": measure_number,
        "quality": round(candidate.quality, 2),
        "source_time": round(candidate.source_time, 3),
        "events": [
            {
                "onset_eighths": event.onset,
                "duration_eighths": event.duration,
                "notes": [_tab_note_to_dict(note) for note in event.notes],
            }
            for event in candidate.events
        ],
        "time_signature": {"numerator": 4, "denominator": 4},
        "validation": _measure_validation(candidate.events, EIGHTH_UNITS_PER_MEASURE),
        "source_frame": frame.source_frame,
        "source_name": frame.path.name,
    }


def update_recognized_measure(
    score_path: Path,
    diagnostics_path: Path,
    *,
    title: str,
    measure_number: int,
    events: list[dict],
    time_signature: tuple[int, int] | None = None,
    human_verified: bool = False,
) -> dict:
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    summary = diagnostics.get("summary") or {}
    start_measure = int(summary.get("start_measure") or measure_number)
    end_measure = int(summary.get("end_measure") or measure_number)
    if not start_measure <= measure_number <= end_measure:
        raise ValueError(f"小节号必须在 {start_measure}～{end_measure} 之间")

    measure_rows = diagnostics.setdefault("measures", [])
    existing = next((row for row in measure_rows if int(row.get("number") or 0) == measure_number), None)
    raw_time_signature = (existing or {}).get("time_signature") or {}
    numerator = int(time_signature[0] if time_signature else raw_time_signature.get("numerator") or summary.get("time_signature_numerator") or 4)
    denominator = int(time_signature[1] if time_signature else raw_time_signature.get("denominator") or summary.get("time_signature_denominator") or 4)
    resolved_time_signature = (numerator, denominator)
    capacity = _measure_capacity_eighths(resolved_time_signature)

    def quantized_eighth(value: object, label: str, multiplier: int, grid_label: str) -> float:
        number = float(value)
        if not math.isfinite(number) or abs(number * multiplier - round(number * multiplier)) > 1e-9:
            raise ValueError(f"{label}必须落在{grid_label}网格上")
        return round(number * multiplier) / multiplier

    normalized_events: list[TabEvent] = []
    previous_end = 0.0
    for raw_event in sorted(
        events,
        key=lambda item: (float(item["onset_eighths"]), float(item["duration_eighths"])),
    ):
        onset = quantized_eighth(raw_event["onset_eighths"], "音符起点", 2, "十六分音符")
        duration = quantized_eighth(raw_event["duration_eighths"], "音符时值", 8, "六十四分音符")
        if onset < 0 or onset >= capacity:
            raise ValueError(f"节拍起点超出当前 {numerator}/{denominator} 小节范围")
        if duration < 0.5 or duration > capacity:
            raise ValueError("音符时值必须在十六分音符到全音符之间")
        if onset < previous_end:
            raise ValueError("音符事件不能互相重叠；同一时刻的音请放在同一个和弦事件中")
        if onset + duration > capacity:
            raise ValueError(f"节拍超出当前 {numerator}/{denominator} 小节范围")
        notes = tuple(
            sorted(
                (_tab_note_from_dict(note) for note in raw_event.get("notes", [])),
                key=lambda note: note.string,
            )
        )
        if notes and raw_event.get("rest"):
            raise ValueError("同一个节拍不能同时是音符与休止符")
        if len({note.string for note in notes}) != len(notes):
            raise ValueError("同一个和弦事件中每根弦只能出现一次")
        normalized_events.append(TabEvent(onset, duration, notes))
        previous_end = onset + duration

    source_time = float(existing.get("source_time") or 0) if existing else 0.0
    quality = float(existing.get("quality") or 100) if existing else 100.0
    baseline_events = (existing or {}).get("ocr_events")
    if baseline_events is None:
        baseline_events = (existing or {}).get("events", [])
    replacement = {
        "number": measure_number,
        "quality": round(quality, 2),
        "source_time": round(source_time, 3),
        "events": [
            {
                "onset_eighths": event.onset,
                "duration_eighths": event.duration,
                "notes": [_tab_note_to_dict(note) for note in event.notes],
                **({"rest": True} if not event.notes else {}),
            }
            for event in normalized_events
        ],
        "ocr_events": baseline_events,
        "human_verified": human_verified,
        "time_signature": {"numerator": numerator, "denominator": denominator},
        "validation": _measure_validation(normalized_events, capacity),
    }
    if existing:
        measure_rows[measure_rows.index(existing)] = replacement
    else:
        measure_rows.append(replacement)
    measure_rows.sort(key=lambda row: int(row["number"]))

    candidates: dict[int, MeasureCandidate] = {}
    for row in measure_rows:
        number = int(row["number"])
        row_events = tuple(
            TabEvent(
                float(event["onset_eighths"]),
                float(event["duration_eighths"]),
                tuple(_tab_note_from_dict(note) for note in event.get("notes", [])),
            )
            for event in row.get("events", [])
        )
        signature = tuple(
            (event.onset, note.string, note.fret)
            for event in row_events
            for note in event.notes
        )
        candidates[number] = MeasureCandidate(
            number,
            row_events,
            float(row.get("quality") or 0),
            float(row.get("source_time") or 0),
            signature,
            time_signature=(
                int((row.get("time_signature") or {}).get("numerator") or summary.get("time_signature_numerator") or 4),
                int((row.get("time_signature") or {}).get("denominator") or summary.get("time_signature_denominator") or 4),
            ),
        )

    missing = [number for number in range(start_measure, end_measure + 1) if number not in candidates]
    summary["measure_count"] = len(candidates)
    summary["missing_measures"] = missing
    invalid_measures = [
        number
        for number, candidate in sorted(candidates.items())
        if not _measure_validation(
            candidate.events,
            _measure_capacity_eighths(candidate.time_signature),
        )["is_complete"]
    ]
    summary["invalid_measures"] = invalid_measures

    verified_rows = [row for row in measure_rows if row.get("human_verified")]
    verified_notes = 0
    correct_notes = 0
    for row in verified_rows:
        original = {
            (round(float(event.get("onset_eighths") or 0), 3), int(note.get("string") or 0)): int(note.get("fret") or 0)
            for event in row.get("ocr_events", [])
            for note in event.get("notes", [])
        }
        corrected = {
            (round(float(event.get("onset_eighths") or 0), 3), int(note.get("string") or 0)): int(note.get("fret") or 0)
            for event in row.get("events", [])
            for note in event.get("notes", [])
        }
        note_positions = set(original) | set(corrected)
        verified_notes += len(note_positions)
        correct_notes += sum(
            position in original
            and position in corrected
            and original[position] == corrected[position]
            for position in note_positions
        )
    summary["verified_measure_count"] = len(verified_rows)
    summary["verified_note_count"] = verified_notes
    summary["correct_note_count"] = correct_notes
    summary["human_verified_accuracy"] = (
        round(correct_notes / verified_notes, 4) if verified_notes else None
    )
    warnings = [
        warning
        for warning in summary.get("warnings", [])
        if "个小节未获得可靠结果" not in str(warning)
        and "个小节尚未用音符或实体休止填满" not in str(warning)
    ]
    if missing:
        warnings.append(f"有 {len(missing)} 个小节未获得可靠结果，已在导出谱中留空")
    if invalid_measures:
        warnings.append(
            f"有 {len(invalid_measures)} 个小节尚未用音符或实体休止填满拍号容量，校对器会标红"
        )
    summary["warnings"] = warnings
    diagnostics["summary"] = summary

    temporary_score = score_path.with_name(f".{score_path.name}.tmp")
    temporary_diagnostics = diagnostics_path.with_name(f".{diagnostics_path.name}.tmp")
    _build_musicxml(
        title,
        candidates,
        temporary_score,
        float(summary.get("estimated_tempo_bpm") or 120),
    )
    temporary_diagnostics.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_score.replace(score_path)
    temporary_diagnostics.replace(diagnostics_path)
    return diagnostics


def append_blank_tab_measure(
    score_path: Path,
    diagnostics_path: Path,
    *,
    title: str,
    after_measure: int | None = None,
) -> dict:
    """Insert one empty measure and atomically rebuild the editable MusicXML score."""
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    summary = diagnostics.get("summary") or {}
    start_measure = int(summary.get("start_measure") or 1)
    end_measure = int(summary.get("end_measure") or start_measure)
    if end_measure - start_measure + 1 >= 128:
        raise ValueError("当前最多支持 128 小节")

    measure_rows = diagnostics.setdefault("measures", [])
    insert_after = end_measure if after_measure is None else int(after_measure)
    if insert_after < start_measure or insert_after > end_measure:
        raise ValueError(f"插入位置必须在第 {start_measure}～{end_measure} 小节")
    next_number = insert_after + 1
    for row in measure_rows:
        if int(row["number"]) > insert_after:
            row["number"] = int(row["number"]) + 1
    for frame in diagnostics.get("frames", []):
        if frame.get("start_measure") is not None and int(frame["start_measure"]) > insert_after:
            frame["start_measure"] = int(frame["start_measure"]) + 1
    measure_rows.append(
        {
            "number": next_number,
            "quality": 100.0,
            "source_time": 0.0,
            "events": [],
            "time_signature": {
                "numerator": int(summary.get("time_signature_numerator") or 4),
                "denominator": int(summary.get("time_signature_denominator") or 4),
            },
            "validation": _measure_validation(
                [],
                _measure_capacity_eighths((
                    int(summary.get("time_signature_numerator") or 4),
                    int(summary.get("time_signature_denominator") or 4),
                )),
            ),
        }
    )
    measure_rows.sort(key=lambda row: int(row["number"]))
    summary["start_measure"] = start_measure
    summary["end_measure"] = end_measure + 1
    summary["measure_count"] = len(measure_rows)
    shifted_invalid = {
        int(value) + 1 if int(value) > insert_after else int(value)
        for value in summary.get("invalid_measures", [])
    }
    summary["invalid_measures"] = sorted({*shifted_invalid, next_number})
    diagnostics["summary"] = summary

    candidates: dict[int, MeasureCandidate] = {}
    for row in measure_rows:
        number = int(row["number"])
        row_events = tuple(
            TabEvent(
                float(event["onset_eighths"]),
                float(event["duration_eighths"]),
                tuple(_tab_note_from_dict(note) for note in event.get("notes", [])),
            )
            for event in row.get("events", [])
        )
        signature = tuple(
            (event.onset, note.string, note.fret)
            for event in row_events
            for note in event.notes
        )
        candidates[number] = MeasureCandidate(
            number,
            row_events,
            float(row.get("quality") or 0),
            float(row.get("source_time") or 0),
            signature,
            time_signature=(
                int((row.get("time_signature") or {}).get("numerator") or summary.get("time_signature_numerator") or 4),
                int((row.get("time_signature") or {}).get("denominator") or summary.get("time_signature_denominator") or 4),
            ),
        )

    temporary_score = score_path.with_name(f".{score_path.name}.tmp")
    temporary_diagnostics = diagnostics_path.with_name(f".{diagnostics_path.name}.tmp")
    _build_musicxml(
        title,
        candidates,
        temporary_score,
        float(summary.get("estimated_tempo_bpm") or 120),
    )
    temporary_diagnostics.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_score.replace(score_path)
    temporary_diagnostics.replace(diagnostics_path)
    return diagnostics
