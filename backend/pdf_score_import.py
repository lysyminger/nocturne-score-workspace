from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import fitz
import numpy as np

from backend.tab_recognition import ExactMeasureLabel, ExactRestToken, ExactTabToken, detect_score_layout


MAX_PDF_PAGES = 80
MAX_RENDERED_PAGE_PIXELS = 24_000_000
PDF_RENDER_SCALE = 2.0
VECTOR_TAB_TOKEN = re.compile(r"^[<(\[]?(X|x|\d{1,2})[>)\]]?$")
VECTOR_REST_DURATIONS = {
    "\ue4e3": 8.0,
    "\ue4e4": 4.0,
    "\ue4e5": 2.0,
    "\ue4e6": 1.0,
    "\ue4e7": 0.5,
    "\ue4e8": 0.25,
}


@dataclass(frozen=True)
class PdfPage:
    path: Path
    page_number: int
    width: int
    height: int


@dataclass(frozen=True)
class PdfScoreSystem:
    path: Path
    page_number: int
    system_number: int
    layout: str
    polarity: str
    exact_tokens: tuple[ExactTabToken, ...] = ()
    exact_measure_labels: tuple[ExactMeasureLabel, ...] = ()
    exact_rests: tuple[ExactRestToken, ...] = ()


@dataclass(frozen=True)
class PdfScoreImport:
    pages: tuple[PdfPage, ...]
    systems: tuple[PdfScoreSystem, ...]
    layout_counts: dict[str, int]
    tempo_bpm: float | None = None


def _group_runs(indices: np.ndarray) -> list[list[int]]:
    if indices.size == 0:
        return []
    groups = [[int(indices[0])]]
    for raw in indices[1:]:
        value = int(raw)
        if value <= groups[-1][-1] + 2:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _horizontal_line_centers(gray: np.ndarray) -> list[int]:
    height, width = gray.shape
    block_size = max(15, (round(min(height, width) / 70) // 2) * 2 + 1)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(48, width // 8), 1)),
    )
    coverage = np.count_nonzero(horizontal, axis=1) / max(width, 1)
    direct_coverage = np.count_nonzero(gray < 235, axis=1) / max(width, 1)
    combined = np.maximum(coverage, direct_coverage)
    runs = _group_runs(np.flatnonzero((coverage >= 0.28) | (direct_coverage >= 0.25)))
    return [round(float(np.average(run, weights=combined[run]))) for run in runs]


def _tab_line_groups(centers: list[int]) -> list[list[int]]:
    candidates: list[tuple[float, list[int]]] = []
    for index in range(max(0, len(centers) - 5)):
        group = centers[index : index + 6]
        if len(group) != 6:
            continue
        gaps = np.diff(group)
        median_gap = float(np.median(gaps))
        if median_gap < 4 or median_gap > 80:
            continue
        irregularity = float(np.mean(np.abs(gaps - median_gap)) / median_gap)
        if irregularity <= 0.2:
            candidates.append((irregularity, group))

    selected: list[list[int]] = []
    for _irregularity, group in sorted(candidates, key=lambda item: (item[1][0], item[0])):
        if any(not (group[-1] < current[0] or group[0] > current[-1]) for current in selected):
            continue
        selected.append(group)

    recovery: list[tuple[float, list[int]]] = []
    for first_index, first in enumerate(centers):
        for second_index in range(first_index + 1, min(len(centers), first_index + 5)):
            second = centers[second_index]
            gap = second - first
            if gap < 4 or gap > 80:
                continue
            group = [first, second]
            cursor = second_index
            for offset in range(2, 6):
                expected = first + gap * offset
                options = [
                    (index, value)
                    for index, value in enumerate(centers[cursor + 1 :], start=cursor + 1)
                    if abs(value - expected) <= max(3.0, gap * 0.25)
                ]
                if not options:
                    break
                cursor, value = min(options, key=lambda item: abs(item[1] - expected))
                group.append(value)
            if len(group) != 6:
                continue
            gaps = np.diff(group)
            median_gap = float(np.median(gaps))
            irregularity = float(np.mean(np.abs(gaps - median_gap)) / median_gap)
            if irregularity <= 0.2:
                recovery.append((irregularity, group))
    for _irregularity, group in sorted(recovery, key=lambda item: (item[0], item[1][0])):
        if any(not (group[-1] < current[0] or group[0] > current[-1]) for current in selected):
            continue
        selected.append(group)
    return sorted(selected, key=lambda group: group[0])


def _system_crops(gray: np.ndarray) -> list[tuple[int, int]]:
    groups = _tab_line_groups(_horizontal_line_centers(gray))
    crops: list[tuple[int, int]] = []
    for index, group in enumerate(groups):
        spacing = float(np.median(np.diff(group)))
        previous_bottom = groups[index - 1][-1] if index else 0
        next_top = groups[index + 1][0] if index + 1 < len(groups) else gray.shape[0]
        top = max(0, round(group[0] - spacing * 15))
        bottom = min(gray.shape[0], round(group[-1] + spacing * 5))
        if index:
            top = max(top, round((previous_bottom + group[0]) / 2))
        if index + 1 < len(groups):
            bottom = min(bottom, round((group[-1] + next_top) / 2))
        if bottom - top >= spacing * 7:
            crops.append((top, bottom))
    return crops


def _covered_length(segments: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted((min(left, right), max(left, right)) for left, right in segments):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _vector_system_crops(
    page: fitz.Page, page_height: int
) -> list[tuple[int, int, list[float]]]:
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
    rows = sorted(
        y
        for y, segments in by_y.items()
        if _covered_length(segments) >= 70
    )
    groups: list[list[float]] = []
    used: set[int] = set()
    for index in range(len(rows) - 5):
        if any(position in used for position in range(index, index + 6)):
            continue
        group = rows[index:index + 6]
        gaps = np.diff(group)
        spacing = float(np.median(gaps))
        if not 4 <= spacing <= 12 or max(abs(float(value) - spacing) for value in gaps) > 0.8:
            continue
        groups.append(group)
        used.update(range(index, index + 6))
    crops: list[tuple[int, int, list[float]]] = []
    for index, group in enumerate(groups):
        spacing = float(np.median(np.diff(group))) * PDF_RENDER_SCALE
        first = group[0] * PDF_RENDER_SCALE
        last = group[-1] * PDF_RENDER_SCALE
        top = max(0, round(first - spacing * 15))
        bottom = min(page_height, round(last + spacing * 5))
        if index:
            previous_last = groups[index - 1][-1] * PDF_RENDER_SCALE
            top = max(top, round((previous_last + first) / 2))
        if index + 1 < len(groups):
            next_first = groups[index + 1][0] * PDF_RENDER_SCALE
            bottom = min(bottom, round((last + next_first) / 2))
        if bottom - top >= spacing * 7:
            crops.append(
                (
                    top,
                    bottom,
                    [value * PDF_RENDER_SCALE - top for value in group],
                )
            )
    return crops


def _vector_tab_tokens(
    words: list[tuple],
    staff_lines: list[int],
    crop_top: int,
    crop_left: int,
    crop_width: int,
) -> tuple[ExactTabToken, ...]:
    spacing = float(np.median(np.diff(staff_lines)))
    tokens: list[ExactTabToken] = []
    for word in words:
        x1, y1, x2, y2, raw_text = word[:5]
        text = str(raw_text).strip().replace(" ", "")
        match = VECTOR_TAB_TOKEN.match(text)
        if not match:
            continue
        value = match.group(1).upper()
        if value != "X" and int(value) > 36:
            continue
        left = round(x1 * PDF_RENDER_SCALE) - crop_left
        right = round(x2 * PDF_RENDER_SCALE) - crop_left
        top = round(y1 * PDF_RENDER_SCALE) - crop_top
        bottom = round(y2 * PDF_RENDER_SCALE) - crop_top
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        if right <= 0 or left >= crop_width:
            continue
        string_index = min(range(6), key=lambda index: abs(staff_lines[index] - center_y))
        if abs(staff_lines[string_index] - center_y) > max(1.8, spacing * 0.30):
            continue
        tokens.append(
            ExactTabToken(
                x=center_x,
                string=string_index + 1,
                text=value,
                box=(left, top, max(1, right - left), max(1, bottom - top)),
            )
        )
    return tuple(sorted(tokens, key=lambda token: (token.x, token.string)))


def _vector_measure_labels(
    words: list[tuple],
    staff_lines: list[float],
    crop_top: int,
    crop_left: int,
    crop_width: int,
) -> tuple[ExactMeasureLabel, ...]:
    spacing = float(np.median(np.diff(staff_lines)))
    labels: list[ExactMeasureLabel] = []
    for word in words:
        x1, y1, x2, y2, raw_text = word[:5]
        text = str(raw_text).strip()
        if not text.isdigit() or int(text) > 1000:
            continue
        center_x = (x1 + x2) * PDF_RENDER_SCALE / 2 - crop_left
        center_y = (y1 + y2) * PDF_RENDER_SCALE / 2 - crop_top
        if not 0 <= center_x < crop_width:
            continue
        if staff_lines[0] - spacing * 2 <= center_y <= staff_lines[0] - spacing * 0.15:
            labels.append(ExactMeasureLabel(x=center_x, text=text))
    return tuple(sorted(labels, key=lambda label: label.x))


def _vector_rests(
    words: list[tuple],
    staff_lines: list[float],
    crop_top: int,
    crop_left: int,
    crop_width: int,
) -> tuple[ExactRestToken, ...]:
    spacing = float(np.median(np.diff(staff_lines)))
    rests: list[ExactRestToken] = []
    for word in words:
        x1, y1, x2, y2, raw_text = word[:5]
        text = str(raw_text).strip()
        duration = VECTOR_REST_DURATIONS.get(text)
        if duration is None:
            continue
        center_x = (x1 + x2) * PDF_RENDER_SCALE / 2 - crop_left
        center_y = (y1 + y2) * PDF_RENDER_SCALE / 2 - crop_top
        if not 0 <= center_x < crop_width:
            continue
        if staff_lines[0] - spacing * 2 <= center_y <= staff_lines[-1] + spacing * 2:
            rests.append(ExactRestToken(x=center_x, duration_units=duration))
    return tuple(sorted(rests, key=lambda rest: rest.x))


def render_and_segment_pdf(pdf_path: Path, output_dir: Path) -> PdfScoreImport:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError("PDF 文件损坏或无法解析") from exc
    try:
        if document.needs_pass:
            raise ValueError("暂不支持加密 PDF")
        if document.page_count < 1:
            raise ValueError("PDF 没有可识别页面")
        if document.page_count > MAX_PDF_PAGES:
            raise ValueError(f"PDF 最多支持 {MAX_PDF_PAGES} 页")

        page_dir = output_dir / "pages"
        system_dir = output_dir / "systems"
        page_dir.mkdir(parents=True, exist_ok=True)
        system_dir.mkdir(parents=True, exist_ok=True)
        pages: list[PdfPage] = []
        systems: list[PdfScoreSystem] = []
        layout_counts: dict[str, int] = {}
        system_number = 0
        tempo_candidates: list[float] = []

        for page_index, page in enumerate(document):
            words = page.get_text("words")
            for match in re.finditer(r"(?:BPM\s*[:=]?|[=＝])\s*(\d{2,3}(?:\.\d+)?)", page.get_text(), re.IGNORECASE):
                value = float(match.group(1))
                if 30 <= value <= 300:
                    tempo_candidates.append(value)
            matrix = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
            width = round(page.rect.width * PDF_RENDER_SCALE)
            height = round(page.rect.height * PDF_RENDER_SCALE)
            if width * height > MAX_RENDERED_PAGE_PIXELS:
                raise ValueError(f"第 {page_index + 1} 页渲染尺寸过大")
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            page_path = page_dir / f"page-{page_index + 1:03d}.jpg"
            pixmap.save(page_path, jpg_quality=94)
            pages.append(PdfPage(page_path, page_index + 1, pixmap.width, pixmap.height))

            color = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
            if color is None:
                raise ValueError(f"第 {page_index + 1} 页无法转成图像")
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            vector_crops = _vector_system_crops(page, gray.shape[0])
            system_crops = vector_crops or [
                (top, bottom, None) for top, bottom in _system_crops(gray)
            ]
            for page_system_index, (top, bottom, vector_staff_lines) in enumerate(system_crops, start=1):
                crop = color[top:bottom]
                if crop.size == 0:
                    continue
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ink_columns = np.flatnonzero(np.count_nonzero(crop_gray < 245, axis=0) >= 2)
                crop_left = 0
                if ink_columns.size:
                    crop_left = max(0, int(ink_columns[0]) - 24)
                    right = min(crop.shape[1], int(ink_columns[-1]) + 25)
                    crop = crop[:, crop_left:right]
                try:
                    layout = detect_score_layout(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                except ValueError:
                    continue
                exact_tokens = _vector_tab_tokens(
                    words,
                    vector_staff_lines or layout.staff_lines,
                    top,
                    crop_left,
                    crop.shape[1],
                )
                exact_measure_labels = _vector_measure_labels(
                    words,
                    vector_staff_lines or layout.staff_lines,
                    top,
                    crop_left,
                    crop.shape[1],
                )
                exact_rests = _vector_rests(
                    words,
                    vector_staff_lines or layout.staff_lines,
                    top,
                    crop_left,
                    crop.shape[1],
                )
                if vector_crops and not exact_tokens:
                    continue
                system_number += 1
                system_path = system_dir / f"system-{system_number:04d}.jpg"
                cv2.imwrite(str(system_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
                systems.append(
                    PdfScoreSystem(
                        system_path,
                        page_index + 1,
                        page_system_index,
                        layout.layout,
                        layout.polarity,
                        exact_tokens,
                        exact_measure_labels,
                        exact_rests,
                    )
                )
                layout_counts[layout.layout] = layout_counts.get(layout.layout, 0) + 1
        tempo_bpm = tempo_candidates[0] if tempo_candidates else None
        return PdfScoreImport(tuple(pages), tuple(systems), layout_counts, tempo_bpm)
    finally:
        document.close()
