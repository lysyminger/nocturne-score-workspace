from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

from backend.tab_recognition import detect_score_layout


MAX_PDF_PAGES = 80
MAX_RENDERED_PAGE_PIXELS = 24_000_000
PDF_RENDER_SCALE = 2.0


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


@dataclass(frozen=True)
class PdfScoreImport:
    pages: tuple[PdfPage, ...]
    systems: tuple[PdfScoreSystem, ...]
    layout_counts: dict[str, int]


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

        for page_index, page in enumerate(document):
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
            for page_system_index, (top, bottom) in enumerate(_system_crops(gray), start=1):
                crop = color[top:bottom]
                if crop.size == 0:
                    continue
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ink_columns = np.flatnonzero(np.count_nonzero(crop_gray < 245, axis=0) >= 2)
                if ink_columns.size:
                    left = max(0, int(ink_columns[0]) - 24)
                    right = min(crop.shape[1], int(ink_columns[-1]) + 25)
                    crop = crop[:, left:right]
                try:
                    layout = detect_score_layout(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                except ValueError:
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
                    )
                )
                layout_counts[layout.layout] = layout_counts.get(layout.layout, 0) + 1
        return PdfScoreImport(tuple(pages), tuple(systems), layout_counts)
    finally:
        document.close()
