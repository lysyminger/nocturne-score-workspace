from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PAGE_WIDTH = 1_654
PAGE_HEIGHT = 2_339
PAGE_MARGIN = 72
PAGE_GAP = 20


def _content_crop(image: Image.Image, padding: int = 12) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    foreground = np.argwhere(gray < 247)
    if foreground.size == 0:
        return image.copy()
    top, left = foreground.min(axis=0)
    bottom, right = foreground.max(axis=0)
    crop = (
        max(0, int(left) - padding),
        max(0, int(top) - padding),
        min(image.width, int(right) + padding + 1),
        min(image.height, int(bottom) + padding + 1),
    )
    return image.crop(crop)


def _load_image(path: Path, crop_box: tuple[int, int, int, int] | None = None) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if crop_box:
        left, top, right, bottom = crop_box
        left = min(image.width - 1, max(0, int(left)))
        top = min(image.height - 1, max(0, int(top)))
        right = min(image.width, max(left + 1, int(right)))
        bottom = min(image.height, max(top + 1, int(bottom)))
        image = image.crop((left, top, right, bottom))
    return _content_crop(image)


def _fingerprint(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)


def _save_pages(pages: list[Image.Image], output_path: Path) -> Path:
    if not pages:
        raise ValueError("没有可写入 PDF 的谱面图像")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = pages
    first.save(output_path, "PDF", save_all=True, append_images=rest, resolution=150)
    for page in pages:
        page.close()
    return output_path


def build_slice_preview_pdf(frame_paths: list[Path], output_path: Path) -> Path:
    """Build a readable, auto-trimmed preview without claiming musical deduplication."""
    strips: list[Image.Image] = []
    previous_fingerprint: np.ndarray | None = None
    for path in frame_paths:
        image = _load_image(path)
        fingerprint = _fingerprint(image)
        if previous_fingerprint is not None:
            difference = float(np.mean(np.abs(fingerprint - previous_fingerprint)))
            if difference < 1.5:
                image.close()
                continue
        strips.append(image)
        previous_fingerprint = fingerprint
    if not strips:
        raise ValueError("切片中没有可用于 PDF 的画面")

    inner_width = PAGE_WIDTH - PAGE_MARGIN * 2
    pages: list[Image.Image] = []
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    y = PAGE_MARGIN
    for strip in strips:
        scale = min(inner_width / strip.width, (PAGE_HEIGHT - PAGE_MARGIN * 2) / strip.height)
        size = (max(1, round(strip.width * scale)), max(1, round(strip.height * scale)))
        rendered = strip.resize(size, Image.Resampling.LANCZOS)
        if y > PAGE_MARGIN and y + rendered.height > PAGE_HEIGHT - PAGE_MARGIN:
            pages.append(page)
            page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
            y = PAGE_MARGIN
        x = PAGE_MARGIN + (inner_width - rendered.width) // 2
        page.paste(rendered, (x, y))
        y += rendered.height + PAGE_GAP
        rendered.close()
        strip.close()
    pages.append(page)
    return _save_pages(pages, output_path)


def build_recognized_score_pdf(
    measure_sources: list[tuple[int, Path, tuple[int, int, int, int]]],
    output_path: Path,
) -> Path:
    """Lay out one best source crop per recognized measure, in score order."""
    if not measure_sources:
        raise ValueError("没有带来源区域的已识别小节")
    columns = 4
    rows = 6
    inner_width = PAGE_WIDTH - PAGE_MARGIN * 2
    inner_height = PAGE_HEIGHT - PAGE_MARGIN * 2
    cell_width = (inner_width - PAGE_GAP * (columns - 1)) // columns
    cell_height = (inner_height - PAGE_GAP * (rows - 1)) // rows
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    pages: list[Image.Image] = []
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(page)
    for index, (measure_number, path, crop_box) in enumerate(measure_sources):
        slot = index % (columns * rows)
        if index and slot == 0:
            pages.append(page)
            page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
            draw = ImageDraw.Draw(page)
        row, column = divmod(slot, columns)
        x = PAGE_MARGIN + column * (cell_width + PAGE_GAP)
        y = PAGE_MARGIN + row * (cell_height + PAGE_GAP)
        draw.rounded_rectangle(
            (x, y, x + cell_width, y + cell_height),
            radius=8,
            fill="white",
            outline=(214, 214, 218),
            width=2,
        )
        draw.text((x + 10, y + 7), f"M{measure_number}", font=font, fill=(70, 70, 76))

        image = _load_image(path, crop_box)
        available_width = cell_width - 16
        available_height = cell_height - 42
        scale = min(available_width / image.width, available_height / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        rendered = image.resize(size, Image.Resampling.LANCZOS)
        image_x = x + (cell_width - rendered.width) // 2
        image_y = y + 34 + (available_height - rendered.height) // 2
        page.paste(rendered, (image_x, image_y))
        rendered.close()
        image.close()
    pages.append(page)
    return _save_pages(pages, output_path)
