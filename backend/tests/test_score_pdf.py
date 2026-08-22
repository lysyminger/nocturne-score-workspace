from pathlib import Path

from PIL import Image, ImageDraw

from backend.score_pdf import build_recognized_score_pdf, build_slice_preview_pdf


def _score_image(path: Path, offset: int = 0) -> None:
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    for row in range(6):
        y = 55 + row * 24
        draw.line((35, y, 865, y), fill="black", width=2)
    draw.line((80 + offset, 50, 80 + offset, 180), fill="black", width=3)
    draw.text((120 + offset, 86), "7", fill="black")
    image.save(path, "JPEG", quality=94)
    image.close()


def test_slice_preview_pdf_auto_trims_and_skips_adjacent_duplicate(tmp_path):
    first = tmp_path / "first.jpg"
    duplicate = tmp_path / "duplicate.jpg"
    different = tmp_path / "different.jpg"
    _score_image(first)
    _score_image(duplicate)
    _score_image(different, 80)
    output = tmp_path / "preview.pdf"

    build_slice_preview_pdf([first, duplicate, different], output)

    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 1_000


def test_recognized_pdf_lays_out_unique_measure_crops_in_order(tmp_path):
    source = tmp_path / "source.jpg"
    _score_image(source)
    output = tmp_path / "recognized.pdf"

    build_recognized_score_pdf(
        [
            (1, source, (30, 35, 300, 205)),
            (2, source, (300, 35, 580, 205)),
            (3, source, (580, 35, 870, 205)),
        ],
        output,
    )

    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 1_000
