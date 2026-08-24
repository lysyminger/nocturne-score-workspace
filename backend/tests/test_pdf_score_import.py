from pathlib import Path

from PIL import Image, ImageDraw
import fitz

from backend.pdf_score_import import render_and_segment_pdf


def test_pdf_is_rendered_split_into_tab_systems_and_classifies_staff_pair(tmp_path: Path):
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    for y in (250, 270, 290, 310, 330, 350):
        draw.line((50, y, 1150, y), fill="black", width=2)
    for y in (700, 720, 740, 760, 780):
        draw.line((50, y, 1150, y), fill="black", width=2)
    for y in (850, 870, 890, 910, 930, 950):
        draw.line((50, y, 1150, y), fill="black", width=2)
    for top, bottom in ((250, 350), (700, 950)):
        for x in (50, 400, 800, 1150):
            draw.line((x, top, x, bottom), fill="black", width=3)
    pdf_path = tmp_path / "paired-score.pdf"
    image.save(pdf_path, "PDF", resolution=144)
    image.close()

    result = render_and_segment_pdf(pdf_path, tmp_path / "rendered")

    assert len(result.pages) == 1
    assert len(result.systems) == 2
    assert result.layout_counts == {"tab_only": 1, "staff_tab_pair": 1}
    assert all(system.path.is_file() for system in result.systems)


def test_vector_pdf_keeps_exact_fret_and_measure_coordinates(tmp_path: Path):
    pdf_path = tmp_path / "vector-tab.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=300)
    for y in (100, 110, 120, 130, 140, 150):
        page.draw_line((50, y), (550, y), width=0.7)
    for x in (50, 300, 550):
        page.draw_line((x, 100), (x, 150), width=1.0)
    page.insert_text((70, 94), "1", fontsize=10)
    page.insert_text((320, 94), "2", fontsize=10)
    page.insert_text((120, 104), "7", fontsize=10)
    page.insert_text((180, 124), "12", fontsize=10)
    page.insert_text((380, 144), "9", fontsize=10)
    page.insert_text((80, 50), "Moderate = 180", fontsize=12)
    document.save(pdf_path)
    document.close()

    result = render_and_segment_pdf(pdf_path, tmp_path / "vector-rendered")

    assert len(result.systems) == 1
    assert result.tempo_bpm == 180
    assert [(token.text, token.string) for token in result.systems[0].exact_tokens] == [
        ("7", 1),
        ("12", 3),
        ("9", 5),
    ]
    assert [label.text for label in result.systems[0].exact_measure_labels] == ["1", "2"]
