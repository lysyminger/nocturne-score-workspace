from pathlib import Path

from PIL import Image, ImageDraw

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

