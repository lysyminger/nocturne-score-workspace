from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from backend.tab_recognition import (
    DigitGlyph,
    FrameInput,
    MeasureCandidate,
    MeasureGeometry,
    NoteToken,
    ParsedFrame,
    TabEvent,
    TabNote,
    _build_musicxml,
    _smooth_frame_starts,
    append_blank_tab_measure,
    assign_rhythm_units,
    create_blank_tab_score,
    detect_measure_boundaries,
    detect_score_layout,
    detect_staff_lines,
    update_recognized_measure,
)


def test_detects_six_line_staff_and_measure_bars():
    image = np.full((220, 800), 255, dtype=np.uint8)
    staff_lines = [50, 68, 86, 104, 122, 140]
    for y in staff_lines:
        cv2.line(image, (0, y), (799, y), 228, 1)
    for x in (10, 205, 400, 595, 790):
        cv2.line(image, (x, 50), (x, 140), 0, 2)

    detected_lines = detect_staff_lines(image)
    boundaries = detect_measure_boundaries(image, detected_lines)

    assert detected_lines == staff_lines
    assert np.allclose(boundaries, [10, 205, 400, 595, 790], atol=2)


def test_detects_light_tab_lines_on_dark_video_overlay():
    image = np.full((220, 800), 24, dtype=np.uint8)
    staff_lines = [50, 68, 86, 104, 122, 140]
    for y in staff_lines:
        cv2.line(image, (0, y), (799, y), 232, 1)
    for x in (10, 400, 790):
        cv2.line(image, (x, 50), (x, 140), 232, 2)

    layout = detect_score_layout(image)

    assert layout.staff_lines == staff_lines
    assert layout.layout == "tab_only"
    assert layout.polarity == "light_on_dark"


def test_detects_standard_notation_above_guitar_tab():
    image = np.full((260, 800), 255, dtype=np.uint8)
    notation_lines = [35, 45, 55, 65, 75]
    tab_lines = [120, 138, 156, 174, 192, 210]
    for y in [*notation_lines, *tab_lines]:
        cv2.line(image, (0, y), (799, y), 205, 1)
    for x in (10, 400, 790):
        cv2.line(image, (x, 35), (x, 210), 0, 2)

    layout = detect_score_layout(image)

    assert layout.staff_lines == tab_lines
    assert layout.notation_staff_lines == notation_lines
    assert layout.layout == "staff_tab_pair"
    assert layout.polarity == "dark_on_light"


def test_reads_beamed_quarter_and_dotted_quarter_rhythm():
    image = np.full((240, 320), 255, dtype=np.uint8)
    staff_lines = [40, 58, 76, 94, 112, 130]
    for y in staff_lines:
        cv2.line(image, (0, y), (319, y), 228, 1)

    # Two beamed eighth notes.
    cv2.line(image, (50, 135), (50, 175), 0, 2)
    cv2.line(image, (90, 135), (90, 175), 0, 2)
    cv2.line(image, (50, 175), (90, 175), 0, 3)
    # A quarter note.
    cv2.line(image, (150, 135), (150, 175), 0, 2)
    # A dotted quarter note.
    cv2.line(image, (220, 135), (220, 175), 0, 2)
    cv2.circle(image, (227, 174), 2, 0, -1)

    glyph = DigitGlyph(np.ones((14, 8), dtype=np.uint8), np.zeros((28, 20), dtype=np.uint8))
    tokens = [NoteToken(x=x, string=3, glyphs=[glyph]) for x in (50, 90, 150, 220)]
    assign_rhythm_units(image, staff_lines, tokens)

    assert [token.duration_units for token in tokens] == [1, 1, 2, 3]


def test_reads_double_beam_as_sixteenth_notes():
    image = np.full((240, 180), 255, dtype=np.uint8)
    staff_lines = [40, 58, 76, 94, 112, 130]
    for y in staff_lines:
        cv2.line(image, (0, y), (179, y), 228, 1)

    cv2.line(image, (50, 135), (50, 175), 0, 2)
    cv2.line(image, (90, 135), (90, 175), 0, 2)
    cv2.line(image, (50, 168), (90, 168), 0, 3)
    cv2.line(image, (50, 175), (90, 175), 0, 3)

    glyph = DigitGlyph(np.ones((14, 8), dtype=np.uint8), np.zeros((28, 20), dtype=np.uint8))
    tokens = [NoteToken(x=x, string=3, glyphs=[glyph]) for x in (50, 90)]
    assign_rhythm_units(image, staff_lines, tokens)

    assert [token.duration_units for token in tokens] == [0.5, 0.5]


def _frame(time_seconds: float, labels: list[str | None]) -> ParsedFrame:
    measures = [
        MeasureGeometry(0, 100, None, raw_label=label, raw_label_confidence=90)
        for label in labels
    ]
    return ParsedFrame(
        source=FrameInput(Path(f"frame-{time_seconds}.jpg"), time_seconds),
        gray=np.zeros((1, 1), dtype=np.uint8),
        staff_lines=[1, 2, 3, 4, 5, 6],
        measures=measures,
        tokens=[],
        highlighted_index=0,
    )


def test_measure_sequence_rejects_large_ocr_jump():
    frames = [
        _frame(0, ["10", "11", "12"]),
        _frame(2, ["12", "13", "14"]),
        _frame(4, ["999", "15", "16"]),
        _frame(6, ["16", "17", "18"]),
    ]

    _smooth_frame_starts(frames)

    starts = [frame.start_measure for frame in frames]
    assert starts == [10, 12, 14, 16]


def test_measure_sequence_does_not_invent_zero_without_any_ocr_anchor():
    frames = [_frame(0, [None, None]), _frame(2, [None, None])]

    _smooth_frame_starts(frames)

    assert [frame.start_measure for frame in frames] == [None, None]


def test_musicxml_contains_playable_tab_technical_data(tmp_path):
    candidate = MeasureCandidate(
        number=1,
        events=(
            TabEvent(0, 2, (TabNote(1, 7),)),
            TabEvent(2, 3, (TabNote(2, 8), TabNote(3, 9))),
        ),
        quality=100,
        source_time=0,
        signature=((0, 1, 7), (2, 2, 8), (2, 3, 9)),
    )
    output = tmp_path / "score.musicxml"

    _build_musicxml("测试 TAB", {1: candidate}, output, 180)

    root = ET.parse(output).getroot()
    assert root.findtext("./work/work-title") == "测试 TAB"
    assert root.findtext(".//clef/sign") == "TAB"
    assert float(root.find(".//sound").attrib["tempo"]) == 180
    assert [node.text for node in root.findall(".//technical/string")] == ["1", "2", "3"]
    assert [node.text for node in root.findall(".//technical/fret")] == ["7", "8", "9"]
    assert root.find(".//note/chord") is not None


def test_musicxml_splits_nonstandard_rest_lengths(tmp_path):
    candidate = MeasureCandidate(
        number=1,
        events=(TabEvent(5, 1, (TabNote(1, 3),)),),
        quality=100,
        source_time=0,
        signature=((5, 1, 3),),
    )
    output = tmp_path / "rests.musicxml"

    _build_musicxml("休止符测试", {1: candidate}, output, 120)

    root = ET.parse(output).getroot()
    rest_durations = [int(note.findtext("duration")) for note in root.findall(".//note") if note.find("rest") is not None]
    assert rest_durations == [32, 8, 16]


def test_musicxml_preserves_sixteenth_note_and_rest(tmp_path):
    candidate = MeasureCandidate(
        number=1,
        events=(TabEvent(0.5, 0.5, (TabNote(1, 7),)),),
        quality=100,
        source_time=0,
        signature=((0.5, 1, 7),),
    )
    output = tmp_path / "sixteenth.musicxml"

    _build_musicxml("十六分音符测试", {1: candidate}, output, 120)

    root = ET.parse(output).getroot()
    assert root.findtext(".//divisions") == "16"
    played_note = next(note for note in root.findall(".//note") if note.find("pitch") is not None)
    assert played_note.findtext("duration") == "4"
    assert played_note.findtext("type") == "16th"
    first_rest = next(note for note in root.findall(".//note") if note.find("rest") is not None)
    assert first_rest.findtext("duration") == "4"
    assert first_rest.findtext("type") == "16th"


def test_musicxml_preserves_double_dotted_sixteenth_and_fractional_rest(tmp_path):
    candidate = MeasureCandidate(
        number=1,
        events=(TabEvent(0, 0.875, (TabNote(1, 7),)),),
        quality=100,
        source_time=0,
        signature=((0, 1, 7),),
    )
    output = tmp_path / "double-dotted.musicxml"

    _build_musicxml("双附点测试", {1: candidate}, output, 120)

    root = ET.parse(output).getroot()
    played_note = next(note for note in root.findall(".//note") if note.find("pitch") is not None)
    assert root.findtext(".//divisions") == "16"
    assert played_note.findtext("duration") == "7"
    assert played_note.findtext("type") == "16th"
    assert len(played_note.findall("dot")) == 2
    rest_durations = [int(note.findtext("duration")) for note in root.findall(".//note") if note.find("rest") is not None]
    assert sum(rest_durations) == 57


def test_musicxml_writes_guitar_techniques_and_link_endpoints(tmp_path):
    candidate = MeasureCandidate(
        number=1,
        events=(
            TabEvent(0, 1, (TabNote(1, 5, "hammer_on"),)),
            TabEvent(1, 1, (TabNote(1, 7, "bend"),)),
            TabEvent(
                2,
                2,
                (
                    TabNote(2, 5, "let_ring"),
                    TabNote(3, 12, "harmonic"),
                    TabNote(4, 0, "dead_note"),
                ),
            ),
        ),
        quality=100,
        source_time=0,
        signature=((0, 1, 5), (1, 1, 7), (2, 2, 5), (2, 3, 12), (2, 4, 0)),
    )
    output = tmp_path / "techniques.musicxml"

    _build_musicxml("技巧测试", {1: candidate}, output, 120)

    root = ET.parse(output).getroot()
    assert [node.attrib["type"] for node in root.findall(".//hammer-on")] == ["start", "stop"]
    assert [node.attrib["type"] for node in root.findall(".//slur")] == ["start", "stop"]
    assert root.findtext(".//bend/bend-alter") == "2"
    assert root.find(".//harmonic/natural") is not None
    assert root.find(".//tied[@type='let-ring']") is not None
    assert root.findtext(".//notehead") == "x"


def test_creates_and_extends_playable_blank_tab_score(tmp_path):
    result = create_blank_tab_score(
        tmp_path / "manual",
        title="空白练习",
        measure_count=3,
        tempo_bpm=96,
    )

    diagnostics = append_blank_tab_measure(
        result.score_path,
        result.diagnostics_path,
        title="空白练习",
    )

    root = ET.parse(result.score_path).getroot()
    assert result.summary["engine"] == "tab_manual_editor"
    assert diagnostics["summary"]["end_measure"] == 4
    assert diagnostics["summary"]["measure_count"] == 4
    assert len(root.findall(".//part/measure")) == 4
    assert len(root.findall(".//rest")) == 4
    assert root.findtext(".//software") == "Nocturne TAB Editor"


def test_updates_recognized_measure_and_rebuilds_musicxml(tmp_path):
    score_path = tmp_path / "recognized-tab.musicxml"
    diagnostics_path = tmp_path / "recognition.json"
    original = MeasureCandidate(
        number=1,
        events=(TabEvent(0, 2, (TabNote(1, 3),)),),
        quality=90,
        source_time=2.0,
        signature=((0, 1, 3),),
    )
    _build_musicxml("编辑测试", {1: original}, score_path, 120)
    diagnostics_path.write_text(
        """{
          "summary": {"start_measure": 1, "end_measure": 1, "estimated_tempo_bpm": 120},
          "frames": [],
          "measures": [{
            "number": 1,
            "quality": 90,
            "source_time": 2.0,
            "events": [{"onset_eighths": 0, "duration_eighths": 2, "notes": [{"string": 1, "fret": 3}]}]
          }]
        }""",
        encoding="utf-8",
    )

    result = update_recognized_measure(
        score_path,
        diagnostics_path,
        title="编辑测试",
        measure_number=1,
        events=[
            {
                "onset_eighths": 0,
                "duration_eighths": 2,
                "notes": [
                    {"string": 2, "fret": 7, "technique": "bend"},
                    {"string": 3, "fret": 9},
                ],
            }
        ],
    )

    root = ET.parse(score_path).getroot()
    assert [node.text for node in root.findall(".//technical/string")] == ["2", "3"]
    assert [node.text for node in root.findall(".//technical/fret")] == ["7", "9"]
    assert result["measures"][0]["events"][0]["notes"][0] == {
        "string": 2,
        "fret": 7,
        "technique": "bend",
    }
    assert root.findtext(".//bend/bend-alter") == "2"


def test_updates_recognized_measure_with_double_dotted_duration(tmp_path):
    score_path = tmp_path / "recognized-tab.musicxml"
    diagnostics_path = tmp_path / "recognition.json"
    original = MeasureCandidate(
        number=1,
        events=(TabEvent(0, 1, (TabNote(1, 3),)),),
        quality=90,
        source_time=0,
        signature=((0, 1, 3),),
    )
    _build_musicxml("双附点编辑", {1: original}, score_path, 120)
    diagnostics_path.write_text(
        """{
          "summary": {"start_measure": 1, "end_measure": 1, "estimated_tempo_bpm": 120},
          "frames": [],
          "measures": [{
            "number": 1,
            "quality": 90,
            "source_time": 0,
            "events": [{"onset_eighths": 0, "duration_eighths": 1, "notes": [{"string": 1, "fret": 3}]}]
          }]
        }""",
        encoding="utf-8",
    )

    result = update_recognized_measure(
        score_path,
        diagnostics_path,
        title="双附点编辑",
        measure_number=1,
        events=[{"onset_eighths": 0, "duration_eighths": 0.875, "notes": [{"string": 1, "fret": 3}]}],
    )

    root = ET.parse(score_path).getroot()
    played_note = next(note for note in root.findall(".//note") if note.find("pitch") is not None)
    assert result["measures"][0]["events"][0]["duration_eighths"] == 0.875
    assert played_note.findtext("duration") == "7"
    assert len(played_note.findall("dot")) == 2
