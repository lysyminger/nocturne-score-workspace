from __future__ import annotations

import numpy as np

from backend.audio_analysis import ANALYSIS_SAMPLE_RATE, analyze_samples


def click_track(duration: float = 24.0, bpm: float = 120.0) -> np.ndarray:
    sample_rate = ANALYSIS_SAMPLE_RATE
    samples = np.zeros(round(duration * sample_rate), dtype=np.float32)
    period = 60 / bpm
    click_length = round(sample_rate * 0.025)
    click = np.hanning(click_length * 2)[:click_length].astype(np.float32)
    for beat in np.arange(0.25, duration, period):
        start = round(beat * sample_rate)
        samples[start : start + click_length] += click
    time = np.arange(samples.size, dtype=np.float32) / sample_rate
    samples += np.where(time < duration / 2, 0.04 * np.sin(2 * np.pi * 220 * time), 0.04 * np.sin(2 * np.pi * 330 * time))
    return samples


def test_detects_tempo_sections_and_audio_grid_alignment():
    result = analyze_samples(
        click_track(),
        ANALYSIS_SAMPLE_RATE,
        source_kind="uploaded_audio",
        score_summary={
            "start_measure": 1,
            "end_measure": 16,
            "estimated_tempo_bpm": 120,
        },
    )

    assert abs(result["tempo_bpm"] - 120) < 5
    assert result["tempo_confidence"] > 0.2
    assert result["beat_count"] >= 40
    assert result["onset_count"] >= 30
    assert result["sections"]
    assert len(result["alignment_suggestions"]) >= 2
    assert {item["basis"] for item in result["alignment_suggestions"]} == {"audio_beat_grid"}


def test_video_highlight_alignment_takes_priority():
    visual_sync = [
        {"measure_number": measure, "time_seconds": measure * 1.25}
        for measure in range(1, 25, 4)
    ]
    result = analyze_samples(
        click_track(duration=32),
        ANALYSIS_SAMPLE_RATE,
        source_kind="video_audio",
        score_summary={"start_measure": 1, "end_measure": 24, "estimated_tempo_bpm": 120},
        visual_sync=visual_sync,
    )

    assert len(result["alignment_suggestions"]) >= 2
    assert {item["basis"] for item in result["alignment_suggestions"]} == {"video_highlight"}
