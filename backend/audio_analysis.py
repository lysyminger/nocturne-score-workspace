from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np


ANALYSIS_SAMPLE_RATE = 11_025
MAX_ANALYSIS_SECONDS = 20 * 60
FRAME_SIZE = 2_048
HOP_SIZE = 512


def decode_audio(
    source_path: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
) -> np.ndarray:
    result = subprocess.run(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(source_path),
            "-t",
            str(MAX_ANALYSIS_SECONDS),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        timeout=10 * 60,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace") if result.stderr else "FFmpeg 解码失败"
        raise RuntimeError(message[-500:])
    samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32_768.0
    if samples.size < sample_rate:
        raise RuntimeError("音频有效时长不足 1 秒")
    return samples


def _frame_features(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if samples.size < FRAME_SIZE:
        samples = np.pad(samples, (0, FRAME_SIZE - samples.size))
    frame_count = 1 + math.ceil((samples.size - FRAME_SIZE) / HOP_SIZE)
    padded_size = (frame_count - 1) * HOP_SIZE + FRAME_SIZE
    if samples.size < padded_size:
        samples = np.pad(samples, (0, padded_size - samples.size))

    window = np.hanning(FRAME_SIZE).astype(np.float32)
    frequencies = np.fft.rfftfreq(FRAME_SIZE, 1 / sample_rate)
    valid_bins = np.flatnonzero((frequencies >= 55) & (frequencies <= 5_000))
    midi = np.rint(69 + 12 * np.log2(frequencies[valid_bins] / 440)).astype(np.int16)
    pitch_classes = np.mod(midi, 12)

    flux = np.zeros(frame_count, dtype=np.float32)
    rms = np.zeros(frame_count, dtype=np.float32)
    chroma = np.zeros((frame_count, 12), dtype=np.float32)
    previous = np.zeros(FRAME_SIZE // 2 + 1, dtype=np.float32)
    for index in range(frame_count):
        start = index * HOP_SIZE
        frame = samples[start : start + FRAME_SIZE]
        rms[index] = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        magnitude = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        positive = np.maximum(magnitude - previous, 0)
        flux[index] = float(positive.sum() / (magnitude.sum() + 1e-8))
        weighted = np.sqrt(magnitude[valid_bins])
        chroma[index] = np.bincount(pitch_classes, weights=weighted, minlength=12)[:12]
        norm = float(np.linalg.norm(chroma[index]))
        if norm > 0:
            chroma[index] /= norm
        previous = magnitude

    times = (np.arange(frame_count, dtype=np.float32) * HOP_SIZE + FRAME_SIZE / 2) / sample_rate
    return times, flux, rms, chroma


def _novelty_envelope(flux: np.ndarray) -> np.ndarray:
    radius = 9
    baseline = np.convolve(flux, np.ones(radius, dtype=np.float32) / radius, mode="same")
    novelty = np.maximum(flux - baseline * 0.72, 0)
    high = float(np.quantile(novelty, 0.98))
    if high > 1e-8:
        novelty = np.clip(novelty / high, 0, 2)
    return novelty.astype(np.float32)


def _detect_onsets(times: np.ndarray, novelty: np.ndarray, sample_rate: int) -> list[float]:
    if novelty.size < 3:
        return []
    threshold = max(0.16, float(np.median(novelty) + np.std(novelty) * 0.65))
    candidates = np.flatnonzero(
        (novelty[1:-1] >= novelty[:-2])
        & (novelty[1:-1] > novelty[2:])
        & (novelty[1:-1] >= threshold)
    ) + 1
    selected: list[int] = []
    min_frames = max(1, round(0.09 / (HOP_SIZE / sample_rate)))
    for candidate in candidates:
        if selected and candidate - selected[-1] < min_frames:
            if novelty[candidate] > novelty[selected[-1]]:
                selected[-1] = int(candidate)
        else:
            selected.append(int(candidate))
    return [round(float(times[index]), 3) for index in selected]


def _estimate_tempo(
    novelty: np.ndarray,
    sample_rate: int,
    *,
    expected_tempo: float | None,
) -> tuple[float, float, float]:
    frame_rate = sample_rate / HOP_SIZE
    centered = novelty - float(np.mean(novelty))
    minimum_lag = max(1, round(frame_rate * 60 / 220))
    maximum_lag = min(len(centered) - 1, round(frame_rate * 60 / 55))
    if maximum_lag <= minimum_lag:
        return expected_tempo or 120.0, 0.0, 0.0

    scores: list[tuple[float, int, float]] = []
    energy = float(np.dot(centered, centered)) + 1e-8
    for lag in range(minimum_lag, maximum_lag + 1):
        correlation = float(np.dot(centered[lag:], centered[:-lag]) / energy)
        bpm = 60 * frame_rate / lag
        preference = 1.0
        if expected_tempo and expected_tempo > 0:
            octave_distance = abs(math.log2(bpm / expected_tempo))
            preference = math.exp(-2.4 * octave_distance)
        elif bpm < 75:
            preference = 0.88
        scores.append((correlation * preference, lag, correlation))
    _, coarse_lag, raw_correlation = max(scores, key=lambda item: item[0])
    positions = np.arange(len(centered), dtype=np.float32)
    refined_scores: list[tuple[float, float, float]] = []
    for lag in np.linspace(max(minimum_lag, coarse_lag - 1), min(maximum_lag, coarse_lag + 1), 81):
        start = math.ceil(float(lag))
        shifted = np.interp(positions[start:] - lag, positions, centered)
        correlation = float(np.dot(centered[start:], shifted) / energy)
        bpm = 60 * frame_rate / lag
        preference = 1.0
        if expected_tempo and expected_tempo > 0:
            preference = math.exp(-2.4 * abs(math.log2(bpm / expected_tempo)))
        refined_scores.append((correlation * preference, float(lag), correlation))
    _, best_lag, raw_correlation = max(refined_scores, key=lambda item: item[0])
    tempo = 60 * frame_rate / best_lag

    phase_scores: list[tuple[float, float]] = []
    for phase in np.linspace(0, best_lag, max(16, round(best_lag * 8)), endpoint=False):
        sample_positions = np.arange(phase, len(novelty), best_lag)
        phase_scores.append((float(np.interp(sample_positions, positions, novelty).sum()), float(phase)))
    _, phase = max(phase_scores, default=(0.0, 0.0))
    confidence = min(0.96, max(0.0, raw_correlation * 2.2))
    return tempo, confidence, phase / frame_rate


def _beat_grid(
    duration: float,
    tempo: float,
    phase_seconds: float,
    rms: np.ndarray,
    sample_rate: int,
) -> list[float]:
    period = 60 / max(tempo, 1)
    frame_rate = sample_rate / HOP_SIZE
    rms_high = float(np.quantile(rms, 0.85)) if rms.size else 0
    active_threshold = max(1e-5, rms_high * 0.12)
    active_frames = np.flatnonzero(rms >= active_threshold)
    active_start = float(active_frames[0] / frame_rate) if active_frames.size else 0.0
    first = phase_seconds
    while first + period <= active_start:
        first += period
    while first - period >= max(0.0, active_start - period):
        first -= period
    beats = np.arange(max(0.0, first), duration + period * 0.25, period)
    return [round(float(value), 3) for value in beats if 0 <= value <= duration]


def _segment_label(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"S{index + 1}"


def _detect_sections(
    times: np.ndarray,
    rms: np.ndarray,
    chroma: np.ndarray,
    onset_novelty: np.ndarray,
    beats: list[float],
    duration: float,
) -> list[dict]:
    if len(beats) >= 8:
        starts = np.asarray(beats[::4], dtype=np.float32)
    else:
        starts = np.arange(0, duration, 2.0, dtype=np.float32)
    if starts.size == 0 or starts[0] > 0.2:
        starts = np.insert(starts, 0, 0.0)
    if starts[-1] < duration:
        edges = np.append(starts, duration)
    else:
        edges = starts
        starts = starts[:-1]
    if starts.size == 0:
        return [{"label": "A", "start_seconds": 0.0, "end_seconds": round(duration, 3), "confidence": 0.25}]

    rms_scale = float(np.quantile(rms, 0.9)) + 1e-8
    features: list[np.ndarray] = []
    for start, end in zip(edges[:-1], edges[1:]):
        indices = np.flatnonzero((times >= start) & (times < end))
        if indices.size:
            tone = chroma[indices].mean(axis=0)
            tone /= float(np.linalg.norm(tone)) + 1e-8
            energy = np.array([float(rms[indices].mean() / rms_scale)], dtype=np.float32)
            activity = np.array(
                [float(onset_novelty[indices].mean()), float(onset_novelty[indices].std())],
                dtype=np.float32,
            )
        else:
            tone = np.zeros(12, dtype=np.float32)
            energy = np.zeros(1, dtype=np.float32)
            activity = np.zeros(2, dtype=np.float32)
        features.append(np.concatenate([tone, energy * 0.72, activity * 0.85]))
    feature_matrix = np.asarray(features)

    novelty = np.zeros(len(features), dtype=np.float32)
    for index in range(2, len(features) - 2):
        before = feature_matrix[index - 2 : index].mean(axis=0)
        after = feature_matrix[index : index + 2].mean(axis=0)
        novelty[index] = float(np.linalg.norm(before - after))
    positive = novelty[novelty > 0]
    threshold = float(np.quantile(positive, 0.58)) if positive.size else float("inf")
    minimum_gap = 8
    candidates = [index for index in range(minimum_gap, len(features) - minimum_gap) if novelty[index] >= threshold]
    boundaries = [0]
    for candidate in sorted(candidates, key=lambda value: float(novelty[value]), reverse=True):
        if all(abs(candidate - existing) >= minimum_gap for existing in boundaries):
            boundaries.append(candidate)
        if len(boundaries) >= 8:
            break
    boundaries.append(len(features))
    boundaries.sort()
    while len(boundaries) < 9:
        gaps = [(right - left, left, right) for left, right in zip(boundaries, boundaries[1:])]
        gap, left, right = max(gaps, default=(0, 0, 0))
        if gap <= 28:
            break
        options = range(left + minimum_gap, right - minimum_gap + 1)
        split = max(options, key=lambda index: float(novelty[index]), default=None)
        if split is None:
            break
        boundaries.append(split)
        boundaries.sort()

    templates: list[np.ndarray] = []
    sections: list[dict] = []
    novelty_peak = float(novelty.max()) + 1e-8
    for start_index, end_index in zip(boundaries[:-1], boundaries[1:]):
        feature = feature_matrix[start_index:end_index].mean(axis=0)
        feature_norm = float(np.linalg.norm(feature)) + 1e-8
        label_index = len(templates)
        for index, template in enumerate(templates):
            similarity = float(np.dot(feature, template) / (feature_norm * (np.linalg.norm(template) + 1e-8)))
            if similarity >= 0.975:
                label_index = index
                break
        else:
            templates.append(feature)
        boundary_confidence = 0.35 if start_index == 0 else min(0.95, 0.35 + float(novelty[start_index] / novelty_peak) * 0.6)
        sections.append(
            {
                "label": _segment_label(label_index),
                "start_seconds": round(float(edges[start_index]), 3),
                "end_seconds": round(float(edges[end_index]), 3),
                "confidence": round(boundary_confidence, 3),
            }
        )
    return sections


def _alignment_suggestions(
    *,
    source_kind: str,
    beat_times: list[float],
    tempo: float,
    duration: float,
    sections: list[dict],
    score_summary: dict | None,
    visual_sync: list[dict] | None,
) -> list[dict]:
    if not score_summary:
        return []
    start_measure = int(score_summary.get("start_measure") or 1)
    end_measure = int(score_summary.get("end_measure") or score_summary.get("measure_count") or start_measure)
    if end_measure <= start_measure:
        return []

    section_starts = {round(float(section["start_seconds"]), 3): str(section["label"]) for section in sections}
    suggestions: list[dict] = []
    if source_kind == "video_audio" and visual_sync:
        ordered = sorted(
            (
                int(item["measure_number"]),
                float(item["time_seconds"]),
            )
            for item in visual_sync
            if start_measure <= int(item.get("measure_number") or 0) <= end_measure
        )
        last_measure = -10_000
        for index, (measure, seconds) in enumerate(ordered):
            if index not in {0, len(ordered) - 1} and measure - last_measure < 8:
                continue
            label = min(section_starts.items(), key=lambda item: abs(item[0] - seconds))[1] if section_starts else ""
            suggestions.append(
                {
                    "measure_number": measure,
                    "time_seconds": round(seconds, 3),
                    "score_position": round((measure - start_measure) / (end_measure - start_measure), 5),
                    "label": f"自动 · {label}" if label else "自动 · 视频高亮",
                    "confidence": 0.88,
                    "basis": "video_highlight",
                }
            )
            last_measure = measure
        if len(suggestions) >= 2:
            return suggestions

    beat_origin = beat_times[0] if beat_times else 0.0
    measure_seconds = 240 / max(tempo, 1)
    for measure in range(start_measure, end_measure + 1, 8):
        seconds = beat_origin + (measure - start_measure) * measure_seconds
        if seconds > duration:
            break
        label = min(section_starts.items(), key=lambda item: abs(item[0] - seconds))[1] if section_starts else ""
        suggestions.append(
            {
                "measure_number": measure,
                "time_seconds": round(seconds, 3),
                "score_position": round((measure - start_measure) / (end_measure - start_measure), 5),
                "label": f"自动 · {label}" if label else "自动 · 节拍网格",
                "confidence": 0.62,
                "basis": "audio_beat_grid",
            }
        )
    if suggestions and suggestions[-1]["measure_number"] != end_measure:
        seconds = beat_origin + (end_measure - start_measure) * measure_seconds
        if seconds <= duration:
            suggestions.append(
                {
                    "measure_number": end_measure,
                    "time_seconds": round(seconds, 3),
                    "score_position": 1.0,
                    "label": "自动 · 结尾",
                    "confidence": 0.58,
                    "basis": "audio_beat_grid",
                }
            )
    return suggestions


def analyze_samples(
    samples: np.ndarray,
    sample_rate: int,
    *,
    source_kind: str,
    score_summary: dict | None = None,
    visual_sync: list[dict] | None = None,
) -> dict:
    samples = np.asarray(samples, dtype=np.float32)
    duration = float(samples.size / sample_rate)
    times, flux, rms, chroma = _frame_features(samples, sample_rate)
    novelty = _novelty_envelope(flux)
    onsets = _detect_onsets(times, novelty, sample_rate)
    expected_tempo = None
    if score_summary and score_summary.get("estimated_tempo_bpm"):
        expected_tempo = float(score_summary["estimated_tempo_bpm"])
    tempo, tempo_confidence, phase_seconds = _estimate_tempo(
        novelty,
        sample_rate,
        expected_tempo=expected_tempo,
    )
    beats = _beat_grid(duration, tempo, phase_seconds, rms, sample_rate)
    sections = _detect_sections(times, rms, chroma, novelty, beats, duration)
    suggestions = _alignment_suggestions(
        source_kind=source_kind,
        beat_times=beats,
        tempo=tempo,
        duration=duration,
        sections=sections,
        score_summary=score_summary,
        visual_sync=visual_sync,
    )
    warnings = [
        "段落名称是按音色与能量相似度得到的 A/B/C 候选，不等同于确定的主歌或副歌",
        "速度可能以半拍或双倍速度表示，请结合听感与拍号校对",
        "自动对齐是可编辑建议，不会覆盖已有的手动同步点",
    ]
    return {
        "engine": "spectral_flux_chroma_v1",
        "source": source_kind,
        "duration_seconds": round(duration, 3),
        "tempo_bpm": round(float(tempo), 1),
        "tempo_confidence": round(float(tempo_confidence), 3),
        "beat_count": len(beats),
        "onset_count": len(onsets),
        "beat_times": beats,
        "onset_times": onsets[:2_000],
        "sections": sections,
        "alignment_suggestions": suggestions,
        "warnings": warnings,
    }


def analyze_audio_file(
    source_path: Path,
    *,
    source_kind: str,
    ffmpeg_path: str = "ffmpeg",
    score_summary: dict | None = None,
    visual_sync: list[dict] | None = None,
) -> dict:
    samples = decode_audio(source_path, ffmpeg_path=ffmpeg_path)
    return analyze_samples(
        samples,
        ANALYSIS_SAMPLE_RATE,
        source_kind=source_kind,
        score_summary=score_summary,
        visual_sync=visual_sync,
    )
