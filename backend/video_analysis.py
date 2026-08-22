from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


MAX_ANALYSIS_FRAMES = 180
MAX_ANALYSIS_SECONDS = 20 * 60
MAX_OUTPUT_DIMENSION = 1920


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    duration: float


@dataclass(frozen=True)
class ExtractedFrame:
    path: Path
    time_seconds: float
    source_frame: int


def _parse_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    numerator, separator, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe_video(video_path: Path, ffprobe_path: str = "ffprobe") -> VideoProbe:
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "无法读取视频信息")[-500:])

    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
        duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("视频缺少可分析的画面信息") from exc

    if width < 2 or height < 2 or fps <= 0 or duration <= 0:
        raise RuntimeError("视频的尺寸、帧率或时长无效")
    return VideoProbe(width=width, height=height, fps=fps, duration=duration)


def estimate_frame_count(start_seconds: float, end_seconds: float, fps: float, frame_interval: int) -> int:
    if end_seconds <= start_seconds or fps <= 0 or frame_interval <= 0:
        return 0
    return math.floor((end_seconds - start_seconds) * fps / frame_interval) + 1


def normalized_crop_to_pixels(
    probe: VideoProbe,
    crop_x: float,
    crop_y: float,
    crop_width: float,
    crop_height: float,
) -> tuple[int, int, int, int]:
    values = (crop_x, crop_y, crop_width, crop_height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("谱面范围包含无效数值")
    if crop_x < 0 or crop_y < 0 or crop_width <= 0 or crop_height <= 0:
        raise ValueError("谱面范围必须位于视频画面内")
    if crop_x + crop_width > 1.000001 or crop_y + crop_height > 1.000001:
        raise ValueError("谱面范围超出了视频画面")

    x = min(probe.width - 2, max(0, round(crop_x * probe.width)))
    y = min(probe.height - 2, max(0, round(crop_y * probe.height)))
    width = min(probe.width - x, max(2, round(crop_width * probe.width)))
    height = min(probe.height - y, max(2, round(crop_height * probe.height)))
    return x, y, width, height


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    frame_interval: int,
    crop_x: float,
    crop_y: float,
    crop_width: float,
    crop_height: float,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
) -> tuple[VideoProbe, list[ExtractedFrame]]:
    probe = probe_video(video_path, ffprobe_path)
    effective_end = min(end_seconds, probe.duration)
    if start_seconds < 0 or effective_end - start_seconds < 0.25:
        raise ValueError("结束时间必须比开始时间至少晚 0.25 秒")
    if effective_end - start_seconds > MAX_ANALYSIS_SECONDS:
        raise ValueError("单次分析最多选择 20 分钟，请分段处理")
    if frame_interval < 1:
        raise ValueError("抽帧间隔至少为 1 帧")

    estimated = estimate_frame_count(start_seconds, effective_end, probe.fps, frame_interval)
    if estimated > MAX_ANALYSIS_FRAMES:
        raise ValueError(f"预计生成 {estimated} 张，最多允许 {MAX_ANALYSIS_FRAMES} 张；请缩短时间或增大抽帧间隔")

    x, y, width, height = normalized_crop_to_pixels(
        probe, crop_x, crop_y, crop_width, crop_height
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    output_pattern = output_dir / "frame-%04d.jpg"
    video_filter = f"select=not(mod(n\\,{frame_interval})),crop={width}:{height}:{x}:{y}"
    scale = min(1.0, MAX_OUTPUT_DIMENSION / max(width, height))
    if scale < 1:
        output_width = max(2, round(width * scale / 2) * 2)
        output_height = max(2, round(height * scale / 2) * 2)
        video_filter += f",scale={output_width}:{output_height}"
    result = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{effective_end - start_seconds:.3f}",
            "-i",
            str(video_path),
            "-vf",
            video_filter,
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            "-start_number",
            "1",
            str(output_pattern),
        ],
        capture_output=True,
        text=True,
        timeout=20 * 60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg 抽帧失败")[-700:])

    paths = sorted(output_dir.glob("frame-*.jpg"))
    if not paths:
        raise RuntimeError("选定范围内没有提取到画面，请调整开始和结束时间")
    if len(paths) > MAX_ANALYSIS_FRAMES:
        raise RuntimeError("实际切片数量超过安全上限，请增大抽帧间隔")

    frames = [
        ExtractedFrame(
            path=path,
            time_seconds=min(effective_end, start_seconds + index * frame_interval / probe.fps),
            source_frame=round(start_seconds * probe.fps) + index * frame_interval,
        )
        for index, path in enumerate(paths)
    ]
    return probe, frames
