from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


BV_PATTERN = re.compile(r"(?<![0-9A-Za-z])(BV[0-9A-Za-z]{10})(?![0-9A-Za-z])", re.IGNORECASE)
AV_PATTERN = re.compile(r"(?<![0-9A-Za-z])av(\d+)(?![0-9A-Za-z])", re.IGNORECASE)
ALLOWED_HOSTS = {"bilibili.com", "www.bilibili.com", "m.bilibili.com"}


@dataclass(frozen=True)
class BilibiliSource:
    kind: str
    source_id: str
    url: str


def parse_bilibili_source(raw: str) -> BilibiliSource:
    value = raw.strip()
    if not value:
        raise ValueError("请输入 B 站链接、BV 号或 AV 号")

    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower() not in ALLOWED_HOSTS:
            raise ValueError("目前只支持 bilibili.com 的公开视频地址")

    bv_match = BV_PATTERN.search(value)
    if bv_match:
        source_id = "BV" + bv_match.group(1)[2:]
        return BilibiliSource("bv", source_id, f"https://www.bilibili.com/video/{source_id}")

    av_match = AV_PATTERN.search(value)
    if av_match:
        source_id = f"av{av_match.group(1)}"
        return BilibiliSource("av", source_id, f"https://www.bilibili.com/video/{source_id}")

    raise ValueError("没有识别到有效的 BV 或 AV 号")


def inspect_bilibili(url: str) -> dict:
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 15,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)

    extractor = str(info.get("extractor_key") or info.get("extractor") or "")
    if "bili" not in extractor.lower():
        raise ValueError("解析结果不是 B 站视频")

    return {
        "id": str(info.get("id") or ""),
        "title": str(info.get("title") or "未命名视频")[:200],
        "uploader": str(info.get("uploader") or "")[:120],
        "duration": float(info.get("duration") or 0),
        "thumbnail": str(info.get("thumbnail") or ""),
        "webpage_url": str(info.get("webpage_url") or url),
        "extractor": extractor,
    }

