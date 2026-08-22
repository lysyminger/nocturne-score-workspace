from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps


BV_PATTERN = re.compile(r"(?<![0-9A-Za-z])(BV[0-9A-Za-z]{10})(?![0-9A-Za-z])", re.IGNORECASE)
AV_PATTERN = re.compile(r"(?<![0-9A-Za-z])av(\d+)(?![0-9A-Za-z])", re.IGNORECASE)
ALLOWED_HOSTS = {"bilibili.com", "www.bilibili.com", "m.bilibili.com"}
THUMBNAIL_HOST_SUFFIXES = ("bilibili.com", "hdslb.com", "biliimg.com")
MAX_THUMBNAIL_BYTES = 8 * 1024 * 1024
MAX_THUMBNAIL_PIXELS = 32_000_000


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


def _is_allowed_thumbnail_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in THUMBNAIL_HOST_SUFFIXES
    )


def cache_bilibili_thumbnail(url: str, destination: Path) -> Path:
    """Download and normalize a Bilibili thumbnail for private local serving."""
    if not _is_allowed_thumbnail_url(url):
        raise ValueError("封面地址不是受支持的 B 站图片域名")

    current_url = url
    content = bytearray()
    with httpx.Client(
        timeout=15,
        follow_redirects=False,
        headers={"Referer": "https://www.bilibili.com/", "User-Agent": "Nocturne/0.1"},
    ) as client:
        for _ in range(4):
            if not _is_allowed_thumbnail_url(current_url):
                raise ValueError("封面重定向离开了受支持的图片域名")
            with client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("封面重定向缺少目标地址")
                    current_url = str(response.url.join(location))
                    continue
                response.raise_for_status()
                length = int(response.headers.get("content-length") or 0)
                if length > MAX_THUMBNAIL_BYTES:
                    raise ValueError("视频封面文件过大")
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_THUMBNAIL_BYTES:
                        raise ValueError("视频封面文件过大")
                break
        else:
            raise ValueError("视频封面重定向次数过多")

    with Image.open(BytesIO(content)) as source:
        if source.width * source.height > MAX_THUMBNAIL_PIXELS:
            raise ValueError("视频封面像素尺寸过大")
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1280, 720), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        try:
            image.save(temporary, "JPEG", quality=88, optimize=True)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination
