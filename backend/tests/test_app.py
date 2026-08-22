from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.bilibili import parse_bilibili_source
from backend.video_analysis import ExtractedFrame, VideoProbe, estimate_frame_count, normalized_crop_to_pixels
from backend.tab_recognition import TabRecognitionResult


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "app-data")) as test_client:
        yield test_client


def register(client: TestClient):
    response = client.post(
        "/api/auth/register",
        json={"email": "player@example.com", "display_name": "夜练", "password": "correct-horse"},
    )
    assert response.status_code == 201
    assert "HttpOnly" in response.headers["set-cookie"]
    return response.json()


@pytest.mark.parametrize(
    ("raw", "kind", "source_id"),
    [
        ("BV1xx411c7mD", "bv", "BV1xx411c7mD"),
        ("https://www.bilibili.com/video/BV1xx411c7mD?p=2", "bv", "BV1xx411c7mD"),
        (
            "https://www.bilibili.com/video/BV1WM4y1b7LQ/?share_source=copy_web&vd_source=test",
            "bv",
            "BV1WM4y1b7LQ",
        ),
        ("av170001", "av", "av170001"),
    ],
)
def test_parse_bilibili_source(raw, kind, source_id):
    source = parse_bilibili_source(raw)
    assert source.kind == kind
    assert source.source_id == source_id
    assert source.url.endswith(source_id)


def test_rejects_non_bilibili_url():
    with pytest.raises(ValueError):
        parse_bilibili_source("https://example.com/video/BV1xx411c7mD")


def test_auth_project_library_and_sync_persistence(client: TestClient):
    user = register(client)
    assert user["display_name"] == "夜练"
    assert client.get("/api/auth/me").status_code == 200

    created = client.post(
        "/api/projects",
        json={"source_input": "BV1xx411c7mD", "title": "午夜练习", "rights_confirmed": True},
    )
    assert created.status_code == 201
    project = created.json()
    assert project["source_id"] == "BV1xx411c7mD"
    assert project["rights_confirmed"] is True

    point = client.post(
        f"/api/projects/{project['id']}/sync-points",
        json={"measure_number": 8, "time_seconds": 12.75, "score_position": 0.36, "label": "主歌"},
    )
    assert point.status_code == 201

    library = client.get("/api/projects").json()
    assert len(library) == 1
    assert library[0]["sync_points"][0]["measure_number"] == 8
    assert library[0]["sync_points"][0]["score_position"] == 0.36

    client.post("/api/auth/logout")
    assert client.get("/api/projects").status_code == 401
    login = client.post("/api/auth/login", json={"email": "player@example.com", "password": "correct-horse"})
    assert login.status_code == 200
    assert client.get("/api/projects").json()[0]["title"] == "午夜练习"


def test_inspect_caches_private_cover_for_library(client: TestClient, monkeypatch):
    register(client)
    project = client.post(
        "/api/projects",
        json={"source_input": "BV1xx411c7mD", "title": "", "rights_confirmed": False},
    ).json()

    monkeypatch.setattr(
        "backend.app.inspect_bilibili",
        lambda _: {
            "id": "BV1xx411c7mD",
            "title": "封面测试曲",
            "uploader": "测试作者",
            "duration": 123.0,
            "thumbnail": "https://i0.hdslb.com/bfs/archive/test.jpg",
            "webpage_url": "https://www.bilibili.com/video/BV1xx411c7mD",
            "extractor": "BiliBili",
        },
    )

    def fake_cache(_: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), "#252238").save(destination, "JPEG")
        return destination

    monkeypatch.setattr("backend.app.cache_bilibili_thumbnail", fake_cache)
    inspected = client.post(f"/api/projects/{project['id']}/inspect")
    assert inspected.status_code == 200
    payload = inspected.json()
    assert payload["title"] == "封面测试曲"
    assert payload["cover_url"].startswith(f"/api/projects/{project['id']}/files/cover")
    assert "_cover_path" not in payload["source_metadata"]

    cover = client.get(payload["cover_url"])
    assert cover.status_code == 200
    assert cover.headers["content-type"] == "image/jpeg"
    assert cover.headers["content-disposition"].startswith("inline")
    assert client.get("/api/projects").json()[0]["cover_url"] == payload["cover_url"]

    second_client = TestClient(client.app)
    assert second_client.get(payload["cover_url"]).status_code == 401


def test_score_images_become_pdf_and_remain_private(client: TestClient):
    register(client)
    project = client.post(
        "/api/projects",
        json={"source_input": "av170001", "title": "图片谱", "rights_confirmed": False},
    ).json()

    image_buffer = io.BytesIO()
    Image.new("RGB", (640, 900), "white").save(image_buffer, format="PNG")
    response = client.post(
        f"/api/projects/{project['id']}/score-images",
        files=[("files", ("page-1.png", image_buffer.getvalue(), "image/png"))],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pdf_ready"
    assert payload["pdf_url"]
    assert len(payload["score_images"]) == 1
    assert client.get(payload["pdf_url"]).headers["content-type"] == "application/pdf"
    assert client.get(f"{payload['pdf_url']}?download=1").headers["content-disposition"].startswith("attachment")
    assert client.get(payload["score_images"][0]["url"]).headers["content-type"] == "image/jpeg"

    second_client = TestClient(client.app)
    assert second_client.get(payload["pdf_url"]).status_code == 401


def test_download_requires_rights_confirmation(client: TestClient):
    register(client)
    project = client.post(
        "/api/projects",
        json={"source_input": "BV1xx411c7mD", "title": "只建草稿", "rights_confirmed": False},
    ).json()
    response = client.post(f"/api/projects/{project['id']}/download")
    assert response.status_code == 403

    confirmed = client.patch(
        f"/api/projects/{project['id']}/rights", json={"rights_confirmed": True}
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["rights_confirmed"] is True


def test_video_analysis_helpers():
    probe = VideoProbe(width=1920, height=1080, fps=30, duration=20)
    assert estimate_frame_count(2, 8, probe.fps, 60) == 4
    assert normalized_crop_to_pixels(probe, 0.1, 0.2, 0.8, 0.5) == (192, 216, 1536, 540)


def test_video_analysis_persists_candidate_frames(client: TestClient, monkeypatch, tmp_path):
    register(client)
    project = client.post(
        "/api/projects",
        json={"source_input": "BV1WM4y1b7LQ", "title": "切片测试", "rights_confirmed": True},
    ).json()
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"test-video-placeholder")
    with client.app.state.db.connect() as connection:
        connection.execute(
            "UPDATE projects SET video_path = ?, status = 'video_ready' WHERE id = ?",
            (str(source_video), project["id"]),
        )

    probe = VideoProbe(width=1280, height=720, fps=30, duration=12)
    monkeypatch.setattr("backend.app.probe_video", lambda _: probe)

    def fake_extract(_: Path, output_dir: Path, **__):
        output_dir.mkdir(parents=True)
        frame_path = output_dir / "frame-0001.jpg"
        Image.new("RGB", (640, 280), "white").save(frame_path, format="JPEG")
        return probe, [ExtractedFrame(frame_path, time_seconds=2.0, source_frame=60)]

    monkeypatch.setattr("backend.app.extract_video_frames", fake_extract)
    response = client.post(
        f"/api/projects/{project['id']}/video-analysis",
        json={
            "start_seconds": 2,
            "end_seconds": 8,
            "frame_interval": 60,
            "crop_x": 0.1,
            "crop_y": 0.5,
            "crop_width": 0.8,
            "crop_height": 0.4,
        },
    )
    assert response.status_code == 202
    assert response.json()["estimated_frames"] == 4

    payload = client.get(f"/api/projects/{project['id']}").json()
    assert payload["status"] == "pdf_ready"
    assert payload["video_analysis"]["status"] == "complete"
    assert payload["video_analysis"]["frame_count"] == 1
    assert payload["video_analysis"]["preview_pdf_status"] == "complete"
    assert payload["pdf_url"]
    assert payload["video_frames"][0]["time_seconds"] == 2.0
    assert client.get(payload["video_frames"][0]["url"]).headers["content-type"] == "image/jpeg"
    assert client.get(payload["pdf_url"]).headers["content-type"] == "application/pdf"


def test_recognition_prefers_video_tab_frames(client: TestClient, monkeypatch, tmp_path):
    register(client)
    project = client.post(
        "/api/projects",
        json={"source_input": "BV1WM4y1b7LQ", "title": "TAB 识别", "rights_confirmed": True},
    ).json()
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (800, 240), "white").save(frame_path)
    with client.app.state.db.connect() as connection:
        connection.execute(
            """
            INSERT INTO assets(id, project_id, kind, original_name, stored_path, media_type,
            metadata, sort_order, created_at) VALUES (?, ?, 'video_frame', ?, ?, 'image/jpeg', ?, 0, ?)
            """,
            (
                "frame-id",
                project["id"],
                "切片 001",
                str(frame_path),
                '{"time_seconds": 1.5, "source_frame": 45}',
                "2026-01-01T00:00:00+00:00",
            ),
        )

    monkeypatch.setattr(
        "backend.app.capability_status",
        lambda: {
            "ffmpeg": True,
            "yt_dlp": True,
            "audiveris": False,
            "audiveris_path": None,
            "tab_ocr": True,
            "tesseract_path": "tesseract",
        },
    )

    def fake_recognize(_frames, output_dir, **_kwargs):
        output_dir.mkdir(parents=True)
        score = output_dir / "recognized.musicxml"
        score.write_text("<score-partwise version='4.0'/>", encoding="utf-8")
        diagnostics = output_dir / "recognition.json"
        diagnostics.write_text(
            """{
              "summary": {"engine": "tab_cv_tesseract", "start_measure": 1, "end_measure": 8, "estimated_tempo_bpm": 120},
              "sync_suggestions": [],
              "frames": [],
              "measures": [{
                "number": 1,
                "quality": 90,
                "source_time": 1.5,
                "events": [{"onset_eighths": 0, "duration_eighths": 2, "notes": [{"string": 1, "fret": 3}]}]
              }]
            }""",
            encoding="utf-8",
        )
        return TabRecognitionResult(
            score,
            diagnostics,
            {
                "engine": "tab_cv_tesseract",
                "engine_label": "六线 TAB 专用识别（Beta）",
                "measure_count": 8,
                "low_confidence_glyphs": 0,
            },
        )

    monkeypatch.setattr("backend.app.recognize_tab_frames", fake_recognize)
    response = client.post(f"/api/projects/{project['id']}/recognize")

    assert response.status_code == 202
    assert response.json()["engine"] == "tablature"
    payload = client.get(f"/api/projects/{project['id']}").json()
    assert payload["status"] == "score_ready"
    assert payload["recognition_summary"]["measure_count"] == 8
    assert client.get(payload["score_file_url"]).status_code == 200

    diagnostics = client.get(f"/api/projects/{project['id']}/recognition")
    assert diagnostics.status_code == 200
    edited = client.patch(
        f"/api/projects/{project['id']}/recognition/measures/1",
        json={
            "events": [
                {
                    "onset_eighths": 0,
                    "duration_eighths": 2,
                    "notes": [{"string": 2, "fret": 7, "technique": "slide"}],
                }
            ]
        },
    )
    assert edited.status_code == 200
    assert edited.json()["measures"][0]["events"][0]["notes"] == [
        {"string": 2, "fret": 7, "technique": "slide"}
    ]


def test_audio_analysis_uses_video_and_applies_non_destructive_alignment(client: TestClient, monkeypatch, tmp_path):
    register(client)
    project = client.post(
        "/api/projects",
        json={"source_input": "BV1WM4y1b7LQ", "title": "音频分析", "rights_confirmed": True},
    ).json()
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video-placeholder")
    with client.app.state.db.connect() as connection:
        connection.execute(
            """
            UPDATE projects SET video_path = ?, recognition_summary = ? WHERE id = ?
            """,
            (
                str(video_path),
                '{"start_measure": 1, "end_measure": 16, "estimated_tempo_bpm": 120}',
                project["id"],
            ),
        )

    monkeypatch.setattr(
        "backend.app.capability_status",
        lambda: {
            "ffmpeg": True,
            "yt_dlp": True,
            "audiveris": False,
            "audiveris_path": None,
            "tab_ocr": True,
            "tesseract_path": "tesseract",
            "audio_analysis": True,
        },
    )
    monkeypatch.setattr(
        "backend.app.analyze_audio_file",
        lambda *_args, **_kwargs: {
            "engine": "test",
            "source": "video_audio",
            "duration_seconds": 20,
            "tempo_bpm": 120,
            "tempo_confidence": 0.9,
            "beat_count": 40,
            "onset_count": 30,
            "beat_times": [],
            "onset_times": [],
            "sections": [{"label": "A", "start_seconds": 0, "end_seconds": 20, "confidence": 0.8}],
            "alignment_suggestions": [
                {"measure_number": 1, "time_seconds": 0, "score_position": 0, "label": "自动 · A"},
                {"measure_number": 9, "time_seconds": 16, "score_position": 0.5, "label": "自动 · A"},
            ],
            "warnings": [],
        },
    )

    response = client.post(f"/api/projects/{project['id']}/audio-analysis")
    assert response.status_code == 202
    analyzed = client.get(f"/api/projects/{project['id']}").json()
    assert analyzed["audio_analysis"]["status"] == "complete"
    assert analyzed["audio_analysis"]["source"] == "video_audio"

    applied = client.post(f"/api/projects/{project['id']}/audio-analysis/apply")
    assert applied.status_code == 200
    assert len(applied.json()["sync_points"]) == 2

    applied_again = client.post(f"/api/projects/{project['id']}/audio-analysis/apply")
    assert applied_again.status_code == 200
    assert len(applied_again.json()["sync_points"]) == 2
