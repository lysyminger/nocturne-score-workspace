from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.bilibili import parse_bilibili_source
from backend.pdf_score_import import PdfPage, PdfScoreImport, PdfScoreSystem
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


def test_manual_tab_project_supports_per_string_chords_and_appending_measures(client: TestClient):
    register(client)

    created = client.post(
        "/api/projects/manual-tab",
        json={"title": "手动打谱测试", "measure_count": 2, "tempo_bpm": 108},
    )
    assert created.status_code == 201
    project = created.json()
    assert project["source_kind"] == "manual_tab"
    assert project["status"] == "score_ready"
    assert project["recognition_summary"]["engine"] == "tab_manual_editor"
    assert client.get(project["score_file_url"]).status_code == 200

    edited = client.patch(
        f"/api/projects/{project['id']}/recognition/measures/1",
        json={
            "events": [
                {
                    "onset_eighths": 0.5,
                    "duration_eighths": 0.875,
                    "notes": [
                        {"string": 1, "fret": 7},
                        {"string": 2, "fret": 8},
                        {"string": 6, "fret": 5},
                    ],
                }
            ]
        },
    )
    assert edited.status_code == 200
    assert [note["string"] for note in edited.json()["measures"][0]["events"][0]["notes"]] == [1, 2, 6]
    assert edited.json()["measures"][0]["events"][0]["duration_eighths"] == 0.875

    appended = client.post(f"/api/projects/{project['id']}/recognition/measures")
    assert appended.status_code == 201
    assert appended.json()["summary"]["end_measure"] == 3
    assert appended.json()["measures"][-1]["events"] == []

    persisted = client.get(f"/api/projects/{project['id']}/recognition").json()
    assert persisted["measures"][0]["events"][0]["notes"][2] == {"string": 6, "fret": 5}
    assert persisted["measures"][0]["events"][0]["duration_eighths"] == 0.875
    retry = client.post(f"/api/projects/{project['id']}/recognition/measures/1/retry")
    assert retry.status_code == 409
    assert "没有源视频帧" in retry.json()["detail"]


def test_manual_tab_api_accepts_explicit_rests_and_reports_complete_bar(client: TestClient):
    register(client)
    project = client.post(
        "/api/projects/manual-tab",
        json={"title": "实体休止测试", "measure_count": 1, "tempo_bpm": 100},
    ).json()

    edited = client.patch(
        f"/api/projects/{project['id']}/recognition/measures/1",
        json={
            "time_signature": {"numerator": 3, "denominator": 4},
            "events": [
                {"onset_eighths": 0, "duration_eighths": 2, "notes": [{"string": 1, "fret": 5}]},
                {"onset_eighths": 2, "duration_eighths": 4, "notes": [], "rest": True},
            ]
        },
    )

    assert edited.status_code == 200
    payload = edited.json()
    assert payload["measures"][0]["time_signature"] == {"numerator": 3, "denominator": 4}
    assert payload["measures"][0]["events"][1]["rest"] is True
    assert payload["measures"][0]["validation"]["is_complete"] is True
    assert payload["summary"]["invalid_measures"] == []


def test_manual_tab_can_insert_measure_and_shift_later_sync_points(client: TestClient):
    register(client)
    project = client.post(
        "/api/projects/manual-tab",
        json={"title": "插入小节测试", "measure_count": 3, "tempo_bpm": 100},
    ).json()
    client.patch(
        f"/api/projects/{project['id']}/recognition/measures/3",
        json={
            "events": [
                {"onset_eighths": 0, "duration_eighths": 2, "notes": [{"string": 1, "fret": 9}]},
            ]
        },
    )
    client.post(
        f"/api/projects/{project['id']}/sync-points",
        json={"measure_number": 3, "time_seconds": 5, "score_position": 0.5, "label": "原第三小节"},
    )

    inserted = client.post(f"/api/projects/{project['id']}/recognition/measures?after_measure=1")

    assert inserted.status_code == 201
    payload = inserted.json()
    assert payload["summary"]["end_measure"] == 4
    assert [measure["number"] for measure in payload["measures"]] == [1, 2, 3, 4]
    assert payload["measures"][1]["events"] == []
    assert payload["measures"][3]["events"][0]["notes"][0]["fret"] == 9
    persisted = client.get(f"/api/projects/{project['id']}").json()
    assert persisted["sync_points"][0]["measure_number"] == 4


def test_project_tempo_lock_survives_score_rebuild(client: TestClient):
    register(client)
    project = client.post(
        "/api/projects/manual-tab",
        json={"title": "锁定速度测试", "measure_count": 1, "tempo_bpm": 100},
    ).json()

    locked = client.patch(
        f"/api/projects/{project['id']}/tempo",
        json={"tempo_bpm": 156},
    )

    assert locked.status_code == 200
    assert locked.json()["tempo_bpm"] == 156
    assert locked.json()["tempo_source"] == "user"
    assert locked.json()["tempo_locked"] is True
    rebuilt = client.patch(
        f"/api/projects/{project['id']}/recognition/measures/1",
        json={"events": [{"onset_eighths": 0, "duration_eighths": 8, "rest": True}]},
    )
    assert rebuilt.status_code == 200
    assert rebuilt.json()["summary"]["estimated_tempo_bpm"] == 156
    score_xml = client.get(f"/api/projects/{project['id']}/files/score").text
    assert 'tempo="156.0"' in score_xml


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


def test_pdf_score_upload_splits_pages_and_starts_tab_recognition(client: TestClient, monkeypatch, tmp_path):
    register(client)
    project = client.post(
        "/api/projects",
        json={"source_input": "av170001", "title": "PDF TAB", "rights_confirmed": False},
    ).json()
    page_path = tmp_path / "page.jpg"
    system_path = tmp_path / "system.jpg"
    Image.new("RGB", (900, 1200), "white").save(page_path)
    Image.new("RGB", (900, 260), "white").save(system_path)
    monkeypatch.setattr(
        "backend.app.render_and_segment_pdf",
        lambda *_args: PdfScoreImport(
            pages=(PdfPage(page_path, 1, 900, 1200),),
            systems=(PdfScoreSystem(system_path, 1, 1, "staff_tab_pair", "dark_on_light"),),
            layout_counts={"staff_tab_pair": 1},
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

    def fake_recognize(frames, output_dir, **_kwargs):
        assert len(frames) == 1
        assert frames[0].path == system_path
        output_dir.mkdir(parents=True)
        score = output_dir / "recognized.musicxml"
        score.write_text("<score-partwise version='4.0'/>", encoding="utf-8")
        diagnostics = output_dir / "recognition.json"
        diagnostics.write_text('{"summary":{"engine":"tab_cv_tesseract"},"frames":[],"measures":[]}', encoding="utf-8")
        return TabRecognitionResult(
            score,
            diagnostics,
            {"engine": "tab_cv_tesseract", "engine_label": "PDF TAB 识别", "measure_count": 3},
        )

    monkeypatch.setattr("backend.app.recognize_tab_frames", fake_recognize)
    response = client.post(
        f"/api/projects/{project['id']}/score-pdf",
        files={"file": ("score.pdf", b"%PDF-1.4\nmock", "application/pdf")},
    )

    assert response.status_code == 202
    uploaded = response.json()
    assert uploaded["recognition_summary"]["engine"] == "pdf_layout"
    assert uploaded["recognition_summary"]["layout_counts"] == {"staff_tab_pair": 1}
    assert len(uploaded["score_images"]) == 1
    completed = client.get(f"/api/projects/{project['id']}").json()
    assert completed["status"] == "score_ready"
    assert completed["recognition_summary"]["measure_count"] == 3
    assert client.get(completed["pdf_url"]).content == b"%PDF-1.4\nmock"
    assert client.get(completed["score_file_url"]).status_code == 200


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
            "ai_tab_recognition": True,
        },
    )

    recognition_options = {}

    def fake_recognize(_frames, output_dir, **kwargs):
        recognition_options.update(kwargs)
        output_dir.mkdir(parents=True)
        score = output_dir / "recognized.musicxml"
        score.write_text("<score-partwise version='4.0'/>", encoding="utf-8")
        diagnostics = output_dir / "recognition.json"
        diagnostics.write_text(
            """{
              "summary": {"engine": "tab_cv_tesseract", "start_measure": 1, "end_measure": 8, "estimated_tempo_bpm": 120},
              "sync_suggestions": [],
              "frames": [{
                "name": "frame.jpg",
                "time_seconds": 1.5,
                "source_frame": 45,
                "start_measure": 1,
                "start_measure_confidence": 96,
                "raw_measure_labels": ["1", "2", "3", "4"]
              }],
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
    response = client.post(f"/api/projects/{project['id']}/recognize?mode=ai")

    assert response.status_code == 202
    assert response.json()["engine"] == "tablature"
    assert response.json()["mode"] == "ai"
    assert recognition_options["recognition_mode"] == "ai"
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
                    "onset_eighths": 0.5,
                    "duration_eighths": 0.5,
                    "notes": [{"string": 2, "fret": 7, "technique": "slide"}],
                }
            ]
        },
    )
    assert edited.status_code == 200
    assert edited.json()["measures"][0]["events"][0]["notes"] == [
        {"string": 2, "fret": 7, "technique": "slide"}
    ]
    assert edited.json()["measures"][0]["events"][0]["duration_eighths"] == 0.5
    assert edited.json()["summary"]["verified_measure_count"] == 1
    assert edited.json()["summary"]["verified_note_count"] == 2
    assert edited.json()["summary"]["human_verified_accuracy"] == 0

    def fake_retry_measure(frame, _output_dir, **kwargs):
        assert frame.path == frame_path
        assert frame.source_frame == 45
        assert kwargs["frame_start_measure"] == 1
        assert kwargs["measure_number"] == 1
        return {
            "number": 1,
            "quality": 82,
            "source_time": 1.5,
            "events": [
                {
                    "onset_eighths": 0,
                    "duration_eighths": 1,
                    "notes": [{"string": 1, "fret": 5}],
                }
            ],
            "source_frame": 45,
            "source_name": "frame.jpg",
        }

    monkeypatch.setattr("backend.app.recognize_tab_measure", fake_retry_measure)
    retried = client.post(f"/api/projects/{project['id']}/recognition/measures/1/retry")
    assert retried.status_code == 200
    assert retried.json()["events"][0]["notes"] == [{"string": 1, "fret": 5}]

    # A retry is only a proposal: the manually saved measure remains unchanged
    # until the user explicitly saves the replacement in the review panel.
    persisted = client.get(f"/api/projects/{project['id']}/recognition").json()
    assert persisted["measures"][0]["events"][0]["duration_eighths"] == 0.5


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
