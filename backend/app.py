from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
import httpx
from starlette.concurrency import run_in_threadpool

from backend.audio_analysis import analyze_audio_file
from backend.bilibili import cache_bilibili_thumbnail, inspect_bilibili, parse_bilibili_source
from backend.database import Database
from backend.pdf_score_import import render_and_segment_pdf
from backend.score_pdf import build_slice_preview_pdf
from backend.security import hash_password, hash_session_token, new_session_token, verify_password
from backend.tab_recognition import FrameInput as TabFrameInput
from backend.tab_recognition import (
    append_blank_tab_measure,
    create_blank_tab_score,
    detect_tempo_from_images,
    recognize_tab_frames,
    recognize_tab_measure,
    update_recognized_measure,
)
from backend.video_analysis import (
    MAX_ANALYSIS_FRAMES,
    MAX_ANALYSIS_SECONDS,
    estimate_frame_count,
    extract_video_frames,
    normalized_crop_to_pixels,
    probe_video,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
SESSION_COOKIE = "nocturne_session"
SESSION_DAYS = 30
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_AUDIO_BYTES = 120 * 1024 * 1024
MAX_SCORE_BYTES = 40 * 1024 * 1024
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
SCORE_EXTENSIONS = {".gp", ".gp3", ".gp4", ".gp5", ".gpx", ".musicxml", ".xml", ".mxl"}


def score_tempo_candidate(path: Path) -> float | None:
    try:
        if path.suffix.lower() == ".mxl":
            with zipfile.ZipFile(path) as archive:
                names = [name for name in archive.namelist() if name.lower().endswith((".xml", ".musicxml")) and not name.startswith("META-INF/")]
                if not names:
                    return None
                root = ET.fromstring(archive.read(names[0]))
        elif path.suffix.lower() in {".xml", ".musicxml"}:
            root = ET.parse(path).getroot()
        else:
            return None
    except (OSError, ET.ParseError, zipfile.BadZipFile, KeyError):
        return None
    for element in root.iter():
        name = element.tag.rsplit("}", 1)[-1]
        raw_value = element.attrib.get("tempo") if name == "sound" else element.text if name == "per-minute" else None
        if raw_value is None:
            continue
        try:
            value = float(raw_value.strip())
        except (AttributeError, ValueError):
            continue
        if 30 <= value <= 300:
            return round(value, 1)
    return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text() -> str:
    return utc_now().isoformat()


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    display_name: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class ProjectCreateRequest(BaseModel):
    source_input: str = Field(min_length=2, max_length=500)
    title: str = Field(default="", max_length=120)
    rights_confirmed: bool = False


class ManualTabProjectRequest(BaseModel):
    title: str = Field(default="未命名六线谱", min_length=1, max_length=120)
    measure_count: int = Field(default=8, ge=1, le=128)
    tempo_bpm: float = Field(default=120, ge=30, le=300)


class ProjectUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ProjectRightsRequest(BaseModel):
    rights_confirmed: bool


class ProjectTempoRequest(BaseModel):
    tempo_bpm: float = Field(ge=30, le=300)


class VideoSliceRequest(BaseModel):
    start_seconds: float = Field(ge=0, le=86400)
    end_seconds: float = Field(gt=0, le=86400)
    frame_interval: int = Field(ge=1, le=3600)
    crop_x: float = Field(ge=0, le=1)
    crop_y: float = Field(ge=0, le=1)
    crop_width: float = Field(gt=0.01, le=1)
    crop_height: float = Field(gt=0.01, le=1)


class SyncPointRequest(BaseModel):
    measure_number: int = Field(ge=1, le=100000)
    time_seconds: float = Field(ge=0, le=86400)
    score_position: float = Field(ge=0, le=1)
    label: str = Field(default="", max_length=80)


class TabNoteEdit(BaseModel):
    string: int = Field(ge=1, le=6)
    fret: int = Field(ge=0, le=36)
    technique: Literal[
        "legato",
        "slide",
        "hammer_on",
        "pull_off",
        "bend",
        "vibrato",
        "harmonic",
        "palm_mute",
        "let_ring",
        "dead_note",
    ] | None = None


class TabEventEdit(BaseModel):
    onset_eighths: float = Field(ge=0, lt=64, multiple_of=0.5)
    duration_eighths: float = Field(ge=0.5, le=64, multiple_of=0.125)
    notes: list[TabNoteEdit] = Field(default_factory=list, max_length=6)
    rest: bool = False


class TimeSignatureEdit(BaseModel):
    numerator: int = Field(ge=1, le=32)
    denominator: int = Field(default=4)


class TabMeasureEdit(BaseModel):
    events: list[TabEventEdit] = Field(max_length=32)
    time_signature: TimeSignatureEdit | None = None
    human_verified: bool = True


def capability_status() -> dict:
    try:
        import yt_dlp  # noqa: F401

        yt_dlp_available = True
    except ImportError:
        yt_dlp_available = False

    audiveris_path = os.getenv("AUDIVERIS_BIN") or shutil.which("audiveris") or shutil.which("Audiveris")
    tesseract_path = os.getenv("TESSERACT_BIN") or shutil.which("tesseract")
    try:
        import cv2  # noqa: F401

        tab_cv_available = True
    except ImportError:
        tab_cv_available = False
    ffmpeg_available = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    vision_model_url = os.getenv("NOCTURNE_VISION_MODEL")
    ai_tab_available = False
    if vision_model_url:
        try:
            health_url = vision_model_url.rstrip("/")
            if health_url.endswith("/v1/fret-ocr"):
                health_url = health_url[: -len("/v1/fret-ocr")]
            ai_tab_available = httpx.get(f"{health_url}/health", timeout=1.5).is_success
        except httpx.HTTPError:
            ai_tab_available = False
    return {
        "ffmpeg": ffmpeg_available,
        "yt_dlp": yt_dlp_available,
        "audiveris": bool(audiveris_path),
        "audiveris_path": audiveris_path or None,
        "tab_ocr": bool(tesseract_path and tab_cv_available),
        "tesseract_path": tesseract_path or None,
        "ai_tab_recognition": ai_tab_available,
        "audio_analysis": ffmpeg_available,
    }


def create_app(data_dir: Path | None = None) -> FastAPI:
    app_data = Path(data_dir or os.getenv("APP_DATA_DIR") or ROOT_DIR / "data").resolve()
    projects_dir = app_data / "projects"
    db = Database(app_data / "nocturne.db")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        projects_dir.mkdir(parents=True, exist_ok=True)
        db.initialize()
        yield

    api = FastAPI(title="Nocturne API", version="0.1.0", lifespan=lifespan)
    api.state.db = db
    api.state.data_dir = app_data

    def current_user(session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE)) -> dict:
        if not session_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
        token_hash = hash_session_token(session_token)
        with db.connect() as connection:
            row = connection.execute(
                """
                SELECT users.id, users.email, users.display_name, users.created_at, sessions.expires_at
                FROM sessions JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if not row or datetime.fromisoformat(row["expires_at"]) <= utc_now():
                if row:
                    connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已过期")
            return dict(row)

    def owned_project(project_id: str, user_id: int):
        with db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="项目不存在")
        return row

    def json_object(value: str | None) -> dict | None:
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def resolve_project_file(project_id: str, path_value: str | None) -> Path | None:
        if not path_value:
            return None
        candidate = Path(path_value).resolve()
        project_root = (projects_dir / project_id).resolve()
        if not candidate.is_relative_to(project_root) or not candidate.is_file():
            return None
        return candidate

    def cache_project_thumbnail(project_id: str, metadata: dict) -> dict:
        result = dict(metadata)
        with db.connect() as connection:
            row = connection.execute(
                "SELECT source_metadata FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        previous = json_object(row["source_metadata"]) if row else None
        previous_path = (previous or {}).get("_cover_path")
        thumbnail_url = str(result.get("thumbnail") or "")
        if thumbnail_url:
            try:
                cover_path = cache_bilibili_thumbnail(
                    thumbnail_url, projects_dir / project_id / "source" / "cover.jpg"
                )
                result["_cover_path"] = str(cover_path)
                return result
            except Exception:
                pass
        if resolve_project_file(project_id, previous_path):
            result["_cover_path"] = previous_path
        return result

    def safe_download_stem(value: str, fallback: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" .")
        return cleaned[:100] or fallback

    def project_payload(row) -> dict:
        project = dict(row)
        project["rights_confirmed"] = bool(project["rights_confirmed"])
        project["tempo_locked"] = bool(project.get("tempo_locked"))
        source_metadata = json_object(project["source_metadata"])
        cover_path = source_metadata.pop("_cover_path", None) if source_metadata else None
        project["source_metadata"] = source_metadata
        project["video_analysis"] = json_object(project.get("video_analysis"))
        project["recognition_summary"] = json_object(project.get("recognition_summary"))
        project["audio_analysis"] = json_object(project.get("audio_analysis"))
        project_id = project["id"]
        with db.connect() as connection:
            sync_points = [
                dict(point)
                for point in connection.execute(
                    "SELECT * FROM sync_points WHERE project_id = ? ORDER BY time_seconds", (project_id,)
                ).fetchall()
            ]
            images = [
                dict(asset)
                for asset in connection.execute(
                    """
                    SELECT id, original_name, media_type, sort_order, created_at
                    FROM assets WHERE project_id = ? AND kind = 'score_image'
                    ORDER BY sort_order, created_at
                    """,
                    (project_id,),
                ).fetchall()
            ]
            video_frames = [
                dict(asset)
                for asset in connection.execute(
                    """
                    SELECT id, original_name, media_type, metadata, sort_order, created_at
                    FROM assets WHERE project_id = ? AND kind = 'video_frame'
                    ORDER BY sort_order, created_at
                    """,
                    (project_id,),
                ).fetchall()
            ]
        for image in images:
            image["url"] = f"/api/projects/{project_id}/assets/{image['id']}"
        for frame in video_frames:
            metadata = json_object(frame.pop("metadata", None)) or {}
            frame.update(metadata)
            frame["time_seconds"] = float(frame.get("time_seconds") or 0)
            frame["source_frame"] = int(frame.get("source_frame") or 0)
            frame["url"] = f"/api/projects/{project_id}/assets/{frame['id']}"
        project["sync_points"] = sync_points
        project["score_images"] = images
        project["video_frames"] = video_frames
        project["cover_url"] = (
            f"/api/projects/{project_id}/files/cover?v={project['updated_at']}"
            if resolve_project_file(project_id, cover_path)
            else (source_metadata or {}).get("thumbnail") or None
        )
        project["pdf_url"] = f"/api/projects/{project_id}/files/pdf" if project["score_pdf_path"] else None
        project["audio_url"] = f"/api/projects/{project_id}/files/audio" if project["audio_path"] else None
        project["score_file_url"] = f"/api/projects/{project_id}/files/score" if project["score_file_path"] else None
        project["video_url"] = f"/api/projects/{project_id}/files/video" if project["video_path"] else None
        for internal_path in ("video_path", "score_pdf_path", "score_file_path", "audio_path"):
            project.pop(internal_path, None)
        return project

    def set_session_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=SESSION_DAYS * 24 * 60 * 60,
            httponly=True,
            samesite="lax",
            secure=os.getenv("APP_SECURE_COOKIES") == "1",
            path="/",
        )

    @api.get("/api/health")
    def health() -> dict:
        capabilities = capability_status()
        capabilities.pop("audiveris_path", None)
        return {"status": "ok", "capabilities": capabilities}

    @api.post("/api/auth/register", status_code=201)
    def register(body: RegisterRequest, response: Response) -> dict:
        email = body.email.strip().lower()
        display_name = body.display_name.strip()
        if not EMAIL_PATTERN.match(email):
            raise HTTPException(status_code=422, detail="邮箱格式不正确")
        if not display_name:
            raise HTTPException(status_code=422, detail="请输入昵称")
        now = utc_text()
        try:
            with db.connect() as connection:
                cursor = connection.execute(
                    "INSERT INTO users(email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (email, display_name, hash_password(body.password), now),
                )
                user_id = cursor.lastrowid
                token = new_session_token()
                connection.execute(
                    "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (
                        hash_session_token(token),
                        user_id,
                        now,
                        (utc_now() + timedelta(days=SESSION_DAYS)).isoformat(),
                    ),
                )
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise HTTPException(status_code=409, detail="这个邮箱已经注册") from exc
            raise
        set_session_cookie(response, token)
        return {"id": user_id, "email": email, "display_name": display_name, "created_at": now}

    @api.post("/api/auth/login")
    def login(body: LoginRequest, response: Response) -> dict:
        email = body.email.strip().lower()
        with db.connect() as connection:
            user = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if not user or not verify_password(body.password, user["password_hash"]):
                raise HTTPException(status_code=401, detail="邮箱或密码不正确")
            token = new_session_token()
            now = utc_text()
            connection.execute(
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (
                    hash_session_token(token),
                    user["id"],
                    now,
                    (utc_now() + timedelta(days=SESSION_DAYS)).isoformat(),
                ),
            )
        set_session_cookie(response, token)
        return {"id": user["id"], "email": user["email"], "display_name": user["display_name"]}

    @api.post("/api/auth/logout", status_code=204)
    def logout(
        response: Response,
        session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    ) -> Response:
        if session_token:
            with db.connect() as connection:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_session_token(session_token),))
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @api.get("/api/auth/me")
    def me(user: dict = Depends(current_user)) -> dict:
        return {key: user[key] for key in ("id", "email", "display_name", "created_at")}

    @api.get("/api/projects")
    def list_projects(user: dict = Depends(current_user)) -> list[dict]:
        with db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC", (user["id"],)
            ).fetchall()
        return [project_payload(row) for row in rows]

    @api.post("/api/projects", status_code=201)
    def create_project(body: ProjectCreateRequest, user: dict = Depends(current_user)) -> dict:
        try:
            source = parse_bilibili_source(body.source_input)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        project_id = uuid.uuid4().hex
        now = utc_text()
        title = body.title.strip() or source.source_id
        with db.connect() as connection:
            connection.execute(
                """
                INSERT INTO projects(
                    id, user_id, title, source_input, source_kind, source_id, source_url,
                    rights_confirmed, status, status_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft', '等待上传谱图', ?, ?)
                """,
                (
                    project_id,
                    user["id"],
                    title,
                    body.source_input.strip(),
                    source.kind,
                    source.source_id,
                    source.url,
                    int(body.rights_confirmed),
                    now,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    @api.post("/api/projects/manual-tab", status_code=201)
    def create_manual_tab_project(
        body: ManualTabProjectRequest,
        user: dict = Depends(current_user),
    ) -> dict:
        title = body.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="请输入乐谱名称")
        project_id = uuid.uuid4().hex
        output_dir = projects_dir / project_id / "manual-score"
        try:
            result = create_blank_tab_score(
                output_dir,
                title=title,
                measure_count=body.measure_count,
                tempo_bpm=body.tempo_bpm,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"空白六线谱创建失败：{str(exc)[:240]}") from exc

        now = utc_text()
        with db.connect() as connection:
            connection.execute(
                """
                INSERT INTO projects(
                    id, user_id, title, source_input, source_kind, source_id, source_url,
                    rights_confirmed, status, status_message, recognition_summary,
                    tempo_bpm, tempo_source, tempo_locked,
                    score_file_path, score_file_name, created_at, updated_at
                ) VALUES (?, ?, ?, 'manual://tab', 'manual_tab', '手动六线谱', '',
                    1, 'score_ready', '空白六线谱已创建，可以逐弦打谱', ?, ?, 'default', 0, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    user["id"],
                    title,
                    json.dumps(result.summary, ensure_ascii=False),
                    body.tempo_bpm,
                    str(result.score_path),
                    result.score_path.name,
                    now,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    @api.get("/api/projects/{project_id}")
    def get_project(project_id: str, user: dict = Depends(current_user)) -> dict:
        return project_payload(owned_project(project_id, user["id"]))

    @api.patch("/api/projects/{project_id}")
    def update_project(
        project_id: str, body: ProjectUpdateRequest, user: dict = Depends(current_user)
    ) -> dict:
        owned_project(project_id, user["id"])
        with db.connect() as connection:
            connection.execute(
                "UPDATE projects SET title = ?, updated_at = ? WHERE id = ?",
                (body.title.strip(), utc_text(), project_id),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    @api.patch("/api/projects/{project_id}/rights")
    def update_project_rights(
        project_id: str, body: ProjectRightsRequest, user: dict = Depends(current_user)
    ) -> dict:
        owned_project(project_id, user["id"])
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET rights_confirmed = ?,
                status_message = CASE WHEN ? THEN '已确认处理权限，可以获取视频' ELSE '处理权限确认已撤销' END,
                updated_at = ? WHERE id = ?
                """,
                (int(body.rights_confirmed), int(body.rights_confirmed), utc_text(), project_id),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    @api.patch("/api/projects/{project_id}/tempo")
    def lock_project_tempo(
        project_id: str, body: ProjectTempoRequest, user: dict = Depends(current_user)
    ) -> dict:
        project = owned_project(project_id, user["id"])
        tempo = round(float(body.tempo_bpm), 1)
        summary = json_object(project["recognition_summary"])
        if summary is not None:
            summary["estimated_tempo_bpm"] = tempo
            summary["tempo_source"] = "user"
            summary["tempo_locked"] = True

        score_path = resolve_project_file(project_id, project["score_file_path"])
        if score_path and score_path.suffix.lower() in {".xml", ".musicxml"}:
            try:
                source = score_path.read_text(encoding="utf-8")
                source = re.sub(r"(<per-minute>)[^<]+", rf"\g<1>{tempo:g}", source)
                source = re.sub(r'(<sound\b[^>]*\btempo=")[^"]+', rf"\g<1>{tempo:g}", source)
                temporary = score_path.with_name(f".{score_path.name}.tempo.tmp")
                temporary.write_text(source, encoding="utf-8")
                temporary.replace(score_path)
            except OSError as exc:
                raise HTTPException(status_code=422, detail=f"写入乐谱 BPM 失败：{str(exc)[:180]}") from exc

        if score_path:
            diagnostics_path = score_path.parent / "recognition.json"
            if diagnostics_path.is_file():
                try:
                    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
                    diagnostics_summary = diagnostics.setdefault("summary", {})
                    diagnostics_summary["estimated_tempo_bpm"] = tempo
                    diagnostics_summary["tempo_source"] = "user"
                    diagnostics_summary["tempo_locked"] = True
                    diagnostics_path.write_text(
                        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise HTTPException(status_code=422, detail=f"保存 BPM 诊断数据失败：{str(exc)[:180]}") from exc

        now = utc_text()
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET tempo_bpm = ?, tempo_source = 'user', tempo_locked = 1,
                recognition_summary = ?, audio_analysis = NULL,
                status_message = ?, updated_at = ? WHERE id = ?
                """,
                (
                    tempo,
                    json.dumps(summary, ensure_ascii=False) if summary is not None else None,
                    f"项目速度已锁定为 {tempo:g} BPM；后续识别、分析与播放均使用此值",
                    now,
                    project_id,
                ),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    @api.post("/api/projects/{project_id}/inspect")
    async def inspect_source(project_id: str, user: dict = Depends(current_user)) -> dict:
        project = owned_project(project_id, user["id"])
        if not capability_status()["yt_dlp"]:
            raise HTTPException(status_code=503, detail="yt-dlp 未安装，暂时不能解析视频信息")
        try:
            metadata = await run_in_threadpool(inspect_bilibili, project["source_url"])
            metadata = await run_in_threadpool(cache_project_thumbnail, project_id, metadata)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"B 站解析失败：{str(exc)[:240]}") from exc
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET source_metadata = ?, title = CASE WHEN title = source_id THEN ? ELSE title END,
                status_message = '视频信息已解析', updated_at = ? WHERE id = ?
                """,
                (json.dumps(metadata, ensure_ascii=False), metadata["title"], utc_text(), project_id),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    def download_video_job(project_id: str, source_url: str, project_folder: Path) -> None:
        try:
            import yt_dlp

            project_folder.mkdir(parents=True, exist_ok=True)
            options = {
                "format": "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[ext=mp4][vcodec^=avc1]/bv*+ba/b",
                "outtmpl": str(project_folder / "source.%(ext)s"),
                "merge_output_format": "mp4",
                "noplaylist": True,
                "restrictfilenames": True,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(source_url, download=True)
            candidates = [
                path
                for path in project_folder.glob("source.*")
                if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".flv"} and path.is_file()
            ]
            if not candidates:
                raise RuntimeError("下载完成但没有找到视频文件")
            video_path = max(candidates, key=lambda path: path.stat().st_size)
            metadata = {
                "id": str(info.get("id") or ""),
                "title": str(info.get("title") or "未命名视频")[:200],
                "uploader": str(info.get("uploader") or "")[:120],
                "duration": float(info.get("duration") or 0),
                "thumbnail": str(info.get("thumbnail") or ""),
                "webpage_url": str(info.get("webpage_url") or source_url),
                "extractor": str(info.get("extractor_key") or "BiliBili"),
            }
            metadata = cache_project_thumbnail(project_id, metadata)
            with db.connect() as connection:
                connection.execute(
                    """
                    UPDATE projects SET video_path = ?, source_metadata = ?, status = 'video_ready',
                    status_message = '视频已准备，等待谱面提取', updated_at = ? WHERE id = ?
                    """,
                    (str(video_path), json.dumps(metadata, ensure_ascii=False), utc_text(), project_id),
                )
        except Exception as exc:
            with db.connect() as connection:
                connection.execute(
                    "UPDATE projects SET status = 'failed', status_message = ?, updated_at = ? WHERE id = ?",
                    (f"视频下载失败：{str(exc)[:300]}", utc_text(), project_id),
                )

    @api.post("/api/projects/{project_id}/download", status_code=202)
    def download_source(
        project_id: str,
        background_tasks: BackgroundTasks,
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        if not project["rights_confirmed"]:
            raise HTTPException(status_code=403, detail="请先确认你有权处理该视频")
        if project["status"] in {"downloading", "analyzing"}:
            raise HTTPException(status_code=409, detail="当前项目正在处理，请等待完成")
        capabilities = capability_status()
        if not capabilities["yt_dlp"] or not capabilities["ffmpeg"]:
            raise HTTPException(status_code=503, detail="需要安装 yt-dlp 和 FFmpeg")
        with db.connect() as connection:
            connection.execute(
                "UPDATE projects SET status = 'downloading', status_message = '正在下载公开视频…', updated_at = ? WHERE id = ?",
                (utc_text(), project_id),
            )
        background_tasks.add_task(
            download_video_job, project_id, project["source_url"], projects_dir / project_id / "video"
        )
        return {"status": "downloading", "message": "下载任务已开始"}

    def analyze_video_job(
        project_id: str,
        video_path: str,
        config: dict,
        output_dir: Path,
        fallback_status: str,
    ) -> None:
        analysis_id = config["analysis_id"]
        try:
            probe, frames = extract_video_frames(
                Path(video_path),
                output_dir,
                start_seconds=config["start_seconds"],
                end_seconds=config["end_seconds"],
                frame_interval=config["frame_interval"],
                crop_x=config["crop_x"],
                crop_y=config["crop_y"],
                crop_width=config["crop_width"],
                crop_height=config["crop_height"],
            )
            preview_pdf_path: Path | None = None
            preview_pdf_error: str | None = None
            try:
                preview_pdf_path = build_slice_preview_pdf(
                    [frame.path for frame in frames],
                    output_dir / "slice-score-preview.pdf",
                )
            except (OSError, ValueError) as exc:
                preview_pdf_error = str(exc)[:240]
            capabilities = capability_status()
            detected_tempo = (
                detect_tempo_from_images(
                    [frame.path for frame in frames],
                    tesseract_path=capabilities["tesseract_path"],
                )
                if capabilities["tab_ocr"]
                else None
            )
            now = utc_text()
            completed = {
                **config,
                "status": "complete",
                "source_fps": round(probe.fps, 5),
                "source_width": probe.width,
                "source_height": probe.height,
                "frame_count": len(frames),
                "preview_pdf_status": "complete" if preview_pdf_path else "failed",
                "preview_pdf_error": preview_pdf_error,
                "detected_tempo_bpm": detected_tempo,
                "completed_at": now,
            }
            asset_rows = [
                (
                    uuid.uuid4().hex,
                    project_id,
                    "video_frame",
                    f"切片 {index + 1:03d} · {frame.time_seconds:.3f}s.jpg",
                    str(frame.path),
                    "image/jpeg",
                    json.dumps(
                        {
                            "time_seconds": round(frame.time_seconds, 3),
                            "source_frame": frame.source_frame,
                        },
                        ensure_ascii=False,
                    ),
                    index,
                    now,
                )
                for index, frame in enumerate(frames)
            ]
            with db.connect() as connection:
                row = connection.execute(
                    "SELECT video_analysis, score_pdf_path, score_file_path FROM projects WHERE id = ?",
                    (project_id,),
                ).fetchone()
                current = json_object(row["video_analysis"]) if row else None
                if not row or not current or current.get("analysis_id") != analysis_id:
                    return
                connection.execute(
                    "DELETE FROM assets WHERE project_id = ? AND kind = 'video_frame'",
                    (project_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO assets(
                        id, project_id, kind, original_name, stored_path, media_type, metadata,
                        sort_order, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    asset_rows,
                )
                pdf_path = str(preview_pdf_path) if preview_pdf_path else row["score_pdf_path"]
                next_status = "score_ready" if row["score_file_path"] else "pdf_ready" if pdf_path else "frames_ready"
                message = f"切片分析完成，共生成 {len(frames)} 张候选帧"
                if preview_pdf_path:
                    message += "；已自动裁边并生成预览 PDF"
                elif preview_pdf_error:
                    message += f"；PDF 生成失败：{preview_pdf_error}"
                connection.execute(
                    """
                    UPDATE projects SET video_analysis = ?, score_pdf_path = ?, status = ?,
                    tempo_bpm = CASE WHEN tempo_locked = 1 OR ? IS NULL THEN tempo_bpm ELSE ? END,
                    tempo_source = CASE WHEN tempo_locked = 1 OR ? IS NULL THEN tempo_source ELSE 'visual_ocr' END,
                    status_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(completed, ensure_ascii=False),
                        pdf_path,
                        next_status,
                        detected_tempo,
                        detected_tempo,
                        detected_tempo,
                        message,
                        now,
                        project_id,
                    ),
                )
        except Exception as exc:
            with db.connect() as connection:
                row = connection.execute(
                    "SELECT video_analysis FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
                current = json_object(row["video_analysis"]) if row else None
                if current and current.get("analysis_id") == analysis_id:
                    failed = {**current, "status": "failed", "error": str(exc)[:300]}
                    connection.execute(
                        """
                        UPDATE projects SET video_analysis = ?, status = ?, status_message = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            json.dumps(failed, ensure_ascii=False),
                            fallback_status,
                            f"切片分析失败：{str(exc)[:300]}",
                            utc_text(),
                            project_id,
                        ),
                    )

    @api.post("/api/projects/{project_id}/video-analysis", status_code=202)
    def analyze_video(
        project_id: str,
        body: VideoSliceRequest,
        background_tasks: BackgroundTasks,
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        if not project["video_path"] or not Path(project["video_path"]).is_file():
            raise HTTPException(status_code=409, detail="请先获取视频，再选择范围进行分析")
        if project["status"] in {"downloading", "analyzing", "recognizing"}:
            raise HTTPException(status_code=409, detail="当前项目正在处理，请等待完成")
        if not capability_status()["ffmpeg"]:
            raise HTTPException(status_code=503, detail="需要安装 FFmpeg 和 FFprobe 才能切片")
        if body.end_seconds - body.start_seconds < 0.25:
            raise HTTPException(status_code=422, detail="结束时间必须比开始时间至少晚 0.25 秒")

        try:
            probe = probe_video(Path(project["video_path"]))
            if body.start_seconds >= probe.duration:
                raise ValueError("开始时间不能晚于视频结尾")
            if body.end_seconds > probe.duration + 0.1:
                raise ValueError(f"结束时间不能超过视频时长 {probe.duration:.2f} 秒")
            if body.end_seconds - body.start_seconds > MAX_ANALYSIS_SECONDS:
                raise ValueError("单次分析最多选择 20 分钟，请分段处理")
            normalized_crop_to_pixels(
                probe, body.crop_x, body.crop_y, body.crop_width, body.crop_height
            )
            estimated = estimate_frame_count(
                body.start_seconds, min(body.end_seconds, probe.duration), probe.fps, body.frame_interval
            )
            if estimated > MAX_ANALYSIS_FRAMES:
                raise ValueError(
                    f"预计生成 {estimated} 张，最多允许 {MAX_ANALYSIS_FRAMES} 张；请缩短时间或增大抽帧间隔"
                )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        analysis_id = uuid.uuid4().hex
        config = {
            **body.model_dump(),
            "analysis_id": analysis_id,
            "status": "pending",
            "source_fps": round(probe.fps, 5),
            "source_width": probe.width,
            "source_height": probe.height,
            "estimated_frames": estimated,
            "created_at": utc_text(),
        }
        fallback_status = (
            project["status"]
            if project["status"] in {"video_ready", "frames_ready", "pdf_ready", "score_ready"}
            else "video_ready"
        )
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET video_analysis = ?, status = 'analyzing',
                status_message = ?, updated_at = ? WHERE id = ?
                """,
                (
                    json.dumps(config, ensure_ascii=False),
                    f"正在按每 {body.frame_interval} 帧提取谱面候选图…",
                    utc_text(),
                    project_id,
                ),
            )
        background_tasks.add_task(
            analyze_video_job,
            project_id,
            project["video_path"],
            config,
            projects_dir / project_id / "analysis" / analysis_id,
            fallback_status,
        )
        return {
            "status": "analyzing",
            "message": "切片分析任务已开始",
            "estimated_frames": estimated,
            "source_fps": round(probe.fps, 5),
        }

    @api.post("/api/projects/{project_id}/score-images")
    async def upload_score_images(
        project_id: str,
        files: Annotated[list[UploadFile], File(description="谱面图片")],
        user: dict = Depends(current_user),
    ) -> dict:
        owned_project(project_id, user["id"])
        if not files or len(files) > 30:
            raise HTTPException(status_code=422, detail="一次请上传 1～30 张谱图")

        upload_id = uuid.uuid4().hex
        target_dir = projects_dir / project_id / "score" / upload_id
        target_dir.mkdir(parents=True, exist_ok=True)
        converted_pages: list[Image.Image] = []
        asset_rows: list[tuple] = []
        now = utc_text()
        try:
            for index, upload in enumerate(files):
                content = await upload.read(MAX_IMAGE_BYTES + 1)
                if len(content) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail=f"{upload.filename} 超过 15 MB")
                try:
                    with Image.open(io.BytesIO(content)) as source_image:
                        source_image.verify()
                    with Image.open(io.BytesIO(content)) as source_image:
                        oriented_image = ImageOps.exif_transpose(source_image)
                        if oriented_image.width * oriented_image.height > 40_000_000:
                            raise HTTPException(status_code=413, detail=f"{upload.filename} 的像素尺寸过大")
                        page = oriented_image.convert("RGB")
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                    raise HTTPException(status_code=422, detail=f"{upload.filename} 不是有效图片") from exc
                suffix = ".jpg"
                stored_path = target_dir / f"{index + 1:03d}{suffix}"
                page.save(stored_path, "JPEG", quality=94, optimize=True)
                converted_pages.append(page.copy())
                asset_rows.append(
                    (
                        uuid.uuid4().hex,
                        project_id,
                        "score_image",
                        upload.filename or f"第 {index + 1} 页",
                        str(stored_path),
                        "image/jpeg",
                        index,
                        now,
                    )
                )

            pdf_path = target_dir / "score.pdf"
            first_page, *other_pages = converted_pages
            first_page.save(pdf_path, "PDF", save_all=True, append_images=other_pages, resolution=180)
            with db.connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO assets(id, project_id, kind, original_name, stored_path, media_type, sort_order, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    asset_rows,
                )
                connection.execute(
                    """
                    UPDATE projects SET score_pdf_path = ?, status = 'pdf_ready',
                    status_message = 'PDF 已生成，等待乐谱识别', updated_at = ? WHERE id = ?
                    """,
                    (str(pdf_path), utc_text(), project_id),
                )
                row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        finally:
            for page in converted_pages:
                page.close()
        return project_payload(row)

    @api.post("/api/projects/{project_id}/score-pdf", status_code=202)
    async def upload_score_pdf(
        project_id: str,
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File(description="PDF 谱图")],
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        if Path(file.filename or "").suffix.lower() != ".pdf":
            raise HTTPException(status_code=422, detail="请选择 PDF 乐谱文件")
        content = await read_bounded(file, MAX_SCORE_BYTES)
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail="文件不是有效 PDF")

        upload_id = uuid.uuid4().hex
        target_dir = projects_dir / project_id / "pdf-score" / upload_id
        target_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = target_dir / "source-score.pdf"
        pdf_path.write_bytes(content)
        try:
            imported = await run_in_threadpool(
                render_and_segment_pdf,
                pdf_path,
                target_dir / "rendered",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"PDF 分析失败：{str(exc)[:240]}") from exc

        capabilities = capability_status()
        page_rows = []
        system_rows = []
        now = utc_text()
        for page in imported.pages:
            page_rows.append(
                (
                    uuid.uuid4().hex,
                    project_id,
                    "score_image",
                    f"{file.filename or 'PDF 乐谱'} · 第 {page.page_number} 页",
                    str(page.path),
                    "image/jpeg",
                    json.dumps({"page_number": page.page_number, "width": page.width, "height": page.height}),
                    page.page_number - 1,
                    now,
                )
            )
        tab_frames: list[TabFrameInput] = []
        for index, system in enumerate(imported.systems):
            sequence_seconds = float(index * 4)
            metadata = {
                "page_number": system.page_number,
                "system_number": system.system_number,
                "layout": system.layout,
                "polarity": system.polarity,
                "time_seconds": sequence_seconds,
                "source_frame": index + 1,
                "source_kind": "pdf_system",
            }
            system_rows.append(
                (
                    uuid.uuid4().hex,
                    project_id,
                    "score_system",
                    f"第 {system.page_number} 页 · 谱行 {system.system_number}",
                    str(system.path),
                    "image/jpeg",
                    json.dumps(metadata, ensure_ascii=False),
                    index,
                    now,
                )
            )
            tab_frames.append(TabFrameInput(system.path, sequence_seconds, index + 1))

        preliminary_summary = {
            "engine": "pdf_layout",
            "engine_label": "PDF 谱面布局分析",
            "page_count": len(imported.pages),
            "system_count": len(imported.systems),
            "layout_counts": imported.layout_counts,
            "warnings": ["PDF 已按谱行切分；自动识别结果仍需逐小节试听校对"],
        }
        if tab_frames and capabilities["tab_ocr"]:
            next_status = "recognizing"
            status_message = f"已拆分 {len(imported.pages)} 页、{len(tab_frames)} 行谱，正在识别品位、节奏与技巧…"
        elif not tab_frames and capabilities["audiveris"]:
            next_status = "recognizing"
            status_message = f"已拆分 {len(imported.pages)} 页，正在识别印刷五线谱…"
        elif tab_frames:
            next_status = "pdf_ready"
            status_message = f"已拆分 {len(imported.pages)} 页、{len(tab_frames)} 行 TAB；安装 Tesseract 后可自动识别"
        else:
            next_status = "pdf_ready"
            status_message = f"已拆分 {len(imported.pages)} 页；未找到六线谱，安装 Audiveris 后可识别五线谱"

        with db.connect() as connection:
            connection.execute(
                "DELETE FROM assets WHERE project_id = ? AND kind IN ('score_image', 'score_system')",
                (project_id,),
            )
            connection.executemany(
                """
                INSERT INTO assets(id, project_id, kind, original_name, stored_path, media_type,
                    metadata, sort_order, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                page_rows + system_rows,
            )
            connection.execute(
                """
                UPDATE projects SET score_pdf_path = ?, score_file_path = NULL, score_file_name = NULL,
                    recognition_summary = ?, status = ?, status_message = ?, updated_at = ? WHERE id = ?
                """,
                (
                    str(pdf_path),
                    json.dumps(preliminary_summary, ensure_ascii=False),
                    next_status,
                    status_message,
                    now,
                    project_id,
                ),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()

        output_dir = projects_dir / project_id / "recognition" / uuid.uuid4().hex
        if tab_frames and capabilities["tab_ocr"]:
            background_tasks.add_task(
                recognize_tab_job,
                project_id,
                project["title"],
                tab_frames,
                capabilities["tesseract_path"],
                output_dir,
                False,
            )
        elif not tab_frames and capabilities["audiveris"]:
            background_tasks.add_task(
                recognize_staff_job,
                project_id,
                str(pdf_path),
                capabilities["audiveris_path"],
                output_dir,
            )
        return project_payload(row)

    async def read_bounded(upload: UploadFile, limit: int) -> bytes:
        content = await upload.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(status_code=413, detail="文件过大")
        return content

    @api.post("/api/projects/{project_id}/audio")
    async def upload_audio(
        project_id: str,
        file: Annotated[UploadFile, File(description="练习音频")],
        user: dict = Depends(current_user),
    ) -> dict:
        owned_project(project_id, user["id"])
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise HTTPException(status_code=422, detail="支持 MP3、WAV、M4A、AAC、OGG 或 FLAC")
        content = await read_bounded(file, MAX_AUDIO_BYTES)
        target_dir = projects_dir / project_id / "audio"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{uuid.uuid4().hex}{suffix}"
        target.write_bytes(content)
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET audio_path = ?, audio_name = ?, audio_analysis = NULL,
                status_message = '练习音频已保存；原同步点仍保留，请重新分析或手动校对',
                updated_at = ? WHERE id = ?
                """,
                (str(target), file.filename or target.name, utc_text(), project_id),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    def audio_analysis_job(
        project_id: str,
        source_path: str,
        source_kind: str,
        source_label: str,
        score_summary: dict | None,
        visual_sync: list[dict],
        started_at: str,
    ) -> None:
        try:
            analysis = analyze_audio_file(
                Path(source_path),
                source_kind=source_kind,
                ffmpeg_path=shutil.which("ffmpeg") or "ffmpeg",
                score_summary=score_summary,
                visual_sync=visual_sync,
            )
            analysis.update(
                {
                    "status": "complete",
                    "source_label": source_label,
                    "started_at": started_at,
                    "completed_at": utc_text(),
                }
            )
            message = (
                f"音频分析完成：约 {analysis['tempo_bpm']:.1f} BPM，"
                f"得到 {len(analysis['sections'])} 个段落候选和 "
                f"{len(analysis['alignment_suggestions'])} 个对齐建议"
            )
            with db.connect() as connection:
                connection.execute(
                    """
                    UPDATE projects SET audio_analysis = ?,
                    tempo_bpm = CASE
                        WHEN tempo_locked = 0 AND COALESCE(tempo_source, 'default') IN ('default', 'video_timing', 'audio_analysis') THEN ?
                        ELSE tempo_bpm END,
                    tempo_source = CASE
                        WHEN tempo_locked = 0 AND COALESCE(tempo_source, 'default') IN ('default', 'video_timing', 'audio_analysis') THEN 'audio_analysis'
                        ELSE tempo_source END,
                    status_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(analysis, ensure_ascii=False),
                        float(analysis["tempo_bpm"]),
                        message,
                        utc_text(),
                        project_id,
                    ),
                )
        except Exception as exc:
            failed = {
                "status": "failed",
                "source": source_kind,
                "source_label": source_label,
                "started_at": started_at,
                "error": str(exc)[:400],
            }
            with db.connect() as connection:
                connection.execute(
                    """
                    UPDATE projects SET audio_analysis = ?, status_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(failed, ensure_ascii=False),
                        f"音频分析失败：{str(exc)[:300]}",
                        utc_text(),
                        project_id,
                    ),
                )

    @api.post("/api/projects/{project_id}/audio-analysis", status_code=202)
    def start_audio_analysis(
        project_id: str,
        background_tasks: BackgroundTasks,
        source: str = "auto",
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        current = json_object(project["audio_analysis"])
        if current and current.get("status") == "pending":
            raise HTTPException(status_code=409, detail="当前音频正在分析，请等待完成")
        if not capability_status()["audio_analysis"]:
            raise HTTPException(status_code=503, detail="音频分析需要 FFmpeg")

        if source not in {"auto", "uploaded", "video"}:
            raise HTTPException(status_code=422, detail="音频来源必须是 auto、uploaded 或 video")

        if source in {"auto", "uploaded"} and project["audio_path"] and Path(project["audio_path"]).is_file():
            source_path = project["audio_path"]
            source_kind = "uploaded_audio"
            source_label = project["audio_name"] or "用户上传音频"
        elif source in {"auto", "video"} and project["video_path"] and Path(project["video_path"]).is_file():
            source_path = project["video_path"]
            source_kind = "video_audio"
            source_label = "视频原声"
        else:
            detail = "没有可分析的上传音频" if source == "uploaded" else "没有可分析的视频原声" if source == "video" else "请先获取视频或上传一段练习音频"
            raise HTTPException(status_code=409, detail=detail)

        score_summary = json_object(project["recognition_summary"])
        if project["tempo_locked"] and project["tempo_bpm"] is not None:
            score_summary = {
                **(score_summary or {}),
                "estimated_tempo_bpm": float(project["tempo_bpm"]),
                "tempo_source": "user",
                "tempo_locked": True,
            }
        visual_sync: list[dict] = []
        if source_kind == "video_audio" and project["score_file_path"]:
            diagnostics_path = Path(project["score_file_path"]).parent / "recognition.json"
            if diagnostics_path.is_file():
                try:
                    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
                    candidate_sync = diagnostics.get("sync_suggestions") or []
                    if isinstance(candidate_sync, list):
                        visual_sync = [item for item in candidate_sync if isinstance(item, dict)]
                except (OSError, json.JSONDecodeError):
                    visual_sync = []

        started_at = utc_text()
        pending = {
            "status": "pending",
            "source": source_kind,
            "source_label": source_label,
            "started_at": started_at,
        }
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET audio_analysis = ?, status_message = '正在分析节拍、起音和段落候选…',
                updated_at = ? WHERE id = ?
                """,
                (json.dumps(pending, ensure_ascii=False), started_at, project_id),
            )
        background_tasks.add_task(
            audio_analysis_job,
            project_id,
            source_path,
            source_kind,
            source_label,
            score_summary,
            visual_sync,
            started_at,
        )
        return {"status": "pending", "message": "音频分析任务已开始", "source": source_kind}

    @api.post("/api/projects/{project_id}/audio-analysis/apply")
    def apply_audio_alignment(
        project_id: str,
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        analysis = json_object(project["audio_analysis"])
        if not analysis or analysis.get("status") != "complete":
            raise HTTPException(status_code=409, detail="请先完成音频分析")
        suggestions = analysis.get("alignment_suggestions") or []
        if len(suggestions) < 2:
            raise HTTPException(status_code=409, detail="当前音频没有足够的自动对齐建议")

        now = utc_text()
        inserted = 0
        with db.connect() as connection:
            existing_measures = {
                int(row["measure_number"])
                for row in connection.execute(
                    "SELECT measure_number FROM sync_points WHERE project_id = ?",
                    (project_id,),
                ).fetchall()
            }
            for suggestion in suggestions:
                measure_number = int(suggestion["measure_number"])
                if measure_number in existing_measures:
                    continue
                connection.execute(
                    """
                    INSERT INTO sync_points(
                        project_id, measure_number, time_seconds, score_position, label, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        measure_number,
                        round(float(suggestion["time_seconds"]), 3),
                        round(float(suggestion["score_position"]), 5),
                        str(suggestion.get("label") or "自动对齐")[:80],
                        now,
                    ),
                )
                existing_measures.add(measure_number)
                inserted += 1
            connection.execute(
                "UPDATE projects SET status_message = ?, updated_at = ? WHERE id = ?",
                (f"已加入 {inserted} 个自动对齐点；原手动同步点未被覆盖", now, project_id),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    @api.post("/api/projects/{project_id}/score-file")
    async def upload_score_file(
        project_id: str,
        file: Annotated[UploadFile, File(description="MusicXML 或 Guitar Pro 谱")],
        user: dict = Depends(current_user),
    ) -> dict:
        owned_project(project_id, user["id"])
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SCORE_EXTENSIONS:
            raise HTTPException(status_code=422, detail="支持 MusicXML、MXL 和 Guitar Pro GP3～GP8 文件")
        content = await read_bounded(file, MAX_SCORE_BYTES)
        target_dir = projects_dir / project_id / "structured-score"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{uuid.uuid4().hex}{suffix}"
        target.write_bytes(content)
        detected_tempo = score_tempo_candidate(target)
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET score_file_path = ?, score_file_name = ?, status = 'score_ready',
                recognition_summary = NULL,
                tempo_bpm = CASE WHEN tempo_locked = 1 OR ? IS NULL THEN tempo_bpm ELSE ? END,
                tempo_source = CASE WHEN tempo_locked = 1 OR ? IS NULL THEN tempo_source ELSE 'score' END,
                status_message = '结构化乐谱已准备，可在线试听',
                updated_at = ? WHERE id = ?
                """,
                (
                    str(target),
                    file.filename or target.name,
                    detected_tempo,
                    detected_tempo,
                    detected_tempo,
                    utc_text(),
                    project_id,
                ),
            )
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return project_payload(row)

    def recognize_staff_job(project_id: str, pdf_path: str, audiveris_path: str, output_dir: Path) -> None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [audiveris_path, "-batch", "-transcribe", "-export", "-output", str(output_dir), "--", pdf_path],
                capture_output=True,
                text=True,
                timeout=30 * 60,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout or "Audiveris 执行失败")[-500:])
            score_files = list(output_dir.rglob("*.mxl")) + list(output_dir.rglob("*.musicxml"))
            if not score_files:
                raise RuntimeError("Audiveris 没有生成 MusicXML 文件")
            score_path = max(score_files, key=lambda path: path.stat().st_mtime)
            detected_tempo = score_tempo_candidate(score_path)
            with db.connect() as connection:
                connection.execute(
                    """
                    UPDATE projects SET score_file_path = ?, score_file_name = ?, status = 'score_ready',
                    recognition_summary = ?,
                    tempo_bpm = CASE WHEN tempo_locked = 1 OR ? IS NULL THEN tempo_bpm ELSE ? END,
                    tempo_source = CASE WHEN tempo_locked = 1 OR ? IS NULL THEN tempo_source ELSE 'score' END,
                    status_message = '五线谱识别完成，请逐小节校对',
                    updated_at = ? WHERE id = ?
                    """,
                    (
                        str(score_path),
                        score_path.name,
                        json.dumps(
                            {
                                "engine": "audiveris",
                                "engine_label": "Audiveris 五线谱识别",
                                "warnings": ["自动 OMR 结果需要逐小节校对"],
                            },
                            ensure_ascii=False,
                        ),
                        detected_tempo,
                        detected_tempo,
                        detected_tempo,
                        utc_text(),
                        project_id,
                    ),
                )
        except Exception as exc:
            with db.connect() as connection:
                connection.execute(
                    "UPDATE projects SET status = 'failed', status_message = ?, updated_at = ? WHERE id = ?",
                    (f"识别失败：{str(exc)[:400]}", utc_text(), project_id),
                )

    def recognize_tab_job(
        project_id: str,
        title: str,
        frames: list[TabFrameInput],
        tesseract_path: str,
        output_dir: Path,
        replace_pdf: bool = True,
        recognition_mode: Literal["ocr", "ai"] = "ocr",
    ) -> None:
        try:
            with db.connect() as connection:
                tempo_row = connection.execute(
                    "SELECT tempo_bpm, tempo_locked FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
            locked_tempo = (
                float(tempo_row["tempo_bpm"])
                if tempo_row and tempo_row["tempo_locked"] and tempo_row["tempo_bpm"] is not None
                else None
            )
            last_progress = 0

            def update_progress(completed: int, total: int) -> None:
                nonlocal last_progress
                if completed != total and completed - last_progress < 4:
                    return
                last_progress = completed
                label = "本地 AI + OCR" if recognition_mode == "ai" else "OCR"
                with db.connect() as progress_connection:
                    progress_connection.execute(
                        "UPDATE projects SET status_message = ?, updated_at = ? WHERE id = ?",
                        (f"{label} 正在写入可编辑谱面：{completed}/{total} 张切片", utc_text(), project_id),
                    )

            result = recognize_tab_frames(
                frames,
                output_dir,
                title=title,
                tesseract_path=tesseract_path,
                tempo_bpm=locked_tempo,
                recognition_mode=recognition_mode,
                vision_model_url=os.getenv("NOCTURNE_VISION_MODEL"),
                progress_callback=update_progress,
            )
            summary = result.summary
            with db.connect() as connection:
                existing_row = connection.execute(
                    "SELECT recognition_summary FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
            existing_summary = json_object(existing_row["recognition_summary"]) if existing_row else None
            if (existing_summary or {}).get("engine") == "pdf_layout":
                summary["page_count"] = existing_summary.get("page_count")
                summary["system_count"] = existing_summary.get("system_count")
                summary["source_kind"] = "pdf"
                summary["engine_label"] = "PDF 六线 TAB 专用识别（Beta）"
            recognized_tempo = summary.get("estimated_tempo_bpm")
            recognized_tempo = float(recognized_tempo) if recognized_tempo is not None else None
            recognized_tempo_source = str(summary.get("tempo_source") or "recognition")
            message = (
                f"已识别 {summary['measure_count']} 小节六线 TAB；"
                "已按小节号去重并合成完整 PDF，请在保存前逐小节试听校对"
            )
            with db.connect() as connection:
                connection.execute(
                    """
                    UPDATE projects SET score_file_path = ?, score_file_name = ?, status = 'score_ready',
                    score_pdf_path = COALESCE(?, score_pdf_path), recognition_summary = ?,
                    tempo_bpm = CASE WHEN tempo_locked = 1 OR ? IS NULL THEN tempo_bpm ELSE ? END,
                    tempo_source = CASE WHEN tempo_locked = 1 OR ? IS NULL THEN tempo_source ELSE ? END,
                    status_message = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        str(result.score_path),
                        result.score_path.name,
                        str(result.pdf_path) if replace_pdf and result.pdf_path else None,
                        json.dumps(summary, ensure_ascii=False),
                        recognized_tempo,
                        recognized_tempo,
                        recognized_tempo,
                        recognized_tempo_source,
                        message,
                        utc_text(),
                        project_id,
                    ),
                )
        except Exception as exc:
            with db.connect() as connection:
                connection.execute(
                    "UPDATE projects SET status = 'failed', status_message = ?, updated_at = ? WHERE id = ?",
                    (f"TAB 识别失败：{str(exc)[:400]}", utc_text(), project_id),
                )

    @api.post("/api/projects/{project_id}/recognize", status_code=202)
    def recognize_score(
        project_id: str,
        background_tasks: BackgroundTasks,
        mode: Literal["ocr", "ai"] = "ocr",
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        if project["status"] == "recognizing":
            raise HTTPException(status_code=409, detail="当前项目正在识别，请等待完成")
        capabilities = capability_status()
        if mode == "ai" and not capabilities["ai_tab_recognition"]:
            raise HTTPException(status_code=503, detail="本机 4060 AI 品位服务尚未连接")
        with db.connect() as connection:
            video_rows = connection.execute(
                """
                SELECT stored_path, metadata, sort_order FROM assets
                WHERE project_id = ? AND kind = 'video_frame'
                ORDER BY sort_order, created_at
                """,
                (project_id,),
            ).fetchall()
            system_rows = connection.execute(
                """
                SELECT stored_path, metadata, sort_order FROM assets
                WHERE project_id = ? AND kind = 'score_system'
                ORDER BY sort_order, created_at
                """,
                (project_id,),
            ).fetchall()
        frame_rows = video_rows or system_rows
        tab_frames: list[TabFrameInput] = []
        for frame_row in frame_rows:
            metadata = json_object(frame_row["metadata"]) or {}
            path = Path(frame_row["stored_path"])
            if path.is_file():
                tab_frames.append(
                    TabFrameInput(
                        path=path,
                        time_seconds=float(metadata.get("time_seconds") or 0),
                        source_frame=int(metadata.get("source_frame") or frame_row["sort_order"]),
                    )
                )

        if tab_frames and capabilities["tab_ocr"]:
            engine = "tablature"
            status_message = (
                "正在连接本机 4060，用 AI + OCR 识别品位、弦号和节奏…"
                if mode == "ai"
                else "正在识别六线 TAB 的品位、弦号和节奏网格…"
            )
        elif project["score_pdf_path"] and capabilities["audiveris"]:
            engine = "staff"
            status_message = "正在用 Audiveris 识别印刷五线谱…"
        elif tab_frames:
            raise HTTPException(status_code=503, detail="六线 TAB 识别需要 OpenCV 和 Tesseract OCR")
        elif project["score_pdf_path"]:
            raise HTTPException(status_code=503, detail="五线谱 PDF 识别需要安装 Audiveris")
        else:
            raise HTTPException(status_code=409, detail="请先生成视频切片，或上传谱图并生成 PDF")
        with db.connect() as connection:
            connection.execute(
                "UPDATE projects SET status = 'recognizing', status_message = ?, updated_at = ? WHERE id = ?",
                (status_message, utc_text(), project_id),
            )
        output_dir = projects_dir / project_id / "recognition" / uuid.uuid4().hex
        if engine == "tablature":
            background_tasks.add_task(
                recognize_tab_job,
                project_id,
                project["title"],
                tab_frames,
                capabilities["tesseract_path"],
                output_dir,
                bool(video_rows),
                mode,
            )
        else:
            background_tasks.add_task(
                recognize_staff_job,
                project_id,
                project["score_pdf_path"],
                capabilities["audiveris_path"],
                output_dir,
            )
        return {"status": "recognizing", "message": "识别任务已开始", "engine": engine, "mode": mode}

    def tab_diagnostics_for_project(project) -> tuple[Path, Path, dict]:
        summary = json_object(project["recognition_summary"])
        if not summary or summary.get("engine") not in {"tab_cv_tesseract", "tab_cv_ai", "tab_manual_editor"}:
            raise HTTPException(status_code=409, detail="当前项目没有可编辑的 TAB 识别结果")
        if not project["score_file_path"]:
            raise HTTPException(status_code=404, detail="识别乐谱文件不存在")
        score_path = Path(project["score_file_path"])
        diagnostics_path = score_path.parent / "recognition.json"
        if not score_path.is_file() or not diagnostics_path.is_file():
            raise HTTPException(status_code=404, detail="识别诊断文件不存在，请重新识别")
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="识别诊断文件损坏") from exc
        return score_path, diagnostics_path, diagnostics

    @api.get("/api/projects/{project_id}/recognition")
    def get_recognition_diagnostics(
        project_id: str,
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        _, _, diagnostics = tab_diagnostics_for_project(project)
        return diagnostics

    @api.post("/api/projects/{project_id}/recognition/measures/{measure_number}/retry")
    def retry_recognized_measure(
        project_id: str,
        measure_number: int,
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        if project["status"] in {"downloading", "analyzing", "recognizing"}:
            raise HTTPException(status_code=409, detail="当前项目正在处理，请稍后再试")
        summary = json_object(project["recognition_summary"])
        if (summary or {}).get("engine") == "tab_manual_editor":
            raise HTTPException(status_code=409, detail="手动六线谱没有源视频帧，不能重新识别")
        capabilities = capability_status()
        if not capabilities["tab_ocr"]:
            raise HTTPException(status_code=503, detail="重新识别需要 OpenCV 和 Tesseract OCR")
        score_path, _, diagnostics = tab_diagnostics_for_project(project)
        frame_candidates = []
        for frame in diagnostics.get("frames") or []:
            start = frame.get("start_measure")
            count = max(1, len(frame.get("raw_measure_labels") or []))
            if isinstance(start, int) and start <= measure_number < start + count:
                frame_candidates.append(frame)
        if not frame_candidates:
            raise HTTPException(status_code=409, detail="识别记录中没有覆盖这个小节的源帧")
        current_measure = next(
            (row for row in diagnostics.get("measures") or [] if int(row.get("number") or 0) == measure_number),
            None,
        )
        target_time = float((current_measure or {}).get("source_time") or frame_candidates[0].get("time_seconds") or 0)
        frame_info = min(
            frame_candidates,
            key=lambda row: (
                abs(float(row.get("time_seconds") or 0) - target_time),
                -float(row.get("start_measure_confidence") or 0),
            ),
        )
        with db.connect() as connection:
            rows = connection.execute(
                """
                SELECT stored_path, metadata, sort_order FROM assets
                WHERE project_id = ? AND kind IN ('video_frame', 'score_system')
                ORDER BY sort_order, created_at
                """,
                (project_id,),
            ).fetchall()
        available = []
        for row in rows:
            path = Path(row["stored_path"])
            if not path.is_file():
                continue
            metadata = json_object(row["metadata"]) or {}
            available.append((row, path, metadata))
        if not available:
            raise HTTPException(status_code=404, detail="原始谱面切片已经不存在，请重新上传或切片")
        matching = [item for item in available if item[1].name == frame_info.get("name")]
        if matching:
            frame_row, frame_path, metadata = matching[0]
        else:
            frame_row, frame_path, metadata = min(
                available,
                key=lambda item: abs(float(item[2].get("time_seconds") or 0) - float(frame_info.get("time_seconds") or 0)),
            )
        frame = TabFrameInput(
            path=frame_path,
            time_seconds=float(metadata.get("time_seconds") or frame_info.get("time_seconds") or 0),
            source_frame=int(metadata.get("source_frame") or frame_row["sort_order"]),
        )
        try:
            with tempfile.TemporaryDirectory(prefix="measure-retry-", dir=score_path.parent) as work_dir:
                return recognize_tab_measure(
                    frame,
                    Path(work_dir),
                    frame_start_measure=int(frame_info["start_measure"]),
                    measure_number=measure_number,
                    tesseract_path=capabilities["tesseract_path"],
                    recognition_mode="ai" if summary.get("engine") == "tab_cv_ai" else "ocr",
                    vision_model_url=os.getenv("NOCTURNE_VISION_MODEL"),
                )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"第 {measure_number} 小节重新识别失败：{str(exc)[:240]}") from exc

    @api.patch("/api/projects/{project_id}/recognition/measures/{measure_number}")
    def edit_recognized_measure(
        project_id: str,
        measure_number: int,
        body: TabMeasureEdit,
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        score_path, diagnostics_path, _ = tab_diagnostics_for_project(project)
        try:
            diagnostics = update_recognized_measure(
                score_path,
                diagnostics_path,
                title=project["title"],
                measure_number=measure_number,
                events=[event.model_dump() for event in body.events],
                time_signature=(body.time_signature.numerator, body.time_signature.denominator)
                if body.time_signature else None,
                human_verified=body.human_verified,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        summary = diagnostics.get("summary") or {}
        now = utc_text()
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET recognition_summary = ?, status_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(summary, ensure_ascii=False),
                    f"第 {measure_number} 小节已人工校正并重新生成乐谱",
                    now,
                    project_id,
                ),
            )
        return diagnostics

    @api.post("/api/projects/{project_id}/recognition/measures", status_code=201)
    def append_recognition_measure(
        project_id: str,
        after_measure: int | None = None,
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        score_path, diagnostics_path, _ = tab_diagnostics_for_project(project)
        try:
            diagnostics = append_blank_tab_measure(
                score_path,
                diagnostics_path,
                title=project["title"],
                after_measure=after_measure,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        summary = diagnostics.get("summary") or {}
        inserted_measure = int(after_measure) + 1 if after_measure is not None else int(summary.get("end_measure") or 1)
        now = utc_text()
        with db.connect() as connection:
            if after_measure is not None:
                connection.execute(
                    """
                    UPDATE sync_points SET measure_number = measure_number + 1
                    WHERE project_id = ? AND measure_number > ?
                    """,
                    (project_id, after_measure),
                )
            connection.execute(
                """
                UPDATE projects SET recognition_summary = ?, status_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(summary, ensure_ascii=False),
                    f"已插入第 {inserted_measure} 小节，乐谱已重新生成",
                    now,
                    project_id,
                ),
            )
        return diagnostics

    @api.post("/api/projects/{project_id}/sync-points", status_code=201)
    def create_sync_point(
        project_id: str,
        body: SyncPointRequest,
        user: dict = Depends(current_user),
    ) -> dict:
        owned_project(project_id, user["id"])
        now = utc_text()
        with db.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sync_points(project_id, measure_number, time_seconds, score_position, label, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    body.measure_number,
                    round(body.time_seconds, 3),
                    round(body.score_position, 5),
                    body.label.strip(),
                    now,
                ),
            )
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
            point = connection.execute("SELECT * FROM sync_points WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(point)

    @api.delete("/api/projects/{project_id}/sync-points/{point_id}", status_code=204)
    def delete_sync_point(
        project_id: str,
        point_id: int,
        user: dict = Depends(current_user),
    ) -> Response:
        owned_project(project_id, user["id"])
        with db.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM sync_points WHERE id = ? AND project_id = ?", (point_id, project_id)
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="同步点不存在")
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (utc_text(), project_id))
        return Response(status_code=204)

    @api.get("/api/projects/{project_id}/assets/{asset_id}")
    def get_asset(project_id: str, asset_id: str, user: dict = Depends(current_user)) -> FileResponse:
        owned_project(project_id, user["id"])
        with db.connect() as connection:
            asset = connection.execute(
                "SELECT * FROM assets WHERE id = ? AND project_id = ?", (asset_id, project_id)
            ).fetchone()
        if not asset or not Path(asset["stored_path"]).is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(
            asset["stored_path"],
            media_type=asset["media_type"],
            filename=asset["original_name"],
            content_disposition_type="inline",
        )

    @api.get("/api/projects/{project_id}/files/{kind}")
    def get_project_file(
        project_id: str,
        kind: str,
        download: bool = False,
        user: dict = Depends(current_user),
    ) -> FileResponse:
        project = owned_project(project_id, user["id"])
        if kind == "cover":
            metadata = json_object(project["source_metadata"]) or {}
            path = resolve_project_file(project_id, metadata.get("_cover_path"))
            if not path:
                raise HTTPException(status_code=404, detail="封面不存在")
            return FileResponse(
                path,
                media_type="image/jpeg",
                filename=f"{safe_download_stem(project['title'], project['source_id'])}-cover.jpg",
                content_disposition_type="attachment" if download else "inline",
            )
        mapping = {
            "pdf": (
                "score_pdf_path",
                f"{safe_download_stem(project['title'], project['source_id'])}.pdf",
                "application/pdf",
            ),
            "audio": ("audio_path", project["audio_name"] or "practice-audio", None),
            "score": ("score_file_path", project["score_file_name"] or "score", "application/octet-stream"),
            "video": ("video_path", "source-video", None),
        }
        if kind not in mapping:
            raise HTTPException(status_code=404, detail="文件类型不存在")
        field, filename, media_type = mapping[kind]
        path = resolve_project_file(project_id, project[field])
        if not path:
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(
            path,
            media_type=media_type,
            filename=filename,
            content_disposition_type="attachment" if download else "inline",
        )

    dist_dir = ROOT_DIR / "dist"
    if dist_dir.is_dir():
        api.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")

    return api


app = create_app()
