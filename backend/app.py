from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from backend.audio_analysis import analyze_audio_file
from backend.bilibili import inspect_bilibili, parse_bilibili_source
from backend.database import Database
from backend.score_pdf import build_slice_preview_pdf
from backend.security import hash_password, hash_session_token, new_session_token, verify_password
from backend.tab_recognition import FrameInput as TabFrameInput
from backend.tab_recognition import recognize_tab_frames, update_recognized_measure
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


class ProjectUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ProjectRightsRequest(BaseModel):
    rights_confirmed: bool


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
    onset_eighths: int = Field(ge=0, le=7)
    duration_eighths: int = Field(ge=1, le=8)
    notes: list[TabNoteEdit] = Field(min_length=1, max_length=6)


class TabMeasureEdit(BaseModel):
    events: list[TabEventEdit] = Field(max_length=32)


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
    return {
        "ffmpeg": ffmpeg_available,
        "yt_dlp": yt_dlp_available,
        "audiveris": bool(audiveris_path),
        "audiveris_path": audiveris_path or None,
        "tab_ocr": bool(tesseract_path and tab_cv_available),
        "tesseract_path": tesseract_path or None,
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

    def project_payload(row) -> dict:
        project = dict(row)
        project["rights_confirmed"] = bool(project["rights_confirmed"])
        project["source_metadata"] = json_object(project["source_metadata"])
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

    @api.post("/api/projects/{project_id}/inspect")
    async def inspect_source(project_id: str, user: dict = Depends(current_user)) -> dict:
        project = owned_project(project_id, user["id"])
        if not capability_status()["yt_dlp"]:
            raise HTTPException(status_code=503, detail="yt-dlp 未安装，暂时不能解析视频信息")
        try:
            metadata = await run_in_threadpool(inspect_bilibili, project["source_url"])
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
                    UPDATE projects SET video_analysis = ?, score_pdf_path = ?, status = ?, status_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(completed, ensure_ascii=False),
                        pdf_path,
                        next_status,
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
                    UPDATE projects SET audio_analysis = ?, status_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (json.dumps(analysis, ensure_ascii=False), message, utc_text(), project_id),
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
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE projects SET score_file_path = ?, score_file_name = ?, status = 'score_ready',
                recognition_summary = NULL, status_message = '结构化乐谱已准备，可在线试听',
                updated_at = ? WHERE id = ?
                """,
                (str(target), file.filename or target.name, utc_text(), project_id),
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
            with db.connect() as connection:
                connection.execute(
                    """
                    UPDATE projects SET score_file_path = ?, score_file_name = ?, status = 'score_ready',
                    recognition_summary = ?, status_message = '五线谱识别完成，请逐小节校对',
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
    ) -> None:
        try:
            result = recognize_tab_frames(
                frames,
                output_dir,
                title=title,
                tesseract_path=tesseract_path,
            )
            summary = result.summary
            message = (
                f"已识别 {summary['measure_count']} 小节六线 TAB；"
                "已按小节号去重并合成完整 PDF，请在保存前逐小节试听校对"
            )
            with db.connect() as connection:
                connection.execute(
                    """
                    UPDATE projects SET score_file_path = ?, score_file_name = ?, status = 'score_ready',
                    score_pdf_path = COALESCE(?, score_pdf_path), recognition_summary = ?,
                    status_message = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        str(result.score_path),
                        result.score_path.name,
                        str(result.pdf_path) if result.pdf_path else None,
                        json.dumps(summary, ensure_ascii=False),
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
        user: dict = Depends(current_user),
    ) -> dict:
        project = owned_project(project_id, user["id"])
        if project["status"] == "recognizing":
            raise HTTPException(status_code=409, detail="当前项目正在识别，请等待完成")
        capabilities = capability_status()
        with db.connect() as connection:
            frame_rows = connection.execute(
                """
                SELECT stored_path, metadata, sort_order FROM assets
                WHERE project_id = ? AND kind = 'video_frame'
                ORDER BY sort_order, created_at
                """,
                (project_id,),
            ).fetchall()
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
            status_message = "正在识别六线 TAB 的品位、弦号和节奏网格…"
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
            )
        else:
            background_tasks.add_task(
                recognize_staff_job,
                project_id,
                project["score_pdf_path"],
                capabilities["audiveris_path"],
                output_dir,
            )
        return {"status": "recognizing", "message": "识别任务已开始", "engine": engine}

    def tab_diagnostics_for_project(project) -> tuple[Path, Path, dict]:
        summary = json_object(project["recognition_summary"])
        if not summary or summary.get("engine") != "tab_cv_tesseract":
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
    def get_project_file(project_id: str, kind: str, user: dict = Depends(current_user)) -> FileResponse:
        project = owned_project(project_id, user["id"])
        mapping = {
            "pdf": ("score_pdf_path", "score.pdf", "application/pdf"),
            "audio": ("audio_path", project["audio_name"] or "practice-audio", None),
            "score": ("score_file_path", project["score_file_name"] or "score", "application/octet-stream"),
            "video": ("video_path", "source-video", None),
        }
        if kind not in mapping:
            raise HTTPException(status_code=404, detail="文件类型不存在")
        field, filename, media_type = mapping[kind]
        path_value = project[field]
        if not path_value or not Path(path_value).is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(path_value, media_type=media_type, filename=filename, content_disposition_type="inline")

    dist_dir = ROOT_DIR / "dist"
    if dist_dir.is_dir():
        api.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")

    return api


app = create_app()
