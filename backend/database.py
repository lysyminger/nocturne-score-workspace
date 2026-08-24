from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    source_input TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    rights_confirmed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'draft',
    status_message TEXT NOT NULL DEFAULT '',
    source_metadata TEXT,
    video_path TEXT,
    video_analysis TEXT,
    recognition_summary TEXT,
    audio_analysis TEXT,
    tempo_bpm REAL,
    tempo_source TEXT,
    tempo_locked INTEGER NOT NULL DEFAULT 0,
    score_pdf_path TEXT,
    score_file_path TEXT,
    score_file_name TEXT,
    audio_path TEXT,
    audio_name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_user_updated
ON projects(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    metadata TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assets_project_kind
ON assets(project_id, kind, sort_order);

CREATE TABLE IF NOT EXISTS sync_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    measure_number INTEGER NOT NULL,
    time_seconds REAL NOT NULL,
    score_position REAL NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sync_points_project_time
ON sync_points(project_id, time_seconds);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            project_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
            }
            if "video_analysis" not in project_columns:
                connection.execute("ALTER TABLE projects ADD COLUMN video_analysis TEXT")
            if "recognition_summary" not in project_columns:
                connection.execute("ALTER TABLE projects ADD COLUMN recognition_summary TEXT")
            if "audio_analysis" not in project_columns:
                connection.execute("ALTER TABLE projects ADD COLUMN audio_analysis TEXT")
            if "tempo_bpm" not in project_columns:
                connection.execute("ALTER TABLE projects ADD COLUMN tempo_bpm REAL")
            if "tempo_source" not in project_columns:
                connection.execute("ALTER TABLE projects ADD COLUMN tempo_source TEXT")
            if "tempo_locked" not in project_columns:
                connection.execute("ALTER TABLE projects ADD COLUMN tempo_locked INTEGER NOT NULL DEFAULT 0")

            asset_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(assets)").fetchall()
            }
            if "metadata" not in asset_columns:
                connection.execute("ALTER TABLE assets ADD COLUMN metadata TEXT")
