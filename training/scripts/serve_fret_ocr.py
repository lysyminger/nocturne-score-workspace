from __future__ import annotations

import argparse
import base64
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from prepare_real_video_corpus import FretRecognizer


MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_IMAGES = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在本机 GPU 上提供只读品位 OCR 服务。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def build_handler(recognizer: FretRecognizer, batch_size: int) -> type[BaseHTTPRequestHandler]:
    predictor_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "NocturneLocalOCR/1.0"

        def send_json(self, status: int, payload: object) -> None:
            body = json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_json(404, {"detail": "not found"})
                return
            self.send_json(200, {"status": "ok", "engine": "paddle-fret-ocr", "device": "local"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/fret-ocr":
                self.send_json(404, {"detail": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self.send_json(413, {"detail": "request body is empty or too large"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                images = payload.get("images")
                if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
                    raise ValueError(f"images must contain 1-{MAX_IMAGES} items")
                with tempfile.TemporaryDirectory(prefix="nocturne-fret-ocr-") as work_dir:
                    paths: list[Path] = []
                    for index, encoded in enumerate(images):
                        if not isinstance(encoded, str):
                            raise ValueError("each image must be base64 text")
                        raw = base64.b64decode(encoded, validate=True)
                        if not raw:
                            raise ValueError("decoded image is empty")
                        path = Path(work_dir) / f"{index:04d}.png"
                        path.write_bytes(raw)
                        paths.append(path)
                    with predictor_lock:
                        predictions = recognizer.predict(paths, batch_size)
                self.send_json(
                    200,
                    {
                        "predictions": [
                            {"text": text, "confidence": round(confidence, 6)}
                            for text, confidence in predictions
                        ]
                    },
                )
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(422, {"detail": str(exc)[:240]})
            except Exception as exc:  # Keep the LAN service alive after one bad batch.
                self.send_json(500, {"detail": str(exc)[:240]})

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    return Handler


def main() -> None:
    args = parse_args()
    recognizer = FretRecognizer(args.model.resolve(), args.device)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(recognizer, args.batch_size),
    )
    print(f"local fret OCR listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
