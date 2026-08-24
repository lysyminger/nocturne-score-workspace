from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MAX_REQUEST_BYTES = 24 * 1024 * 1024
MAX_IMAGES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the full TAB fret-position detector")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--paddledetection", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--threshold", type=float, default=0.35)
    return parser.parse_args()


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def main() -> None:
    args = parse_args()
    deploy_python = args.paddledetection.resolve() / "deploy" / "python"
    sys.path.insert(0, str(deploy_python))
    from infer import Detector  # noqa: PLC0415

    detector = Detector(
        str(args.model.resolve()),
        device=args.device,
        batch_size=1,
        threshold=args.threshold,
    )
    labels = list(detector.pred_config.labels)
    predictor_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "NocturneTabEventDetector/1.0"

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
            self.send_json(200, {"status": "ok", "engine": "tab-event-detector", "device": args.device.lower()})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/tab-token-detect":
                self.send_json(404, {"detail": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("request body is empty or too large")
                payload = json.loads(self.rfile.read(length))
                images = payload.get("images")
                if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
                    raise ValueError(f"images must contain 1-{MAX_IMAGES} items")
                with tempfile.TemporaryDirectory(prefix="nocturne-tab-detector-") as work_dir:
                    paths = []
                    for index, encoded in enumerate(images):
                        if not isinstance(encoded, str):
                            raise ValueError("each image must be base64 text")
                        raw = base64.b64decode(encoded, validate=True)
                        if not raw:
                            raise ValueError("decoded image is empty")
                        image_path = Path(work_dir) / f"{index:03d}.png"
                        image_path.write_bytes(raw)
                        paths.append(str(image_path))
                    with predictor_lock:
                        result = detector.predict_image(paths, visual=False)
                boxes = result["boxes"]
                counts = result["boxes_num"]
                detections = []
                offset = 0
                for count in counts:
                    image_detections = []
                    for class_id, score, x1, y1, x2, y2 in boxes[offset:offset + int(count)]:
                        if float(score) < args.threshold:
                            continue
                        class_index = int(class_id)
                        image_detections.append(
                            {
                                "class": labels[class_index],
                                "confidence": round(float(score), 6),
                                "box": [round(float(x1), 3), round(float(y1), 3), round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                            }
                        )
                    detections.append(image_detections)
                    offset += int(count)
                self.send_json(200, {"detections": detections})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(422, {"detail": str(exc)[:240]})
            except Exception as exc:
                self.send_json(500, {"detail": str(exc)[:240]})

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TAB event detector listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
