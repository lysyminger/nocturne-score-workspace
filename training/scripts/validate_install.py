from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy
import paddle
import paddleocr


def main() -> None:
    training_root = Path(__file__).resolve().parents[1]
    detection_root = training_root / "vendor" / "PaddleDetection"
    sys.path.insert(0, str(detection_root))
    import ppdet  # noqa: F401

    if not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise SystemExit("Paddle CUDA GPU is unavailable")
    print(f"PaddlePaddle {paddle.__version__} / PaddleOCR {paddleocr.__version__}")
    print(f"OpenCV {cv2.__version__} / NumPy {numpy.__version__}")
    print(f"PaddleDetection 2.9 source: {detection_root}")
    print(f"Device: {paddle.device.get_device()}")


if __name__ == "__main__":
    main()
