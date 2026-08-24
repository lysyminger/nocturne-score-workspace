from __future__ import annotations

import paddle


def main() -> None:
    paddle.utils.run_check()
    if not paddle.device.is_compiled_with_cuda():
        raise SystemExit("PaddlePaddle 当前不是 CUDA 构建")
    paddle.set_device("gpu:0")
    left = paddle.randn((2048, 2048), dtype="float16")
    right = paddle.randn((2048, 2048), dtype="float16")
    result = paddle.matmul(left, right)
    print(f"PaddlePaddle {paddle.__version__}")
    print(f"Device: {paddle.device.get_device()} / FP16 check: {float(result[0, 0]):.4f}")


if __name__ == "__main__":
    main()
