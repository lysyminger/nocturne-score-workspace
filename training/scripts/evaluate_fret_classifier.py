from __future__ import annotations

import argparse
import json
from pathlib import Path

import paddle
from paddle.io import DataLoader

from train_fret_classifier import FretDataset, FretNet, evaluate, pdf_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the full-token fret classifier")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--weights", type=Path, default=Path("training/runs/fret-classifier/best.pdparams"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paddle.set_device("gpu:0")
    dataset = FretDataset(pdf_samples(args.pdf, args.split), train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = FretNet()
    model.set_state_dict(paddle.load(str(args.weights)))
    result = evaluate(model, loader)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
