from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddle import nn
from paddle.io import DataLoader, Dataset


DURATIONS = [1, 2, 4, 8, 16, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-head TAB rhythm classifier")
    parser.add_argument("--dataset", type=Path, default=Path("training/data/tab-events"))
    parser.add_argument("--output", type=Path, default=Path("training/runs/rhythm-model"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


class RhythmDataset(Dataset):
    def __init__(self, root: Path, split: str, train: bool) -> None:
        self.root = root
        self.train = train
        rows = [
            json.loads(line)
            for line in (root / f"beats-{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rows = [row for row in rows if int(row["duration"]) in DURATIONS and 0 <= int(row["dots"]) <= 2]
        self.images = np.empty((len(self.rows), 128, 128), dtype=np.uint8)
        for index, row in enumerate(self.rows):
            image = cv2.imread(str(self.root / row["image"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(row["image"])
            self.images[index] = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
            if (index + 1) % 10000 == 0:
                print(f"cached {split} rhythm images {index + 1}/{len(self.rows)}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = self.images[index].copy()
        if self.train:
            if random.random() < 0.30:
                image = 255 - image
            alpha = random.uniform(0.72, 1.28)
            beta = random.uniform(-24, 24)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            if random.random() < 0.25:
                image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.1, 0.8))
        tensor = (image.astype(np.float32) / 255.0)[None, :, :]
        return (
            tensor,
            np.int64(DURATIONS.index(int(row["duration"]))),
            np.int64(int(row["dots"])),
            np.int64(bool(row["rest"])),
        )


class RhythmNet(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2D(1, 32, 3, padding=1), nn.BatchNorm2D(32), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(32, 64, 3, padding=1), nn.BatchNorm2D(64), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(64, 128, 3, padding=1), nn.BatchNorm2D(128), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(128, 192, 3, padding=1), nn.BatchNorm2D(192), nn.ReLU(),
            nn.AdaptiveAvgPool2D(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(192, 192), nn.ReLU(),
        )
        self.duration_head = nn.Linear(192, len(DURATIONS))
        self.dots_head = nn.Linear(192, 3)
        self.rest_head = nn.Linear(192, 2)

    def forward(self, image):
        shared = self.features(image)
        return self.duration_head(shared), self.dots_head(shared), self.rest_head(shared)


def class_weights(dataset: RhythmDataset, field: str, class_count: int) -> paddle.Tensor:
    if field == "duration":
        values_for_rows = [DURATIONS.index(int(row["duration"])) for row in dataset.rows]
    elif field == "dots":
        values_for_rows = [int(row["dots"]) for row in dataset.rows]
    else:
        values_for_rows = [int(bool(row["rest"])) for row in dataset.rows]
    counts = Counter(values_for_rows)
    values = np.array([counts.get(index, 1) for index in range(class_count)], dtype=np.float32)
    weights = np.sqrt(values.max() / values)
    weights = np.clip(weights, 1.0, 12.0)
    return paddle.to_tensor(weights / weights.mean(), dtype="float32")


def macro_f1(truth: list[int], predicted: list[int], class_count: int) -> float:
    scores = []
    for class_id in range(class_count):
        tp = sum(t == class_id and p == class_id for t, p in zip(truth, predicted))
        fp = sum(t != class_id and p == class_id for t, p in zip(truth, predicted))
        fn = sum(t == class_id and p != class_id for t, p in zip(truth, predicted))
        if tp + fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(sum(scores) / max(1, len(scores)))


@paddle.no_grad()
def evaluate(model: RhythmNet, loader: DataLoader) -> dict[str, float]:
    model.eval()
    truths = [[], [], []]
    predictions = [[], [], []]
    exact = total = 0
    for image, duration, dots, rest in loader:
        outputs = model(image)
        targets = (duration, dots, rest)
        batch_predictions = [paddle.argmax(output, axis=1) for output in outputs]
        batch_exact = paddle.ones_like(duration, dtype="bool")
        for head, (target, prediction) in enumerate(zip(targets, batch_predictions)):
            truths[head].extend(target.numpy().tolist())
            predictions[head].extend(prediction.numpy().tolist())
            batch_exact = paddle.logical_and(batch_exact, target == prediction)
        exact += int(batch_exact.astype("int64").sum().item())
        total += int(duration.shape[0])
    metrics = {
        "duration_accuracy": float(np.mean(np.array(truths[0]) == np.array(predictions[0]))),
        "dots_accuracy": float(np.mean(np.array(truths[1]) == np.array(predictions[1]))),
        "rest_accuracy": float(np.mean(np.array(truths[2]) == np.array(predictions[2]))),
        "joint_accuracy": exact / max(1, total),
        "duration_macro_f1": macro_f1(truths[0], predictions[0], len(DURATIONS)),
        "dots_macro_f1": macro_f1(truths[1], predictions[1], 3),
        "rest_macro_f1": macro_f1(truths[2], predictions[2], 2),
    }
    model.train()
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device("gpu:0")
    args.output.mkdir(parents=True, exist_ok=True)

    train_data = RhythmDataset(args.dataset, "train", train=True)
    val_data = RhythmDataset(args.dataset, "val", train=False)
    test_data = RhythmDataset(args.dataset, "test", train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RhythmNet()
    optimizer = paddle.optimizer.AdamW(
        learning_rate=paddle.optimizer.lr.CosineAnnealingDecay(3e-4, args.epochs),
        parameters=model.parameters(),
        weight_decay=1e-4,
    )
    losses = [
        nn.CrossEntropyLoss(weight=class_weights(train_data, "duration", len(DURATIONS))),
        nn.CrossEntropyLoss(weight=class_weights(train_data, "dots", 3)),
        nn.CrossEntropyLoss(weight=class_weights(train_data, "rest", 2)),
    ]
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for image, duration, dots, rest in train_loader:
            outputs = model(image)
            loss = losses[0](outputs[0], duration) + losses[1](outputs[1], dots) + losses[2](outputs[2], rest)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            running_loss += float(loss.item())
            batches += 1
        metrics = evaluate(model, val_loader)
        score = (metrics["joint_accuracy"] + metrics["duration_macro_f1"] + metrics["rest_macro_f1"]) / 3
        row = {"epoch": epoch, "loss": running_loss / max(1, batches), **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            paddle.save(model.state_dict(), str(args.output / "best.pdparams"))

    model.set_state_dict(paddle.load(str(args.output / "best.pdparams")))
    test_metrics = evaluate(model, test_loader)
    report = {
        "schema_version": 1,
        "source_isolated": True,
        "durations": DURATIONS,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "history": history,
        "test": test_metrics,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
