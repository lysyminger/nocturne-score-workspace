from __future__ import annotations

import argparse
import functools
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddle import nn
from paddle.io import DataLoader, Dataset


CLASS_NAMES = [f"fret_{value}" for value in range(23)] + ["dead_note"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full-token fret classifier on synthetic and vector PDF crops")
    parser.add_argument("--synthetic", type=Path, default=Path("training/data/fret-ocr"))
    parser.add_argument("--pdf", type=Path, default=Path("training/data/pdf-frets"))
    parser.add_argument("--extra-pdf", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("training/runs/fret-classifier"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--synthetic-per-class", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


@functools.lru_cache(maxsize=512)
def read_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


class FretDataset(Dataset):
    def __init__(self, samples: list[dict], train: bool) -> None:
        self.samples = samples
        self.train = train
        self.images = np.empty((len(samples), 64, 96), dtype=np.uint8)
        for index, row in enumerate(samples):
            self.images[index] = self._load_image(row)
            if (index + 1) % 10000 == 0:
                print(f"cached {index + 1}/{len(samples)} token images", flush=True)

    @staticmethod
    def _load_image(row: dict) -> np.ndarray:
        source = read_gray(row["image"])
        if "bbox" not in row:
            image = source
        else:
            x, y, width, height = row["bbox"]
            pad_x = max(2, int(round(width * 0.45)))
            pad_y = max(2, int(round(height * 0.32)))
            left = max(0, int(np.floor(x)) - pad_x)
            top = max(0, int(np.floor(y)) - pad_y)
            right = min(source.shape[1], int(np.ceil(x + width)) + pad_x)
            bottom = min(source.shape[0], int(np.ceil(y + height)) + pad_y)
            image = source[top:bottom, left:right]
        return cv2.resize(image, (96, 64), interpolation=cv2.INTER_AREA)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        image = self.images[index].copy()
        if self.train:
            if random.random() < 0.32:
                image = 255 - image
            image = cv2.convertScaleAbs(image, alpha=random.uniform(0.68, 1.32), beta=random.uniform(-28, 28))
            if random.random() < 0.25:
                image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.1, 0.8))
        tensor = (image.astype(np.float32) / 255.0)[None, :, :]
        return tensor, np.int64(row["class_id"])


class FretNet(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2D(1, 32, 3, padding=1), nn.BatchNorm2D(32), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(32, 64, 3, padding=1), nn.BatchNorm2D(64), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(64, 128, 3, padding=1), nn.BatchNorm2D(128), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(128, 192, 3, padding=1), nn.BatchNorm2D(192), nn.ReLU(),
            nn.AdaptiveAvgPool2D(1), nn.Flatten(), nn.Dropout(0.18), nn.Linear(192, len(CLASS_NAMES)),
        )

    def forward(self, image):
        return self.network(image)


def synthetic_samples(root: Path, per_class: int, seed: int) -> list[dict]:
    root = root.absolute()
    grouped: dict[int, list[dict]] = {class_id: [] for class_id in range(23)}
    for line in (root / "train.txt").read_text(encoding="utf-8").splitlines():
        relative, label = line.split("\t", 1)
        if label.isdigit() and 0 <= int(label) <= 22:
            grouped[int(label)].append(
                {"image": str(root / relative), "class_id": int(label), "domain": "synthetic"}
            )
    generator = random.Random(seed)
    rows = []
    for samples in grouped.values():
        generator.shuffle(samples)
        rows.extend(samples[:per_class])
    generator.shuffle(rows)
    return rows


def pdf_samples(root: Path, split: str) -> list[dict]:
    root = root.absolute()
    payload = json.loads((root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    categories = {int(row["id"]): CLASS_NAMES.index(row["name"]) for row in payload["categories"]}
    rows = []
    for annotation in payload["annotations"]:
        info = images[int(annotation["image_id"])]
        rows.append(
            {
                "image": str(root / "images" / split / info["file_name"]),
                "bbox": annotation["bbox"],
                "class_id": categories[int(annotation["category_id"])],
                "domain": "pdf",
            }
        )
    return rows


def macro_f1(truth: list[int], predicted: list[int]) -> float:
    scores = []
    for class_id in sorted(set(truth)):
        tp = sum(t == class_id and p == class_id for t, p in zip(truth, predicted))
        fp = sum(t != class_id and p == class_id for t, p in zip(truth, predicted))
        fn = sum(t == class_id and p != class_id for t, p in zip(truth, predicted))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(np.mean(scores))


@paddle.no_grad()
def evaluate(model: FretNet, loader: DataLoader) -> dict[str, float]:
    model.eval()
    truth, predicted = [], []
    for images, labels in loader:
        predictions = paddle.argmax(model(images), axis=1)
        truth.extend(labels.numpy().tolist())
        predicted.extend(predictions.numpy().tolist())
    model.train()
    return {
        "accuracy": float(np.mean(np.array(truth) == np.array(predicted))),
        "macro_f1": macro_f1(truth, predicted),
        "samples": len(truth),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device("gpu:0")
    args.output.mkdir(parents=True, exist_ok=True)
    training_samples = synthetic_samples(args.synthetic, args.synthetic_per_class, args.seed) + pdf_samples(args.pdf, "train")
    for extra_pdf in args.extra_pdf:
        training_samples.extend(pdf_samples(extra_pdf, "train"))
    validation_samples = pdf_samples(args.pdf, "val")
    test_samples = pdf_samples(args.pdf, "test")
    train_data = FretDataset(training_samples, train=True)
    val_data = FretDataset(validation_samples, train=False)
    test_data = FretDataset(test_samples, train=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        use_shared_memory=False,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = FretNet()
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(4e-4, args.epochs)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4)
    counts = Counter(row["class_id"] for row in training_samples)
    values = np.array([counts.get(index, 1) for index in range(len(CLASS_NAMES))], dtype=np.float32)
    weights = np.clip(np.sqrt(values.max() / values), 1.0, 10.0)
    criterion = nn.CrossEntropyLoss(weight=paddle.to_tensor(weights / weights.mean(), dtype="float32"))
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, labels in train_loader:
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            total_loss += float(loss.item())
            batches += 1
        scheduler.step()
        metrics = evaluate(model, val_loader)
        row = {"epoch": epoch, "loss": total_loss / max(1, batches), **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = (metrics["accuracy"] + metrics["macro_f1"]) / 2
        if score > best_score:
            best_score = score
            paddle.save(model.state_dict(), str(args.output / "best.pdparams"))
    model.set_state_dict(paddle.load(str(args.output / "best.pdparams")))
    test_metrics = evaluate(model, test_loader)
    report = {
        "schema_version": 1,
        "classes": CLASS_NAMES,
        "source_isolated_pdf_test": True,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "history": history,
        "test": test_metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
