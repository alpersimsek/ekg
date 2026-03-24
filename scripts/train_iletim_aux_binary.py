from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_uc_sinif_baseline import BaselineECGNet, set_seed


CLASS_NAMES = ["diger", "iletim"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="finalist_v2 icin iletim yardimci binary model egitimi.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--augment", choices=["none", "light"], default="light")
    parser.add_argument("--sampler", choices=["none", "weighted"], default="weighted")
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="balanced")
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["target_class"] not in {"normal", "ritim", "iletim"}:
                continue
            if row["split"] == "egitim":
                train_rows.append(row)
            elif row["split"] == "dogrulama":
                val_rows.append(row)
            elif row["split"] == "test":
                test_rows.append(row)
    return train_rows, val_rows, test_rows


class BinaryECGDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], augment: str = "none") -> None:
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        signal = loadmat(row["mat_yolu"])["val"].astype(np.float32)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        signal = (signal - mean) / std
        if self.augment == "light":
            scale = np.random.uniform(0.95, 1.05, size=(signal.shape[0], 1)).astype(np.float32)
            signal = signal * scale
            shift = np.random.randint(-20, 21)
            if shift != 0:
                signal = np.roll(signal, shift=shift, axis=1)
            signal = signal + np.random.normal(0.0, 0.01, size=signal.shape).astype(np.float32)
        label = 1 if row["target_class"] == "iletim" else 0
        return torch.from_numpy(signal), label


def build_train_sampler(rows: list[dict[str, str]]) -> WeightedRandomSampler:
    counts = Counter(1 if row["target_class"] == "iletim" else 0 for row in rows)
    sample_weights = [1.0 / counts[1 if row["target_class"] == "iletim" else 0] for row in rows]
    return WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), len(rows), replacement=True)


def compute_class_weights(rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = Counter(1 if row["target_class"] == "iletim" else 0 for row in rows)
    total = len(rows)
    weights = [total / (2 * counts[idx]) for idx in range(2)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_backbone_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    matched = {}
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            matched[key] = value
    target_state.update(matched)
    model.load_state_dict(target_state)
    return len(matched)


@dataclass
class BinaryEvalResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> BinaryEvalResult:
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(signals)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_items += labels.size(0)
            for truth, pred in zip(labels.cpu().numpy(), preds.cpu().numpy(), strict=False):
                confusion[int(truth), int(pred)] += 1

    tp = float(confusion[1, 1])
    fp = float(confusion[0, 1])
    fn = float(confusion[1, 0])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    return BinaryEvalResult(
        loss=total_loss / max(total_items, 1),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=confusion.tolist(),
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)
    return total_loss / max(total_items, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_rows, val_rows, test_rows = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sampler = build_train_sampler(train_rows) if args.sampler == "weighted" else None
    train_loader = DataLoader(
        BinaryECGDataset(train_rows, augment=args.augment),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        BinaryECGDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        BinaryECGDataset(test_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = BaselineECGNet(num_classes=2, dropout=args.dropout).to(device)
    matched_weights = load_backbone_weights(model, args.base_checkpoint, device)

    class_weights = compute_class_weights(train_rows, device) if args.class_weight_mode == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_checkpoint = args.output_dir / "best_model.pt"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            "val_f1": val_result.f1,
            "val_precision": val_result.precision,
            "val_recall": val_result.recall,
            "val_accuracy": val_result.accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))
        if scheduler is not None:
            scheduler.step()
        if val_result.f1 > best_val_f1:
            best_val_f1 = val_result.f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "class_names": CLASS_NAMES,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val = evaluate(model, val_loader, criterion, device)
    final_test = evaluate(model, test_loader, criterion, device)
    report = {
        "manifest": str(args.manifest),
        "base_checkpoint": str(args.base_checkpoint),
        "matched_weights": matched_weights,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "augment": args.augment,
        "sampler": args.sampler,
        "class_weight_mode": args.class_weight_mode,
        "best_epoch": best_epoch,
        "history": history,
        "validation": {
            "loss": final_val.loss,
            "accuracy": final_val.accuracy,
            "precision": final_val.precision,
            "recall": final_val.recall,
            "f1": final_val.f1,
            "confusion_matrix": final_val.confusion_matrix,
        },
        "test": {
            "loss": final_test.loss,
            "accuracy": final_test.accuracy,
            "precision": final_test.precision,
            "recall": final_test.recall,
            "f1": final_test.f1,
            "confusion_matrix": final_test.confusion_matrix,
        },
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
