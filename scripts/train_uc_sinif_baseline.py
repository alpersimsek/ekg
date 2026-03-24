from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CLASS_NAMES = ["normal", "ritim", "iletim"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3 sinifli 12 kanal EKG baz model egitimi.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif_baseline"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--model",
        choices=["baseline", "resnet_se"],
        default="baseline",
        help="Kullanilacak model mimarisi.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine"],
        default="none",
        help="Ogrenme orani planlayicisi.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Cosine scheduler icin alt ogrenme orani.",
    )
    parser.add_argument(
        "--augment",
        choices=["none", "light"],
        default="none",
        help="Egitim icin hafif sinyal augmentasyonu uygula.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Var olan checkpoint'ten egitime devam et.",
    )
    parser.add_argument(
        "--sampler",
        choices=["none", "weighted"],
        default="none",
        help="Egitim veri yukleyicisi icin ornekleyici tipi.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["balanced", "none"],
        default="balanced",
        help="Cross-entropy icin sinif agirligi modu.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["target_class"] not in CLASS_TO_INDEX:
                continue
            if row["split"] == "egitim":
                train_rows.append(row)
            elif row["split"] == "dogrulama":
                val_rows.append(row)
            elif row["split"] == "test":
                test_rows.append(row)
    return train_rows, val_rows, test_rows


class ECGDataset(Dataset):
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
            # Morfolojiyi bozmadan hafif genlik ve zaman oynatması uygula.
            scale = np.random.uniform(0.95, 1.05, size=(signal.shape[0], 1)).astype(np.float32)
            signal = signal * scale
            shift = np.random.randint(-20, 21)
            if shift != 0:
                signal = np.roll(signal, shift=shift, axis=1)
            signal = signal + np.random.normal(0.0, 0.01, size=signal.shape).astype(np.float32)
        label = CLASS_TO_INDEX[row["target_class"]]
        return torch.from_numpy(signal), label


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act(x + residual)


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x).squeeze(-1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale


class ResNetSEBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        return self.act(x + residual)


class BaselineECGNet(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.layer1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock1D(64),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            ResidualBlock1D(128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)


class ResNetSEECGNet(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(
            ResNetSEBlock1D(32, 64, stride=2, dropout=dropout * 0.25),
            ResNetSEBlock1D(64, 64, stride=1, dropout=dropout * 0.25),
        )
        self.layer2 = nn.Sequential(
            ResNetSEBlock1D(64, 128, stride=2, dropout=dropout * 0.5),
            ResNetSEBlock1D(128, 128, stride=1, dropout=dropout * 0.5),
        )
        self.layer3 = nn.Sequential(
            ResNetSEBlock1D(128, 256, stride=2, dropout=dropout),
            ResNetSEBlock1D(256, 256, stride=1, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


def compute_class_weights(train_rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = Counter(row["target_class"] for row in train_rows)
    total = len(train_rows)
    weights = [total / (len(CLASS_NAMES) * counts[name]) for name in CLASS_NAMES]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_train_sampler(train_rows: list[dict[str, str]]) -> WeightedRandomSampler:
    counts = Counter(row["target_class"] for row in train_rows)
    sample_weights = [1.0 / counts[row["target_class"]] for row in train_rows]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_rows),
        replacement=True,
    )


@dataclass
class EvalResult:
    loss: float
    macro_f1: float
    accuracy: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(signals)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

            for truth, pred in zip(labels.cpu().numpy(), preds.cpu().numpy(), strict=False):
                confusion[truth, pred] += 1

    per_precision: dict[str, float] = {}
    per_recall: dict[str, float] = {}
    per_f1: dict[str, float] = {}
    f1_values = []

    for idx, class_name in enumerate(CLASS_NAMES):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - tp)
        fn = float(confusion[idx, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_precision[class_name] = precision
        per_recall[class_name] = recall
        per_f1[class_name] = f1
        f1_values.append(f1)

    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    avg_loss = total_loss / max(total_items, 1)
    return EvalResult(
        loss=avg_loss,
        macro_f1=float(sum(f1_values) / len(f1_values)),
        accuracy=accuracy,
        per_class_precision=per_precision,
        per_class_recall=per_recall,
        per_class_f1=per_f1,
        confusion_matrix=confusion.tolist(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_rows, val_rows, test_rows = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_sampler = build_train_sampler(train_rows) if args.sampler == "weighted" else None

    train_loader = DataLoader(
        ECGDataset(train_rows, augment=args.augment),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        ECGDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        ECGDataset(test_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    if args.model == "baseline":
        model = BaselineECGNet(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    else:
        model = ResNetSEECGNet(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    class_weights = compute_class_weights(train_rows, device) if args.class_weight_mode == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    history: list[dict[str, float]] = []
    best_val_macro_f1 = -1.0
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
            "val_macro_f1": val_result.macro_f1,
            "val_accuracy": val_result.accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))
        if scheduler is not None:
            scheduler.step()

        if val_result.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_result.macro_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "class_names": CLASS_NAMES,
                    "args": vars(args),
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val = evaluate(model, val_loader, criterion, device)
    final_test = evaluate(model, test_loader, criterion, device)

    report = {
        "manifest": str(args.manifest),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": args.model,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "augment": args.augment,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        "sampler": args.sampler,
        "class_weight_mode": args.class_weight_mode,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "class_names": CLASS_NAMES,
        "best_epoch": best_epoch,
        "history": history,
        "validation": {
            "loss": final_val.loss,
            "macro_f1": final_val.macro_f1,
            "accuracy": final_val.accuracy,
            "per_class_precision": final_val.per_class_precision,
            "per_class_recall": final_val.per_class_recall,
            "per_class_f1": final_val.per_class_f1,
            "confusion_matrix": final_val.confusion_matrix,
        },
        "test": {
            "loss": final_test.loss,
            "macro_f1": final_test.macro_f1,
            "accuracy": final_test.accuracy,
            "per_class_precision": final_test.per_class_precision,
            "per_class_recall": final_test.per_class_recall,
            "per_class_f1": final_test.per_class_f1,
            "confusion_matrix": final_test.confusion_matrix,
        },
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
