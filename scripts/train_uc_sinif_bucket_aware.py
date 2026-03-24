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

from train_uc_sinif_baseline import (
    BaselineECGNet,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    ResNetSEECGNet,
    evaluate,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run007 icin bucket-aware 3 sinif EKG egitimi.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif_bucket_aware"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--augment", choices=["none", "light"], default="none")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--sampler", choices=["none", "weighted"], default="weighted")
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="none")
    parser.add_argument("--model", choices=["baseline", "resnet_se"], default="baseline")
    parser.add_argument(
        "--focus-normal-codes",
        nargs="*",
        default=["426783006"],
        help="normal->ritim bucket icin hedef kodlar",
    )
    parser.add_argument(
        "--focus-ritim-codes",
        nargs="*",
        default=["427393009", "426177001", "284470004"],
        help="ritim->normal bucket icin hedef kodlar",
    )
    parser.add_argument(
        "--focus-normal-weight",
        type=float,
        default=2.0,
        help="hedef normal kodlari icin ornek agirligi carpani",
    )
    parser.add_argument(
        "--focus-ritim-weight",
        type=float,
        default=2.0,
        help="hedef ritim kodlari icin ornek agirligi carpani",
    )
    parser.add_argument(
        "--max-focus-weight",
        type=float,
        default=8.0,
        help="tek bir ornegin alabilecegi en yuksek sampler agirligi",
    )
    return parser.parse_args()


def parse_codes(raw_codes: str) -> set[str]:
    return {part.strip() for part in raw_codes.replace(";", ",").split(",") if part.strip()}


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
            scale = np.random.uniform(0.95, 1.05, size=(signal.shape[0], 1)).astype(np.float32)
            signal = signal * scale
            shift = np.random.randint(-20, 21)
            if shift != 0:
                signal = np.roll(signal, shift=shift, axis=1)
            signal = signal + np.random.normal(0.0, 0.01, size=signal.shape).astype(np.float32)
        return torch.from_numpy(signal), CLASS_TO_INDEX[row["target_class"]]


def compute_class_weights(train_rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = Counter(row["target_class"] for row in train_rows)
    total = len(train_rows)
    weights = [total / (len(CLASS_NAMES) * counts[name]) for name in CLASS_NAMES]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_bucket_aware_sampler(
    train_rows: list[dict[str, str]],
    sampler_mode: str,
    focus_normal_codes: set[str],
    focus_ritim_codes: set[str],
    focus_normal_weight: float,
    focus_ritim_weight: float,
    max_focus_weight: float,
) -> tuple[WeightedRandomSampler | None, dict[str, object]]:
    if sampler_mode == "none":
        return None, {
            "sampler_mode": "none",
            "focus_normal_hits": 0,
            "focus_ritim_hits": 0,
            "avg_weight": None,
        }

    counts = Counter(row["target_class"] for row in train_rows)
    sample_weights = []
    focus_normal_hits = 0
    focus_ritim_hits = 0

    for row in train_rows:
        weight = 1.0 / counts[row["target_class"]]
        codes = parse_codes(row["tani_kodlari"])
        if row["target_class"] == "normal" and codes.intersection(focus_normal_codes):
            weight *= focus_normal_weight
            focus_normal_hits += 1
        if row["target_class"] == "ritim" and codes.intersection(focus_ritim_codes):
            weight *= focus_ritim_weight
            focus_ritim_hits += 1
        weight = min(weight, max_focus_weight / len(train_rows))
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_rows),
        replacement=True,
    )
    return sampler, {
        "sampler_mode": sampler_mode,
        "focus_normal_hits": focus_normal_hits,
        "focus_ritim_hits": focus_ritim_hits,
        "avg_weight": float(np.mean(sample_weights)),
        "max_weight": float(np.max(sample_weights)),
        "min_weight": float(np.min(sample_weights)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_rows, val_rows, test_rows = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    focus_normal_codes = set(args.focus_normal_codes)
    focus_ritim_codes = set(args.focus_ritim_codes)
    train_sampler, sampler_stats = build_bucket_aware_sampler(
        train_rows=train_rows,
        sampler_mode=args.sampler,
        focus_normal_codes=focus_normal_codes,
        focus_ritim_codes=focus_ritim_codes,
        focus_normal_weight=args.focus_normal_weight,
        focus_ritim_weight=args.focus_ritim_weight,
        max_focus_weight=args.max_focus_weight,
    )

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
        "focus_normal_codes": sorted(focus_normal_codes),
        "focus_ritim_codes": sorted(focus_ritim_codes),
        "focus_normal_weight": args.focus_normal_weight,
        "focus_ritim_weight": args.focus_ritim_weight,
        "sampler_stats": sampler_stats,
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
