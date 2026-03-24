from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_uc_sinif_baseline import (
    BaselineECGNet,
    CLASS_NAMES,
    ResNetSEECGNet,
    evaluate,
    set_seed,
    train_one_epoch,
)
from train_uc_sinif_bucket_aware import ECGDataset, compute_class_weights, load_manifest, parse_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="finalist_v2 icin iletim odakli dar fine-tuning.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/alper/ekg/secili_model_adaylari/finalist_v2/iletim_sweep"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--augment", choices=["none", "light"], default="light")
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--sampler", choices=["none", "weighted"], default="weighted")
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="none")
    parser.add_argument("--model", choices=["baseline", "resnet_se"], default="baseline")
    parser.add_argument("--focus-normal-codes", nargs="*", default=["426783006"])
    parser.add_argument("--focus-ritim-codes", nargs="*", default=["427393009", "426177001", "284470004"])
    parser.add_argument(
        "--focus-iletim-codes",
        nargs="*",
        default=["59118001", "270492004", "10370003", "164909002", "445118002", "713426002"],
    )
    parser.add_argument("--focus-normal-weight", type=float, default=1.5)
    parser.add_argument("--focus-ritim-weight", type=float, default=2.0)
    parser.add_argument("--focus-iletim-weight", type=float, default=1.25)
    parser.add_argument("--max-focus-weight", type=float, default=8.0)
    return parser.parse_args()


def build_iletim_focus_sampler(
    train_rows: list[dict[str, str]],
    sampler_mode: str,
    focus_normal_codes: set[str],
    focus_ritim_codes: set[str],
    focus_iletim_codes: set[str],
    focus_normal_weight: float,
    focus_ritim_weight: float,
    focus_iletim_weight: float,
    max_focus_weight: float,
) -> tuple[WeightedRandomSampler | None, dict[str, object]]:
    if sampler_mode == "none":
        return None, {
            "sampler_mode": "none",
            "focus_normal_hits": 0,
            "focus_ritim_hits": 0,
            "focus_iletim_hits": 0,
            "avg_weight": None,
        }

    counts = Counter(row["target_class"] for row in train_rows)
    sample_weights = []
    focus_normal_hits = 0
    focus_ritim_hits = 0
    focus_iletim_hits = 0

    for row in train_rows:
        weight = 1.0 / counts[row["target_class"]]
        codes = parse_codes(row["tani_kodlari"])
        if row["target_class"] == "normal" and codes.intersection(focus_normal_codes):
            weight *= focus_normal_weight
            focus_normal_hits += 1
        if row["target_class"] == "ritim" and codes.intersection(focus_ritim_codes):
            weight *= focus_ritim_weight
            focus_ritim_hits += 1
        if row["target_class"] == "iletim" and codes.intersection(focus_iletim_codes):
            weight *= focus_iletim_weight
            focus_iletim_hits += 1
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
        "focus_iletim_hits": focus_iletim_hits,
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
    focus_iletim_codes = set(args.focus_iletim_codes)
    train_sampler, sampler_stats = build_iletim_focus_sampler(
        train_rows=train_rows,
        sampler_mode=args.sampler,
        focus_normal_codes=focus_normal_codes,
        focus_ritim_codes=focus_ritim_codes,
        focus_iletim_codes=focus_iletim_codes,
        focus_normal_weight=args.focus_normal_weight,
        focus_ritim_weight=args.focus_ritim_weight,
        focus_iletim_weight=args.focus_iletim_weight,
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

    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
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
        "resume_checkpoint": str(args.resume_checkpoint),
        "sampler": args.sampler,
        "class_weight_mode": args.class_weight_mode,
        "focus_normal_codes": sorted(focus_normal_codes),
        "focus_ritim_codes": sorted(focus_ritim_codes),
        "focus_iletim_codes": sorted(focus_iletim_codes),
        "focus_normal_weight": args.focus_normal_weight,
        "focus_ritim_weight": args.focus_ritim_weight,
        "focus_iletim_weight": args.focus_iletim_weight,
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
