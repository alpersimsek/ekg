from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_uc_sinif_baseline import BaselineECGNet, CLASS_NAMES, CLASS_TO_INDEX, ResNetSEECGNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3 sinif EKG modeli icin temperature scaling uygular.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


class ECGDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        signal = loadmat(row["mat_yolu"])["val"].astype(np.float32)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        signal = (signal - mean) / std
        return torch.from_numpy(signal), CLASS_TO_INDEX[row["target_class"]]


def load_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    splits = {"egitim": [], "dogrulama": [], "test": []}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["target_class"] in CLASS_TO_INDEX and row["split"] in splits:
                splits[row["split"]].append(row)
    return splits


def build_model(checkpoint: dict[str, object], device: torch.device) -> nn.Module:
    args = checkpoint.get("args", {})
    model_name = str(args.get("model", "baseline"))
    dropout = float(args.get("dropout", 0.2))
    if model_name == "resnet_se":
        model = ResNetSEECGNet(num_classes=len(CLASS_NAMES), dropout=dropout)
    else:
        model = BaselineECGNet(num_classes=len(CLASS_NAMES), dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def collect_logits(
    model: nn.Module,
    rows: list[dict[str, str]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        ECGDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    logits_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for signals, labels in loader:
            logits = model(signals.to(device, non_blocking=True))
            logits_list.append(logits.cpu())
            labels_list.append(labels)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict[str, object]:
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=1)
    preds = probs.argmax(dim=1)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for truth, pred in zip(labels.numpy(), preds.numpy(), strict=False):
        confusion[int(truth), int(pred)] += 1

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

    nll = float(F.cross_entropy(scaled_logits, labels).item())
    brier = float(torch.mean(torch.sum((probs - F.one_hot(labels, num_classes=len(CLASS_NAMES)).float()) ** 2, dim=1)).item())
    ece = expected_calibration_error(probs, labels)
    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    return {
        "temperature": temperature,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "accuracy": accuracy,
        "per_class_precision": per_precision,
        "per_class_recall": per_recall,
        "per_class_f1": per_f1,
        "confusion_matrix": confusion.tolist(),
    }


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0.0, 1.0, steps=n_bins + 1)
    ece = torch.zeros(1, dtype=torch.float32)
    for lower, upper in zip(bins[:-1], bins[1:], strict=False):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].float().mean()
            ece += (mask.float().mean()) * torch.abs(bin_conf - bin_acc)
    return float(ece.item())


def optimize_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.clamp(temperature, min=1e-3)
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(temperature.detach(), min=1e-3).item())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint, device)

    val_logits, val_labels = collect_logits(model, splits["dogrulama"], args.batch_size, args.num_workers, device)
    test_logits, test_labels = collect_logits(model, splits["test"], args.batch_size, args.num_workers, device)

    temperature = optimize_temperature(val_logits, val_labels)

    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "optimized_temperature": temperature,
        "validation_uncalibrated": compute_metrics(val_logits, val_labels, 1.0),
        "validation_calibrated": compute_metrics(val_logits, val_labels, temperature),
        "test_uncalibrated": compute_metrics(test_logits, test_labels, 1.0),
        "test_calibrated": compute_metrics(test_logits, test_labels, temperature),
        "note": "Temperature scaling olasilik kalibrasyonu icindir; argmax tahminleri sabit kalacagi icin Macro F1 ve confusion matrix degismeyebilir.",
    }
    (args.output_dir / "temperature_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
