from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_uc_sinif_baseline import BaselineECGNet, CLASS_NAMES, CLASS_TO_INDEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3 sinif modeli icin hata analizi yapar.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif_baseline/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif_analysis"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        signal = loadmat(row["mat_yolu"])["val"].astype(np.float32)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        signal = (signal - mean) / std
        return torch.from_numpy(signal), CLASS_TO_INDEX[row["target_class"]], index


def load_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    splits = {"egitim": [], "dogrulama": [], "test": []}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["target_class"] in CLASS_TO_INDEX and row["split"] in splits:
                splits[row["split"]].append(row)
    return splits


def analyze_split(
    model: nn.Module,
    rows: list[dict[str, str]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, object]:
    loader = DataLoader(
        ManifestDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    misclassified: list[dict[str, object]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()

    model.eval()
    with torch.no_grad():
        for signals, labels, indices in loader:
            logits = model(signals.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            labels_np = labels.numpy()
            indices_np = indices.numpy()
            for truth_idx, pred_idx, row_idx, prob_vec in zip(labels_np, preds, indices_np, probs, strict=False):
                confusion[truth_idx, pred_idx] += 1
                if truth_idx != pred_idx:
                    row = rows[int(row_idx)]
                    pair_counts[(CLASS_NAMES[int(truth_idx)], CLASS_NAMES[int(pred_idx)])] += 1
                    misclassified.append(
                        {
                            "kayit_id": row["kayit_id"],
                            "split": row["split"],
                            "gercek_sinif": CLASS_NAMES[int(truth_idx)],
                            "tahmin_sinif": CLASS_NAMES[int(pred_idx)],
                            "tani_kodlari": row["tani_kodlari"],
                            "yas": row["yas"],
                            "cinsiyet": row["cinsiyet"],
                            "normal_olasilik": float(prob_vec[0]),
                            "ritim_olasilik": float(prob_vec[1]),
                            "iletim_olasilik": float(prob_vec[2]),
                        }
                    )

    return {
        "confusion_matrix": confusion.tolist(),
        "top_error_pairs": [
            {"gercek": truth, "tahmin": pred, "adet": count}
            for (truth, pred), count in pair_counts.most_common(10)
        ],
        "misclassified_count": len(misclassified),
        "misclassified_examples": misclassified[:100],
        "iletim_false_positive_examples": [
            item for item in misclassified if item["tahmin_sinif"] == "iletim" and item["gercek_sinif"] != "iletim"
        ][:50],
        "ritim_to_normal_examples": [
            item for item in misclassified if item["gercek_sinif"] == "ritim" and item["tahmin_sinif"] == "normal"
        ][:50],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_rows(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaselineECGNet(num_classes=len(CLASS_NAMES)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "validation": analyze_split(model, splits["dogrulama"], args.batch_size, args.num_workers, device),
        "test": analyze_split(model, splits["test"], args.batch_size, args.num_workers, device),
    }
    (args.output_dir / "error_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
