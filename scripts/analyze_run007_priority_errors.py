from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_uc_sinif_baseline import BaselineECGNet, CLASS_NAMES, CLASS_TO_INDEX, ResNetSEECGNet


PRIORITY_BUCKETS = [
    ("normal", "ritim"),
    ("ritim", "normal"),
    ("ritim", "iletim"),
    ("normal", "iletim"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run007 icin oncelikli hata analizi uretir.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif_final_run007/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/run007_priority_error_analysis"),
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


def parse_codes(raw_codes: str) -> list[str]:
    parts = re.split(r"[;,]", raw_codes)
    return [part.strip() for part in parts if part.strip()]


def safe_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def instantiate_model(checkpoint_args: dict[str, object]) -> nn.Module:
    model_name = str(checkpoint_args.get("model", "baseline"))
    dropout = float(checkpoint_args.get("dropout", 0.2))
    if model_name == "resnet_se":
        return ResNetSEECGNet(num_classes=len(CLASS_NAMES), dropout=dropout)
    return BaselineECGNet(num_classes=len(CLASS_NAMES), dropout=dropout)


def analyze_split(
    model: nn.Module,
    rows: list[dict[str, str]],
    split_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dir: Path,
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
    predictions: list[dict[str, object]] = []
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
                row = rows[int(row_idx)]
                predicted_prob = float(prob_vec[int(pred_idx)])
                margin = float(
                    np.partition(prob_vec, -1)[-1] - np.partition(prob_vec, -2)[-2]
                    if len(prob_vec) >= 2
                    else prob_vec[int(pred_idx)]
                )
                record = {
                    "kayit_id": row["kayit_id"],
                    "hasta_id": row.get("hasta_id", ""),
                    "split": row["split"],
                    "gercek_sinif": CLASS_NAMES[int(truth_idx)],
                    "tahmin_sinif": CLASS_NAMES[int(pred_idx)],
                    "tani_kodlari": row["tani_kodlari"],
                    "yas": row["yas"],
                    "cinsiyet": row["cinsiyet"],
                    "normal_olasilik": float(prob_vec[0]),
                    "ritim_olasilik": float(prob_vec[1]),
                    "iletim_olasilik": float(prob_vec[2]),
                    "tahmin_olasiligi": predicted_prob,
                    "guven_marji": margin,
                    "dogru_mu": bool(int(truth_idx) == int(pred_idx)),
                }
                predictions.append(record)
                if truth_idx != pred_idx:
                    pair_counts[(CLASS_NAMES[int(truth_idx)], CLASS_NAMES[int(pred_idx)])] += 1

    metrics = compute_metrics(confusion)
    misclassified = [item for item in predictions if not item["dogru_mu"]]
    ordered_pairs = [
        {"gercek": truth, "tahmin": pred, "adet": count}
        for (truth, pred), count in pair_counts.most_common()
    ]

    buckets = {}
    bucket_dir = output_dir / split_name
    bucket_dir.mkdir(parents=True, exist_ok=True)
    for truth, pred in PRIORITY_BUCKETS:
        bucket_name = f"{truth}_to_{pred}"
        bucket_records = [
            item
            for item in misclassified
            if item["gercek_sinif"] == truth and item["tahmin_sinif"] == pred
        ]
        bucket_records.sort(key=lambda item: item["tahmin_olasiligi"], reverse=True)
        bucket_summary = summarize_bucket(bucket_records)
        buckets[bucket_name] = bucket_summary
        write_csv(bucket_dir / f"{bucket_name}.csv", bucket_records)

    write_csv(bucket_dir / "all_misclassified.csv", misclassified)
    write_csv(
        bucket_dir / "top_confidence_errors.csv",
        sorted(misclassified, key=lambda item: item["tahmin_olasiligi"], reverse=True)[:200],
    )

    return {
        "split": split_name,
        "size": len(rows),
        "metrics": metrics,
        "top_error_pairs": ordered_pairs[:10],
        "priority_buckets": buckets,
        "misclassified_count": len(misclassified),
        "all_misclassified_csv": str(bucket_dir / "all_misclassified.csv"),
        "top_confidence_errors_csv": str(bucket_dir / "top_confidence_errors.csv"),
    }


def compute_metrics(confusion: np.ndarray) -> dict[str, object]:
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
    return {
        "accuracy": accuracy,
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "per_class_precision": per_precision,
        "per_class_recall": per_recall,
        "per_class_f1": per_f1,
        "confusion_matrix": confusion.tolist(),
    }


def summarize_bucket(records: list[dict[str, object]]) -> dict[str, object]:
    code_counter: Counter[str] = Counter()
    combo_counter: Counter[str] = Counter()
    sex_counter: Counter[str] = Counter()
    ages: list[float] = []
    confidences: list[float] = []
    margins: list[float] = []
    by_patient: Counter[str] = Counter()

    for item in records:
        codes = parse_codes(str(item["tani_kodlari"]))
        code_counter.update(codes)
        combo_counter[str(item["tani_kodlari"])] += 1
        sex_counter[str(item["cinsiyet"] or "unknown")] += 1
        age_value = safe_float(str(item["yas"]))
        if age_value is not None and not math.isnan(age_value):
            ages.append(age_value)
        confidences.append(float(item["tahmin_olasiligi"]))
        margins.append(float(item["guven_marji"]))
        by_patient[str(item["hasta_id"])] += 1

    return {
        "count": len(records),
        "top_codes": [{"kod": code, "adet": count} for code, count in code_counter.most_common(10)],
        "top_code_combinations": [
            {"kombinasyon": combo, "adet": count}
            for combo, count in combo_counter.most_common(10)
        ],
        "sex_distribution": dict(sex_counter),
        "age_stats": summarize_numeric(ages),
        "confidence_stats": summarize_numeric(confidences),
        "margin_stats": summarize_numeric(margins),
        "repeated_patient_count": int(sum(1 for count in by_patient.values() if count > 1)),
        "top_confident_examples": records[:20],
    }


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values_np.size),
        "mean": float(values_np.mean()),
        "median": float(np.median(values_np)),
        "min": float(values_np.min()),
        "max": float(values_np.max()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_priority_plan(test_summary: dict[str, object]) -> list[dict[str, object]]:
    buckets = test_summary["priority_buckets"]
    plan = []
    for rank, bucket_name in enumerate(
        ["normal_to_ritim", "ritim_to_normal", "ritim_to_iletim", "normal_to_iletim"],
        start=1,
    ):
        summary = buckets[bucket_name]
        top_codes = [item["kod"] for item in summary["top_codes"][:5]]
        plan.append(
            {
                "oncelik": rank,
                "bucket": bucket_name,
                "adet": summary["count"],
                "neden": priority_reason(bucket_name),
                "onerilen_sonraki_aksiyon": priority_action(bucket_name),
                "one_cikan_tani_kodlari": top_codes,
            }
        )
    return plan


def priority_reason(bucket_name: str) -> str:
    reasons = {
        "normal_to_ritim": "Normal kayitlarin ritim olarak asiri etiketlenmesi ritim precision tarafini bozuyor.",
        "ritim_to_normal": "Ritim kayitlarinin normal gorulmesi ritim recall tarafindaki temel kayip alani.",
        "ritim_to_iletim": "Ritim ile iletim arasindaki daha kucuk ama klinik olarak hassas sinir.",
        "normal_to_iletim": "Iletim precision zaten guclu oldugu icin bu bucket daha dusuk oncelikli.",
    }
    return reasons[bucket_name]


def priority_action(bucket_name: str) -> str:
    actions = {
        "normal_to_ritim": "Kod bazli hard-example listesi ile hedefli oversampling veya pair-aware loss.",
        "ritim_to_normal": "Yanlis siniflanan ritim alt tipleri icin hedefli replay ve etiket siniri kontrolu.",
        "ritim_to_iletim": "Iletim kararlari icin ek calibration ve hata kodu incelemesi.",
        "normal_to_iletim": "Dusuk hacimli bucket; sadece ortak kodlar belirginse ele alinmali.",
    }
    return actions[bucket_name]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_rows(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = instantiate_model(checkpoint_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    validation = analyze_split(
        model=model,
        rows=splits["dogrulama"],
        split_name="dogrulama",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        output_dir=args.output_dir,
    )
    test = analyze_split(
        model=model,
        rows=splits["test"],
        split_name="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        output_dir=args.output_dir,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "device": str(device),
        "model": checkpoint_args.get("model", "baseline"),
        "best_epoch": checkpoint.get("epoch"),
        "validation": validation,
        "test": test,
        "priority_plan": build_priority_plan(test),
    }
    (args.output_dir / "priority_error_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
