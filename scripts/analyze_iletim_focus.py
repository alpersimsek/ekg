from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_uc_sinif_baseline import BaselineECGNet, CLASS_NAMES, CLASS_TO_INDEX, ResNetSEECGNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iletim sinifi icin FN/FP odakli analiz uretir.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/alper/ekg/secili_model_adaylari/finalist_v2/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/alper/ekg/secili_model_adaylari/finalist_v2/iletim_analysis"),
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
    splits = {"dogrulama": [], "test": []}
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


def parse_codes(raw_codes: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", raw_codes) if part.strip()]


def safe_float(raw: str) -> float | None:
    try:
        return float(raw)
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    code_counter: Counter[str] = Counter()
    combo_counter: Counter[str] = Counter()
    sex_counter: Counter[str] = Counter()
    ages = []
    iletim_probs = []
    margins = []
    for item in records:
        code_counter.update(parse_codes(str(item["tani_kodlari"])))
        combo_counter[str(item["tani_kodlari"])] += 1
        sex_counter[str(item["cinsiyet"] or "unknown")] += 1
        age = safe_float(str(item["yas"]))
        if age is not None:
            ages.append(age)
        iletim_probs.append(float(item["iletim_olasilik"]))
        margins.append(float(item["guven_marji"]))
    return {
        "count": len(records),
        "top_codes": [{"kod": code, "adet": count} for code, count in code_counter.most_common(10)],
        "top_code_combinations": [{"kombinasyon": combo, "adet": count} for combo, count in combo_counter.most_common(10)],
        "sex_distribution": dict(sex_counter),
        "age_mean": float(np.mean(ages)) if ages else None,
        "iletim_prob_mean": float(np.mean(iletim_probs)) if iletim_probs else None,
        "margin_mean": float(np.mean(margins)) if margins else None,
        "top_examples": records[:20],
    }


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
    iletim_fn = []
    iletim_fp = []
    iletim_tp = []

    model.eval()
    with torch.no_grad():
        for signals, labels, indices in loader:
            logits = model(signals.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for truth_idx, pred_idx, row_idx, prob_vec in zip(labels.numpy(), preds, indices.numpy(), probs, strict=False):
                row = rows[int(row_idx)]
                record = {
                    "kayit_id": row["kayit_id"],
                    "hasta_id": row.get("hasta_id", ""),
                    "split": split_name,
                    "gercek_sinif": CLASS_NAMES[int(truth_idx)],
                    "tahmin_sinif": CLASS_NAMES[int(pred_idx)],
                    "tani_kodlari": row["tani_kodlari"],
                    "yas": row["yas"],
                    "cinsiyet": row["cinsiyet"],
                    "normal_olasilik": float(prob_vec[0]),
                    "ritim_olasilik": float(prob_vec[1]),
                    "iletim_olasilik": float(prob_vec[2]),
                    "guven_marji": float(np.partition(prob_vec, -1)[-1] - np.partition(prob_vec, -2)[-2]),
                }
                if truth_idx == CLASS_TO_INDEX["iletim"] and pred_idx != CLASS_TO_INDEX["iletim"]:
                    iletim_fn.append(record)
                elif truth_idx != CLASS_TO_INDEX["iletim"] and pred_idx == CLASS_TO_INDEX["iletim"]:
                    iletim_fp.append(record)
                elif truth_idx == CLASS_TO_INDEX["iletim"] and pred_idx == CLASS_TO_INDEX["iletim"]:
                    iletim_tp.append(record)

    iletim_fn.sort(key=lambda x: x["iletim_olasilik"])
    iletim_fp.sort(key=lambda x: x["iletim_olasilik"], reverse=True)
    iletim_tp.sort(key=lambda x: x["iletim_olasilik"], reverse=True)

    split_dir = output_dir / split_name
    write_csv(split_dir / "iletim_false_negative.csv", iletim_fn)
    write_csv(split_dir / "iletim_false_positive.csv", iletim_fp)
    write_csv(split_dir / "iletim_true_positive.csv", iletim_tp[:200])

    return {
        "split": split_name,
        "iletim_false_negative": summarize(iletim_fn),
        "iletim_false_positive": summarize(iletim_fp),
        "iletim_true_positive": summarize(iletim_tp),
        "iletim_fn_csv": str(split_dir / "iletim_false_negative.csv"),
        "iletim_fp_csv": str(split_dir / "iletim_false_positive.csv"),
        "iletim_tp_csv": str(split_dir / "iletim_true_positive.csv"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_rows(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint, device)

    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "validation": analyze_split(model, splits["dogrulama"], "dogrulama", args.batch_size, args.num_workers, device, args.output_dir),
        "test": analyze_split(model, splits["test"], "test", args.batch_size, args.num_workers, device, args.output_dir),
    }
    (args.output_dir / "iletim_focus_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
