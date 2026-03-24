from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_uc_sinif_baseline import BaselineECGNet, CLASS_NAMES, CLASS_TO_INDEX
from train_iletim_aux_binary import load_manifest as load_binary_manifest


REPO_ROOT = Path("/home/alper/ekg")
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_iletim_aux_binary.py"
BASE_DIR = REPO_ROOT / "secili_model_adaylari" / "finalist_v2"
BASE_CHECKPOINT = BASE_DIR / "best_model.pt"
OUTPUT_ROOT = BASE_DIR / "iletim_iki_asamali"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
BASE_MACRO_F1 = 0.9382599872565627
BASE_ILETIM_F1 = 0.9046321525885558


class ThreeClassDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        from scipy.io import loadmat

        row = self.rows[index]
        signal = loadmat(row["mat_yolu"])["val"].astype(np.float32)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        signal = (signal - mean) / std
        return torch.from_numpy(signal), CLASS_TO_INDEX[row["target_class"]]


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Komut basarisiz oldu: {' '.join(command)}")


def load_three_class_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dropout = float(checkpoint.get("args", {}).get("dropout", 0.2))
    model = BaselineECGNet(num_classes=3, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_binary_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dropout = float(checkpoint.get("args", {}).get("dropout", 0.2))
    model = BaselineECGNet(num_classes=2, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def collect_outputs(
    base_model: torch.nn.Module,
    aux_model: torch.nn.Module,
    rows: list[dict[str, str]],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    loader = DataLoader(
        ThreeClassDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    base_logits = []
    aux_probs = []
    labels = []
    with torch.no_grad():
        for signals, y in loader:
            signals = signals.to(device, non_blocking=True)
            base_batch = base_model(signals)
            aux_batch = torch.softmax(aux_model(signals), dim=1)[:, 1]
            base_logits.append(base_batch.cpu().numpy())
            aux_probs.append(aux_batch.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(base_logits), np.concatenate(aux_probs), np.concatenate(labels)


def compute_three_class_metrics(confusion: np.ndarray) -> dict[str, object]:
    per_precision = {}
    per_recall = {}
    per_f1 = {}
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


def apply_two_stage(base_logits: np.ndarray, aux_probs: np.ndarray, labels: np.ndarray, positive_threshold: float, negative_threshold: float) -> dict[str, object]:
    preds = base_logits.argmax(axis=1)
    for idx in range(len(preds)):
        if aux_probs[idx] >= positive_threshold:
            preds[idx] = CLASS_TO_INDEX["iletim"]
        elif preds[idx] == CLASS_TO_INDEX["iletim"] and aux_probs[idx] < negative_threshold:
            non_iletim_logits = base_logits[idx].copy()
            non_iletim_logits[CLASS_TO_INDEX["iletim"]] = -1e9
            preds[idx] = int(non_iletim_logits.argmax())
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, pred in zip(labels, preds, strict=False):
        confusion[int(truth), int(pred)] += 1
    metrics = compute_three_class_metrics(confusion)
    metrics["positive_threshold"] = positive_threshold
    metrics["negative_threshold"] = negative_threshold
    metrics["iletim_fn"] = int(confusion[2, 0] + confusion[2, 1])
    metrics["iletim_fp"] = int(confusion[0, 2] + confusion[1, 2])
    metrics["iletim_total_error"] = metrics["iletim_fn"] + metrics["iletim_fp"]
    return metrics


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    aux_dir = OUTPUT_ROOT / "aux_binary"
    run_command(
        [
            str(PYTHON),
            str(TRAIN_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--base-checkpoint",
            str(BASE_CHECKPOINT),
            "--output-dir",
            str(aux_dir),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_three_class_model(BASE_CHECKPOINT, device)
    aux_model = load_binary_model(aux_dir / "best_model.pt", device)
    train_rows, val_rows, test_rows = load_binary_manifest(MANIFEST)
    val_base_logits, val_aux_probs, val_labels = collect_outputs(base_model, aux_model, val_rows, device)
    test_base_logits, test_aux_probs, test_labels = collect_outputs(base_model, aux_model, test_rows, device)

    search_results: list[dict[str, object]] = []
    for pos_t in threshold_values(0.50, 0.90, 0.05):
        for neg_t in threshold_values(0.05, 0.45, 0.05):
            metrics = apply_two_stage(val_base_logits, val_aux_probs, val_labels, pos_t, neg_t)
            search_results.append(
                {
                    "positive_threshold": pos_t,
                    "negative_threshold": neg_t,
                    "val_macro_f1": metrics["macro_f1"],
                    "val_iletim_f1": metrics["per_class_f1"]["iletim"],
                    "val_iletim_fn": metrics["iletim_fn"],
                    "val_iletim_fp": metrics["iletim_fp"],
                    "val_iletim_total_error": metrics["iletim_total_error"],
                }
            )
    write_csv(OUTPUT_ROOT / "threshold_search.csv", search_results)

    best = sorted(
        search_results,
        key=lambda item: (item["val_macro_f1"], item["val_iletim_f1"], -item["val_iletim_total_error"]),
        reverse=True,
    )[0]
    test_metrics = apply_two_stage(
        test_base_logits,
        test_aux_probs,
        test_labels,
        float(best["positive_threshold"]),
        float(best["negative_threshold"]),
    )
    val_metrics = apply_two_stage(
        val_base_logits,
        val_aux_probs,
        val_labels,
        float(best["positive_threshold"]),
        float(best["negative_threshold"]),
    )
    meaningful_gain = (
        test_metrics["per_class_f1"]["iletim"] >= BASE_ILETIM_F1 + 0.005
        and test_metrics["macro_f1"] >= BASE_MACRO_F1 - 0.001
    )
    report = {
        "base_checkpoint": str(BASE_CHECKPOINT),
        "aux_model_dir": str(aux_dir),
        "base_reference": {
            "test_macro_f1": BASE_MACRO_F1,
            "test_iletim_f1": BASE_ILETIM_F1,
        },
        "selected_thresholds": best,
        "validation": val_metrics,
        "test": test_metrics,
        "meaningful_gain": meaningful_gain,
    }
    (OUTPUT_ROOT / "two_stage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
