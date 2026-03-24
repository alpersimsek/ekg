from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_finalist_v2_iletim_two_stage import collect_outputs, load_binary_model, load_three_class_model
from train_iletim_aux_binary import load_manifest as load_binary_manifest
from train_uc_sinif_baseline import CLASS_NAMES, CLASS_TO_INDEX


REPO_ROOT = Path("/home/alper/ekg")
BASE_DIR = REPO_ROOT / "secili_model_adaylari" / "finalist_v4_fusion_aday"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
BASE_CHECKPOINT = BASE_DIR / "base_best_model.pt"
AUX_CHECKPOINT = BASE_DIR / "aux_binary" / "best_model.pt"
BASE_MACRO_F1 = 0.9396172738374782
BASE_ILETIM_F1 = 0.9086021505376344


def frange(start: float, stop: float, step: float) -> list[float]:
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


def compute_metrics(confusion: np.ndarray) -> dict[str, object]:
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


def apply_fusion(
    base_probs: np.ndarray,
    base_logits: np.ndarray,
    aux_probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    beta: float,
    positive_threshold: float,
    negative_threshold: float,
) -> dict[str, object]:
    preds = base_logits.argmax(axis=1)
    iletim_idx = CLASS_TO_INDEX["iletim"]
    for idx in range(len(preds)):
        fused_score = alpha * float(base_probs[idx, iletim_idx]) + beta * float(aux_probs[idx])
        if fused_score >= positive_threshold:
            preds[idx] = iletim_idx
        elif preds[idx] == iletim_idx and fused_score < negative_threshold:
            masked = base_logits[idx].copy()
            masked[iletim_idx] = -1e9
            preds[idx] = int(masked.argmax())
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, pred in zip(labels, preds, strict=False):
        confusion[int(truth), int(pred)] += 1
    metrics = compute_metrics(confusion)
    metrics["alpha"] = alpha
    metrics["beta"] = beta
    metrics["positive_threshold"] = positive_threshold
    metrics["negative_threshold"] = negative_threshold
    metrics["iletim_fn"] = int(confusion[2, 0] + confusion[2, 1])
    metrics["iletim_fp"] = int(confusion[0, 2] + confusion[1, 2])
    metrics["iletim_total_error"] = metrics["iletim_fn"] + metrics["iletim_fp"]
    return metrics


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_three_class_model(BASE_CHECKPOINT, device)
    aux_model = load_binary_model(AUX_CHECKPOINT, device)
    _train_rows, val_rows, test_rows = load_binary_manifest(MANIFEST)

    val_base_logits, val_aux_probs, val_labels = collect_outputs(base_model, aux_model, val_rows, device)
    test_base_logits, test_aux_probs, test_labels = collect_outputs(base_model, aux_model, test_rows, device)
    val_base_probs = torch.softmax(torch.tensor(val_base_logits), dim=1).numpy()
    test_base_probs = torch.softmax(torch.tensor(test_base_logits), dim=1).numpy()

    search_rows: list[dict[str, object]] = []
    for alpha in frange(0.2, 1.2, 0.1):
        for beta in frange(0.2, 1.4, 0.1):
            for pos_t in frange(0.40, 0.95, 0.05):
                for neg_t in frange(0.05, 0.35, 0.05):
                    metrics = apply_fusion(
                        val_base_probs,
                        val_base_logits,
                        val_aux_probs,
                        val_labels,
                        alpha,
                        beta,
                        pos_t,
                        neg_t,
                    )
                    search_rows.append(
                        {
                            "alpha": alpha,
                            "beta": beta,
                            "positive_threshold": pos_t,
                            "negative_threshold": neg_t,
                            "val_macro_f1": metrics["macro_f1"],
                            "val_iletim_f1": metrics["per_class_f1"]["iletim"],
                            "val_iletim_precision": metrics["per_class_precision"]["iletim"],
                            "val_iletim_recall": metrics["per_class_recall"]["iletim"],
                            "val_iletim_fn": metrics["iletim_fn"],
                            "val_iletim_fp": metrics["iletim_fp"],
                            "val_iletim_total_error": metrics["iletim_total_error"],
                        }
                    )

    write_csv(BASE_DIR / "fusion_sweep.csv", search_rows)

    best = sorted(
        search_rows,
        key=lambda item: (item["val_macro_f1"], item["val_iletim_f1"], -item["val_iletim_total_error"]),
        reverse=True,
    )[0]
    validation = apply_fusion(
        val_base_probs,
        val_base_logits,
        val_aux_probs,
        val_labels,
        float(best["alpha"]),
        float(best["beta"]),
        float(best["positive_threshold"]),
        float(best["negative_threshold"]),
    )
    test = apply_fusion(
        test_base_probs,
        test_base_logits,
        test_aux_probs,
        test_labels,
        float(best["alpha"]),
        float(best["beta"]),
        float(best["positive_threshold"]),
        float(best["negative_threshold"]),
    )
    report = {
        "base_reference": {
            "test_macro_f1": BASE_MACRO_F1,
            "test_iletim_f1": BASE_ILETIM_F1,
        },
        "selected_params": best,
        "validation": validation,
        "test": test,
        "meaningful_gain": (
            test["per_class_f1"]["iletim"] >= BASE_ILETIM_F1 + 0.002
            and test["macro_f1"] >= BASE_MACRO_F1
        ),
    }
    (BASE_DIR / "fusion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
