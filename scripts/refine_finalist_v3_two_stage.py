from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_finalist_v2_iletim_two_stage import (
    BASE_ILETIM_F1,
    BASE_MACRO_F1,
    ThreeClassDataset,
    apply_two_stage,
    collect_outputs,
    load_binary_model,
    load_three_class_model,
    threshold_values,
)
from train_iletim_aux_binary import load_manifest as load_binary_manifest


REPO_ROOT = Path("/home/alper/ekg")
BASE_DIR = REPO_ROOT / "secili_model_adaylari" / "finalist_v3_aday"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
BASE_CHECKPOINT = BASE_DIR / "base_best_model.pt"
AUX_CHECKPOINT = BASE_DIR / "aux_binary" / "best_model.pt"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_three_class_model(BASE_CHECKPOINT, device)
    aux_model = load_binary_model(AUX_CHECKPOINT, device)
    _train_rows, val_rows, test_rows = load_binary_manifest(MANIFEST)
    val_base_logits, val_aux_probs, val_labels = collect_outputs(base_model, aux_model, val_rows, device)
    test_base_logits, test_aux_probs, test_labels = collect_outputs(base_model, aux_model, test_rows, device)

    search_results: list[dict[str, object]] = []
    for pos_t in threshold_values(0.78, 0.92, 0.01):
        for neg_t in threshold_values(0.18, 0.32, 0.01):
            metrics = apply_two_stage(val_base_logits, val_aux_probs, val_labels, pos_t, neg_t)
            search_results.append(
                {
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
    write_csv(BASE_DIR / "threshold_refine.csv", search_results)

    best = sorted(
        search_results,
        key=lambda item: (item["val_macro_f1"], item["val_iletim_f1"], -item["val_iletim_total_error"]),
        reverse=True,
    )[0]

    validation = apply_two_stage(
        val_base_logits,
        val_aux_probs,
        val_labels,
        float(best["positive_threshold"]),
        float(best["negative_threshold"]),
    )
    test = apply_two_stage(
        test_base_logits,
        test_aux_probs,
        test_labels,
        float(best["positive_threshold"]),
        float(best["negative_threshold"]),
    )
    report = {
        "base_reference": {
            "test_macro_f1": BASE_MACRO_F1,
            "test_iletim_f1": BASE_ILETIM_F1,
        },
        "selected_thresholds": best,
        "validation": validation,
        "test": test,
        "meaningful_gain": (
            test["per_class_f1"]["iletim"] >= BASE_ILETIM_F1 + 0.003
            and test["macro_f1"] >= BASE_MACRO_F1
        ),
    }
    (BASE_DIR / "v3_two_stage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
