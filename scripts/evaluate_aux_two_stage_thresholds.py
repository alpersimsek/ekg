from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_finalist_v2_iletim_two_stage import apply_two_stage, collect_outputs, load_three_class_model
from train_iletim_aux_binary import load_manifest as load_binary_manifest
from train_uc_sinif_baseline import BaselineECGNet, ResNetSEECGNet


def frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary yardimci modeli iki-asamali 3-sinif sistem olarak degerlendirir.")
    parser.add_argument("--manifest", type=Path, default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"))
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--aux-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-macro-f1", type=float, required=True)
    parser.add_argument("--base-iletim-f1", type=float, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_three_class_model(args.base_checkpoint, device)
    checkpoint = torch.load(args.aux_checkpoint, map_location=device, weights_only=False)
    model_name = str(checkpoint.get("args", {}).get("model", "baseline"))
    dropout = float(checkpoint.get("args", {}).get("dropout", 0.2))
    if model_name == "resnet_se":
        aux_model = ResNetSEECGNet(num_classes=2, dropout=dropout)
    else:
        aux_model = BaselineECGNet(num_classes=2, dropout=dropout)
    aux_model.load_state_dict(checkpoint["model_state_dict"])
    aux_model.to(device)
    aux_model.eval()
    _train_rows, val_rows, test_rows = load_binary_manifest(args.manifest)
    val_base_logits, val_aux_probs, val_labels = collect_outputs(base_model, aux_model, val_rows, device)
    test_base_logits, test_aux_probs, test_labels = collect_outputs(base_model, aux_model, test_rows, device)

    search_results = []
    for pos_t in frange(0.50, 0.90, 0.05):
        for neg_t in frange(0.05, 0.35, 0.05):
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
    write_csv(args.output_dir / "threshold_search.csv", search_results)
    best = sorted(search_results, key=lambda item: (item["val_macro_f1"], item["val_iletim_f1"], -item["val_iletim_total_error"]), reverse=True)[0]
    validation = apply_two_stage(val_base_logits, val_aux_probs, val_labels, float(best["positive_threshold"]), float(best["negative_threshold"]))
    test = apply_two_stage(test_base_logits, test_aux_probs, test_labels, float(best["positive_threshold"]), float(best["negative_threshold"]))
    report = {
        "base_reference": {"test_macro_f1": args.base_macro_f1, "test_iletim_f1": args.base_iletim_f1},
        "selected_thresholds": best,
        "validation": validation,
        "test": test,
        "meaningful_gain": (
            test["per_class_f1"]["iletim"] >= args.base_iletim_f1 + 0.003
            and test["macro_f1"] >= args.base_macro_f1
        ),
    }
    (args.output_dir / "two_stage_eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
