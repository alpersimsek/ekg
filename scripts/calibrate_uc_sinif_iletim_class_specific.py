from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from calibrate_uc_sinif_temperature import build_model, collect_logits, expected_calibration_error, load_manifest
from train_uc_sinif_baseline import CLASS_NAMES


ILETIM_INDEX = CLASS_NAMES.index("iletim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="iletim sinifi icin sinif-ozel logit kalibrasyonu dener.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bias-min", type=float, default=-0.6)
    parser.add_argument("--bias-max", type=float, default=0.6)
    parser.add_argument("--bias-step", type=float, default=0.05)
    parser.add_argument("--temp-min", type=float, default=0.75)
    parser.add_argument("--temp-max", type=float, default=1.25)
    parser.add_argument("--temp-step", type=float, default=0.05)
    parser.add_argument("--macro-floor-delta", type=float, default=0.002)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def apply_iletim_calibration(logits: torch.Tensor, iletim_bias: float, iletim_temperature: float) -> torch.Tensor:
    adjusted = logits.clone()
    adjusted[:, ILETIM_INDEX] = adjusted[:, ILETIM_INDEX] / iletim_temperature + iletim_bias
    return adjusted


def compute_metrics(adjusted_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    probs = torch.softmax(adjusted_logits, dim=1)
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

    nll = float(F.cross_entropy(adjusted_logits, labels).item())
    brier = float(torch.mean(torch.sum((probs - F.one_hot(labels, num_classes=len(CLASS_NAMES)).float()) ** 2, dim=1)).item())
    ece = expected_calibration_error(probs, labels)
    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    iletim_fn = int(confusion[ILETIM_INDEX, :].sum() - confusion[ILETIM_INDEX, ILETIM_INDEX])
    iletim_fp = int(confusion[:, ILETIM_INDEX].sum() - confusion[ILETIM_INDEX, ILETIM_INDEX])
    return {
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "accuracy": accuracy,
        "per_class_precision": per_precision,
        "per_class_recall": per_recall,
        "per_class_f1": per_f1,
        "confusion_matrix": confusion.tolist(),
        "iletim_fn": iletim_fn,
        "iletim_fp": iletim_fp,
        "iletim_total_error": iletim_fn + iletim_fp,
    }


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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint, device)

    val_logits, val_labels = collect_logits(model, splits["dogrulama"], args.batch_size, args.num_workers, device)
    test_logits, test_labels = collect_logits(model, splits["test"], args.batch_size, args.num_workers, device)

    base_validation = compute_metrics(val_logits, val_labels)
    base_test = compute_metrics(test_logits, test_labels)
    macro_floor = base_validation["macro_f1"] - args.macro_floor_delta

    search_rows: list[dict[str, object]] = []
    biases = frange(args.bias_min, args.bias_max, args.bias_step)
    temperatures = frange(args.temp_min, args.temp_max, args.temp_step)

    for iletim_bias in biases:
        for iletim_temperature in temperatures:
            val_metrics = compute_metrics(
                apply_iletim_calibration(val_logits, iletim_bias=iletim_bias, iletim_temperature=iletim_temperature),
                val_labels,
            )
            row = {
                "iletim_bias": iletim_bias,
                "iletim_temperature": iletim_temperature,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_iletim_f1": val_metrics["per_class_f1"]["iletim"],
                "val_iletim_precision": val_metrics["per_class_precision"]["iletim"],
                "val_iletim_recall": val_metrics["per_class_recall"]["iletim"],
                "val_iletim_fn": val_metrics["iletim_fn"],
                "val_iletim_fp": val_metrics["iletim_fp"],
                "val_iletim_total_error": val_metrics["iletim_total_error"],
                "val_ece": val_metrics["ece"],
                "passes_macro_floor": val_metrics["macro_f1"] >= macro_floor,
            }
            search_rows.append(row)

    admissible = [row for row in search_rows if row["passes_macro_floor"]]
    if admissible:
        best_row = sorted(
            admissible,
            key=lambda item: (
                item["val_iletim_f1"],
                item["val_macro_f1"],
                -item["val_iletim_total_error"],
            ),
            reverse=True,
        )[0]
    else:
        best_row = sorted(
            search_rows,
            key=lambda item: (item["val_iletim_f1"], item["val_macro_f1"]),
            reverse=True,
        )[0]

    chosen_bias = float(best_row["iletim_bias"])
    chosen_temp = float(best_row["iletim_temperature"])
    selected_validation = compute_metrics(
        apply_iletim_calibration(val_logits, iletim_bias=chosen_bias, iletim_temperature=chosen_temp),
        val_labels,
    )
    selected_test = compute_metrics(
        apply_iletim_calibration(test_logits, iletim_bias=chosen_bias, iletim_temperature=chosen_temp),
        test_labels,
    )

    top_candidates = sorted(
        admissible if admissible else search_rows,
        key=lambda item: (item["val_iletim_f1"], item["val_macro_f1"]),
        reverse=True,
    )[: args.top_k]

    write_csv(args.output_dir / "grid_search_results.csv", search_rows)

    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "search_space": {
            "bias_min": args.bias_min,
            "bias_max": args.bias_max,
            "bias_step": args.bias_step,
            "temp_min": args.temp_min,
            "temp_max": args.temp_max,
            "temp_step": args.temp_step,
            "total_candidates": len(search_rows),
        },
        "selection_policy": {
            "primary": "validation iletim F1 max",
            "secondary": "validation macro F1 max",
            "macro_floor_delta": args.macro_floor_delta,
            "validation_macro_floor": macro_floor,
        },
        "base_validation": base_validation,
        "base_test": base_test,
        "selected_params": {
            "iletim_bias": chosen_bias,
            "iletim_temperature": chosen_temp,
        },
        "selected_validation": selected_validation,
        "selected_test": selected_test,
        "top_candidates": top_candidates,
        "meaningful_gain": (
            selected_test["per_class_f1"]["iletim"] >= base_test["per_class_f1"]["iletim"] + 0.005
            and selected_test["macro_f1"] >= base_test["macro_f1"] - 0.001
        ),
        "note": "Bu deney, yalnizca iletim logitine bias ve temperature uygular. Model agirliklari degismez.",
    }
    (args.output_dir / "iletim_class_specific_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
