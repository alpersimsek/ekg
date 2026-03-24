from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from render_confusion_and_gradcam import (
    compute_gradcam,
    preprocess_signal,
    save_confusion_matrix,
    save_gradcam,
)
from run_finalist_v2_iletim_two_stage import load_binary_model, load_three_class_model
from train_uc_sinif_baseline import CLASS_NAMES, CLASS_TO_INDEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iki asamali finalist icin confusion matrix ve Grad-CAM uretir.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--aux-checkpoint", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=float, required=True)
    parser.add_argument("--negative-threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dogrulama", "test"], default="test")
    return parser.parse_args()


def load_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and row["target_class"] in CLASS_TO_INDEX:
                rows.append(row)
    return rows


def evaluate_two_stage(
    base_model: torch.nn.Module,
    aux_model: torch.nn.Module,
    rows: list[dict[str, str]],
    device: torch.device,
    positive_threshold: float,
    negative_threshold: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    evaluated: list[dict[str, object]] = []
    with torch.no_grad():
        for row in rows:
            signal = preprocess_signal(Path(row["mat_yolu"]))
            tensor = torch.from_numpy(signal).unsqueeze(0).to(device)
            base_logits = base_model(tensor)
            base_probs = torch.softmax(base_logits, dim=1).cpu().numpy()[0]
            aux_probs = torch.softmax(aux_model(tensor), dim=1).cpu().numpy()[0]
            base_pred_idx = int(np.argmax(base_probs))
            final_pred_idx = base_pred_idx
            override_type = "none"
            if aux_probs[1] >= positive_threshold:
                final_pred_idx = CLASS_TO_INDEX["iletim"]
                if final_pred_idx != base_pred_idx:
                    override_type = "force_iletim"
            elif base_pred_idx == CLASS_TO_INDEX["iletim"] and aux_probs[1] < negative_threshold:
                masked = base_logits.cpu().numpy()[0].copy()
                masked[CLASS_TO_INDEX["iletim"]] = -1e9
                final_pred_idx = int(np.argmax(masked))
                if final_pred_idx != base_pred_idx:
                    override_type = "revert_from_iletim"
            truth_idx = CLASS_TO_INDEX[row["target_class"]]
            confusion[truth_idx, final_pred_idx] += 1
            evaluated.append(
                {
                    "row": row,
                    "truth_idx": truth_idx,
                    "base_pred_idx": base_pred_idx,
                    "final_pred_idx": final_pred_idx,
                    "base_probs": base_probs.tolist(),
                    "aux_probs": aux_probs.tolist(),
                    "override_type": override_type,
                }
            )
    return confusion, evaluated


def choose_examples(evaluated: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        idx = CLASS_TO_INDEX[class_name]
        candidates = [item for item in evaluated if item["truth_idx"] == idx and item["final_pred_idx"] == idx]
        candidates.sort(key=lambda item: item["base_probs"][idx] if idx < 3 else 0.0, reverse=True)
        if candidates:
            chosen.append(candidates[0])
    force_candidates = [item for item in evaluated if item["override_type"] == "force_iletim"]
    force_candidates.sort(key=lambda item: item["aux_probs"][1], reverse=True)
    if force_candidates:
        chosen.append(force_candidates[0])
    revert_candidates = [item for item in evaluated if item["override_type"] == "revert_from_iletim"]
    revert_candidates.sort(key=lambda item: item["aux_probs"][1])
    if revert_candidates:
        chosen.append(revert_candidates[0])
    error_candidates = [
        item for item in evaluated if item["truth_idx"] != item["final_pred_idx"] and item["override_type"] == "none"
    ]
    error_candidates.sort(key=lambda item: item["base_probs"][item["final_pred_idx"]], reverse=True)
    if error_candidates:
        chosen.append(error_candidates[0])
    uniq = []
    seen = set()
    for item in chosen:
        key = item["row"]["kayit_id"]
        if key not in seen:
            uniq.append(item)
            seen.add(key)
    return uniq[:6]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_three_class_model(args.base_checkpoint, device)
    aux_model = load_binary_model(args.aux_checkpoint, device)
    base_model._gradcam_target_layer = base_model.layer2[-1]  # type: ignore[attr-defined]
    aux_model._gradcam_target_layer = aux_model.layer2[-1]  # type: ignore[attr-defined]
    rows = load_rows(args.manifest, args.split)
    confusion, evaluated = evaluate_two_stage(
        base_model,
        aux_model,
        rows,
        device,
        args.positive_threshold,
        args.negative_threshold,
    )
    save_confusion_matrix(confusion, args.output_dir / f"confusion_matrix_{args.split}.png", f"{args.split} confusion matrix")
    chosen = choose_examples(evaluated)

    summary: list[dict[str, object]] = []
    for item in chosen:
        row = item["row"]
        signal = preprocess_signal(Path(row["mat_yolu"]))
        base_pred_idx = int(item["base_pred_idx"])
        final_pred_idx = int(item["final_pred_idx"])
        base_heatmap = compute_gradcam(base_model, signal, base_pred_idx, device)
        aux_heatmap = compute_gradcam(aux_model, signal, 1, device)

        base_path = args.output_dir / (
            f"base_gradcam_{row['kayit_id']}_{CLASS_NAMES[item['truth_idx']]}_to_{CLASS_NAMES[base_pred_idx]}.png"
        )
        aux_path = args.output_dir / (
            f"aux_gradcam_{row['kayit_id']}_{CLASS_NAMES[item['truth_idx']]}_to_{CLASS_NAMES[final_pred_idx]}.png"
        )
        save_gradcam(signal, base_heatmap, row["kayit_id"], CLASS_NAMES[item["truth_idx"]], CLASS_NAMES[base_pred_idx], base_path)
        save_gradcam(signal, aux_heatmap, row["kayit_id"], CLASS_NAMES[item["truth_idx"]], "aux_iletim", aux_path)
        summary.append(
            {
                "kayit_id": row["kayit_id"],
                "gercek": CLASS_NAMES[item["truth_idx"]],
                "base_tahmin": CLASS_NAMES[base_pred_idx],
                "final_tahmin": CLASS_NAMES[final_pred_idx],
                "override_type": item["override_type"],
                "tani_kodlari": row["tani_kodlari"],
                "base_gradcam": str(base_path),
                "aux_gradcam": str(aux_path),
                "aux_iletim_olasilik": item["aux_probs"][1],
            }
        )
    (args.output_dir / f"gradcam_summary_{args.split}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"confusion_matrix": confusion.tolist(), "gradcam_examples": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
