from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

from train_uc_sinif_baseline import (
    BaselineECGNet,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    ResNetSEECGNet,
)


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confusion matrix ve Grad-CAM ciktisi uretir.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dogrulama", "test"], default="test")
    return parser.parse_args()


def load_model(report: dict[str, object], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model_name = report.get("model", "baseline")
    dropout = float(report.get("dropout", 0.2))
    if model_name == "resnet_se":
        model = ResNetSEECGNet(num_classes=len(CLASS_NAMES), dropout=dropout)
        target_layer = model.layer3[-1]
    else:
        model = BaselineECGNet(num_classes=len(CLASS_NAMES), dropout=dropout)
        target_layer = model.layer2[-1]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model._gradcam_target_layer = target_layer  # type: ignore[attr-defined]
    return model


def load_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and row["target_class"] in CLASS_TO_INDEX:
                rows.append(row)
    return rows


def preprocess_signal(mat_path: Path) -> np.ndarray:
    signal = loadmat(mat_path)["val"].astype(np.float32)
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return (signal - mean) / std


def evaluate_rows(model: torch.nn.Module, rows: list[dict[str, str]], device: torch.device) -> tuple[np.ndarray, list[dict[str, object]]]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    evaluated: list[dict[str, object]] = []
    with torch.no_grad():
        for row in rows:
            signal = preprocess_signal(Path(row["mat_yolu"]))
            tensor = torch.from_numpy(signal).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            truth_idx = CLASS_TO_INDEX[row["target_class"]]
            confusion[truth_idx, pred_idx] += 1
            evaluated.append(
                {
                    "row": row,
                    "truth_idx": truth_idx,
                    "pred_idx": pred_idx,
                    "probs": probs.tolist(),
                }
            )
    return confusion, evaluated


def save_confusion_matrix(confusion: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title(title)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def choose_gradcam_examples(evaluated: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    # One confident TP per class.
    for class_name in CLASS_NAMES:
        idx = CLASS_TO_INDEX[class_name]
        candidates = [
            item for item in evaluated if item["truth_idx"] == idx and item["pred_idx"] == idx
        ]
        candidates.sort(key=lambda item: item["probs"][idx], reverse=True)
        if candidates:
            chosen.append(candidates[0])
    # One common FP/ambiguous example if present.
    extra_pairs = [
        ("ritim", "normal"),
        ("ritim", "iletim"),
        ("normal", "iletim"),
    ]
    for truth_name, pred_name in extra_pairs:
        truth_idx = CLASS_TO_INDEX[truth_name]
        pred_idx = CLASS_TO_INDEX[pred_name]
        candidates = [
            item for item in evaluated if item["truth_idx"] == truth_idx and item["pred_idx"] == pred_idx
        ]
        candidates.sort(key=lambda item: item["probs"][pred_idx], reverse=True)
        if candidates:
            chosen.append(candidates[0])
    return chosen[:6]


def compute_gradcam(model: torch.nn.Module, signal: np.ndarray, target_class: int, device: torch.device) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    layer = model._gradcam_target_layer  # type: ignore[attr-defined]
    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    try:
        tensor = torch.from_numpy(signal).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        logits = model(tensor)
        score = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        act = activations[-1][0]  # [C, L]
        grad = gradients[-1][0]   # [C, L]
        weights = grad.mean(dim=1, keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0)).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = np.interp(np.linspace(0, len(cam) - 1, signal.shape[1]), np.arange(len(cam)), cam)
        heatmap = np.tile(cam[None, :], (signal.shape[0], 1))
        return heatmap
    finally:
        handle_f.remove()
        handle_b.remove()


def save_gradcam(signal: np.ndarray, heatmap: np.ndarray, record_id: str, truth: str, pred: str, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    x = np.arange(signal.shape[1])
    for lead_idx, ax in enumerate(axes):
        ax.plot(x, signal[lead_idx], color="black", linewidth=0.7)
        ax.imshow(
            heatmap[lead_idx][None, :],
            extent=[0, signal.shape[1], signal[lead_idx].min(), signal[lead_idx].max()],
            aspect="auto",
            cmap="magma",
            alpha=0.45,
        )
        ax.set_title(LEAD_NAMES[lead_idx])
    fig.suptitle(f"{record_id} | gercek={truth} | tahmin={pred}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(report, args.checkpoint, device)
    rows = load_rows(args.manifest, args.split)
    confusion, evaluated = evaluate_rows(model, rows, device)
    save_confusion_matrix(
        confusion,
        args.output_dir / f"confusion_matrix_{args.split}.png",
        title=f"{args.split} confusion matrix",
    )

    chosen = choose_gradcam_examples(evaluated)
    summary: list[dict[str, object]] = []
    for item in chosen:
        row = item["row"]
        signal = preprocess_signal(Path(row["mat_yolu"]))
        pred_idx = int(item["pred_idx"])
        truth_idx = int(item["truth_idx"])
        heatmap = compute_gradcam(model, signal, pred_idx, device)
        output_path = args.output_dir / f"gradcam_{row['kayit_id']}_{CLASS_NAMES[truth_idx]}_to_{CLASS_NAMES[pred_idx]}.png"
        save_gradcam(signal, heatmap, row["kayit_id"], CLASS_NAMES[truth_idx], CLASS_NAMES[pred_idx], output_path)
        summary.append(
            {
                "kayit_id": row["kayit_id"],
                "gercek": CLASS_NAMES[truth_idx],
                "tahmin": CLASS_NAMES[pred_idx],
                "tani_kodlari": row["tani_kodlari"],
                "dosya": str(output_path),
            }
        )
    (args.output_dir / f"gradcam_summary_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"confusion_matrix": confusion.tolist(), "gradcam_examples": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
