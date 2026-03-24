from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from train_uc_sinif_baseline import BaselineECGNet, CLASS_NAMES, ResNetSEECGNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tek EKG kaydi icin 3 sinif tahmini uretir.")
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/alper/ekg/secili_model_adaylari/rba02_final_aday/tuning/ft06_lr5e5_e8_fn150_fr200_light_wd5e5/best_model.pt"),
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=Path("/home/alper/ekg/secili_model_adaylari/rba02_final_aday/calibration_ft06/temperature_calibration.json"),
    )
    return parser.parse_args()


def build_model(checkpoint: dict[str, object], device: torch.device) -> torch.nn.Module:
    args = checkpoint.get("args", {})
    model_name = str(args.get("model", "baseline"))
    dropout = float(args.get("dropout", 0.2))
    if model_name == "resnet_se":
        model = ResNetSEECGNet(num_classes=len(CLASS_NAMES), dropout=dropout)
    else:
        model = BaselineECGNet(num_classes=len(CLASS_NAMES), dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_signal(mat_path: Path) -> torch.Tensor:
    signal = loadmat(mat_path)["val"].astype(np.float32)
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    signal = (signal - mean) / std
    return torch.from_numpy(signal).unsqueeze(0)


def load_temperature(path: Path) -> float:
    if not path.exists():
        return 1.0
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj.get("optimized_temperature", 1.0))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint, device)
    temperature = load_temperature(args.calibration_json)

    signal = load_signal(args.mat_path).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(signal) / temperature
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    result = {
        "mat_path": str(args.mat_path),
        "checkpoint": str(args.checkpoint),
        "temperature": temperature,
        "predicted_class": CLASS_NAMES[pred_idx],
        "probabilities": {name: float(probs[idx]) for idx, name in enumerate(CLASS_NAMES)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
