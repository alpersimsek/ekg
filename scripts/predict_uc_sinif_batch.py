from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from train_uc_sinif_baseline import BaselineECGNet, CLASS_NAMES, ResNetSEECGNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manifest veya csv liste uzerinden toplu 3 sinif tahmini uretir.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"),
    )
    parser.add_argument("--split", choices=["egitim", "dogrulama", "test", "all"], default="test")
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
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        signal = loadmat(row["mat_yolu"])["val"].astype(np.float32)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        signal = (signal - mean) / std
        return torch.from_numpy(signal), index


def load_rows(path: Path, split: str) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split != "all" and row["split"] != split:
                continue
            rows.append(row)
    return rows


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


def load_temperature(path: Path) -> float:
    if not path.exists():
        return 1.0
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj.get("optimized_temperature", 1.0))


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint, device)
    temperature = load_temperature(args.calibration_json)

    loader = DataLoader(
        ManifestDataset(rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    outputs: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for signals, indices in loader:
            logits = model(signals.to(device, non_blocking=True)) / temperature
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for prob_vec, idx in zip(probs, indices.numpy(), strict=False):
                row = rows[int(idx)]
                pred_idx = int(np.argmax(prob_vec))
                outputs.append(
                    {
                        "kayit_id": row["kayit_id"],
                        "mat_yolu": row["mat_yolu"],
                        "gercek_sinif": row.get("target_class", ""),
                        "tahmin_sinif": CLASS_NAMES[pred_idx],
                        "normal_olasilik": float(prob_vec[0]),
                        "ritim_olasilik": float(prob_vec[1]),
                        "iletim_olasilik": float(prob_vec[2]),
                    }
                )

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0].keys()))
        writer.writeheader()
        writer.writerows(outputs)
    print(json.dumps({"output_csv": str(args.output_csv), "rows": len(outputs), "temperature": temperature}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
