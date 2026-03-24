from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from run_finalist_v2_iletim_two_stage import load_binary_model, load_three_class_model
from train_uc_sinif_baseline import CLASS_NAMES, CLASS_TO_INDEX


REPO_ROOT = Path("/home/alper/ekg")
FINAL_MODEL_ROOT = REPO_ROOT / "final_model_v3"
BASE_CHECKPOINT = FINAL_MODEL_ROOT / "base_best_model.pt"
AUX_CHECKPOINT = FINAL_MODEL_ROOT / "aux_binary" / "best_model.pt"
REPORT_JSON = FINAL_MODEL_ROOT / "v3_two_stage_report.json"


@dataclass
class FinalModelV3Predictor:
    device: torch.device
    base_model: torch.nn.Module
    aux_model: torch.nn.Module
    positive_threshold: float
    negative_threshold: float

    @classmethod
    def load(
        cls,
        base_checkpoint: Path = BASE_CHECKPOINT,
        aux_checkpoint: Path = AUX_CHECKPOINT,
        report_json: Path = REPORT_JSON,
    ) -> "FinalModelV3Predictor":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = load_three_class_model(base_checkpoint, device)
        aux_model = load_binary_model(aux_checkpoint, device)
        obj = json.loads(report_json.read_text(encoding="utf-8"))
        positive_threshold = float(obj["selected_thresholds"]["positive_threshold"])
        negative_threshold = float(obj["selected_thresholds"]["negative_threshold"])
        return cls(device, base_model, aux_model, positive_threshold, negative_threshold)

    def predict_tensor(self, signal: torch.Tensor) -> dict[str, object]:
        tensor = signal.to(self.device, non_blocking=True)
        with torch.no_grad():
            base_logits = self.base_model(tensor)
            base_probs = torch.softmax(base_logits, dim=1).cpu().numpy()[0]
            aux_probs = torch.softmax(self.aux_model(tensor), dim=1).cpu().numpy()[0]
        iletim_idx = CLASS_TO_INDEX["iletim"]
        base_pred_idx = int(np.argmax(base_probs))
        final_pred_idx = base_pred_idx
        override_type = "none"
        if aux_probs[1] >= self.positive_threshold:
            final_pred_idx = iletim_idx
            if final_pred_idx != base_pred_idx:
                override_type = "force_iletim"
        elif base_pred_idx == iletim_idx and aux_probs[1] < self.negative_threshold:
            masked = base_logits.cpu().numpy()[0].copy()
            masked[iletim_idx] = -1e9
            final_pred_idx = int(np.argmax(masked))
            if final_pred_idx != base_pred_idx:
                override_type = "revert_from_iletim"
        return {
            "base_predicted_class": CLASS_NAMES[base_pred_idx],
            "predicted_class": CLASS_NAMES[final_pred_idx],
            "override_type": override_type,
            "base_probabilities": {name: float(base_probs[idx]) for idx, name in enumerate(CLASS_NAMES)},
            "aux_probabilities": {"diger": float(aux_probs[0]), "iletim": float(aux_probs[1])},
            "thresholds": {
                "positive_threshold": self.positive_threshold,
                "negative_threshold": self.negative_threshold,
            },
        }


def load_signal(mat_path: Path) -> torch.Tensor:
    signal = loadmat(mat_path)["val"].astype(np.float32)
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    signal = (signal - mean) / std
    return torch.from_numpy(signal).unsqueeze(0)


def load_rows(path: Path, split: str) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split != "all" and row["split"] != split:
                continue
            rows.append(row)
    return rows
