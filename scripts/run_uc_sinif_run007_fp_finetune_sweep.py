from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/home/alper/ekg")
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_uc_sinif_baseline.py"
VIS_SCRIPT = REPO_ROOT / "scripts" / "render_confusion_and_gradcam.py"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
BASE_CHECKPOINT = REPO_ROOT / "artifacts" / "uc_sinif_final_run007" / "best_model.pt"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "uc_sinif_run007_fp_finetune"


RUN_CONFIGS = [
    {
        "run_name": "ft01_lr1e4_e6_min1e6_none_sampler_none_weight_none_ls005",
        "epochs": 6,
        "lr": 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "augment": "none",
        "sampler": "none",
        "class_weight_mode": "none",
    },
    {
        "run_name": "ft02_lr5e5_e8_min1e6_none_sampler_none_weight_none_ls005",
        "epochs": 8,
        "lr": 5e-5,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "augment": "none",
        "sampler": "none",
        "class_weight_mode": "none",
    },
    {
        "run_name": "ft03_lr1e4_e6_min1e5_none_sampler_none_weight_none_ls010",
        "epochs": 6,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "augment": "none",
        "sampler": "none",
        "class_weight_mode": "none",
    },
    {
        "run_name": "ft04_lr5e5_e8_min1e5_none_sampler_none_weight_none_ls010",
        "epochs": 8,
        "lr": 5e-5,
        "min_lr": 1e-5,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "augment": "none",
        "sampler": "none",
        "class_weight_mode": "none",
    },
    {
        "run_name": "ft05_lr1e4_e6_min1e6_light_sampler_none_weight_none_ls005",
        "epochs": 6,
        "lr": 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "augment": "light",
        "sampler": "none",
        "class_weight_mode": "none",
    },
    {
        "run_name": "ft06_lr1e4_e6_min1e6_none_sampler_none_weight_balanced_ls005",
        "epochs": 6,
        "lr": 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "augment": "none",
        "sampler": "none",
        "class_weight_mode": "balanced",
    },
    {
        "run_name": "ft07_lr5e5_e8_min1e6_none_sampler_weighted_weight_none_ls005",
        "epochs": 8,
        "lr": 5e-5,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "augment": "none",
        "sampler": "weighted",
        "class_weight_mode": "none",
    },
    {
        "run_name": "ft08_lr1e4_e6_min1e6_none_sampler_none_weight_none_ls005_wd5e4",
        "epochs": 6,
        "lr": 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 5e-4,
        "label_smoothing": 0.05,
        "augment": "none",
        "sampler": "none",
        "class_weight_mode": "none",
    },
]


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Komut basarisiz oldu: {' '.join(command)}")


def summarize(report: dict[str, object], output_dir: Path) -> dict[str, object]:
    validation = report["validation"]
    test = report["test"]
    return {
        "run_dir": str(output_dir),
        "best_epoch": report["best_epoch"],
        "epochs": report["epochs"],
        "lr": report["lr"],
        "min_lr": report["min_lr"],
        "weight_decay": report["weight_decay"],
        "label_smoothing": report["label_smoothing"],
        "augment": report["augment"],
        "sampler": report["sampler"],
        "class_weight_mode": report["class_weight_mode"],
        "val_macro_f1": validation["macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "val_precision": validation["per_class_precision"],
        "test_precision": test["per_class_precision"],
        "val_recall": validation["per_class_recall"],
        "test_recall": test["per_class_recall"],
        "test_iletim_f1": test["per_class_f1"]["iletim"],
        "test_confusion_matrix": test["confusion_matrix"],
        "confusion_matrix_test_png": str(output_dir / "visuals" / "confusion_matrix_test.png"),
        "gradcam_summary_test_json": str(output_dir / "visuals" / "gradcam_summary_test.json"),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = {
        "base_checkpoint": str(BASE_CHECKPOINT),
        "total_runs": len(RUN_CONFIGS),
        "runs": RUN_CONFIGS,
    }
    (OUTPUT_ROOT / "fp_finetune_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    summaries: list[dict[str, object]] = []
    for config in RUN_CONFIGS:
        output_dir = OUTPUT_ROOT / config["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            str(PYTHON),
            str(TRAIN_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--output-dir",
            str(output_dir),
            "--model",
            "baseline",
            "--epochs",
            str(config["epochs"]),
            "--batch-size",
            "64",
            "--lr",
            str(config["lr"]),
            "--weight-decay",
            str(config["weight_decay"]),
            "--dropout",
            "0.2",
            "--label-smoothing",
            str(config["label_smoothing"]),
            "--scheduler",
            "cosine",
            "--min-lr",
            str(config["min_lr"]),
            "--augment",
            str(config["augment"]),
            "--sampler",
            str(config["sampler"]),
            "--class-weight-mode",
            str(config["class_weight_mode"]),
            "--resume-checkpoint",
            str(BASE_CHECKPOINT),
        ]
        run_command(train_cmd)

        visuals_dir = output_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        vis_cmd = [
            str(PYTHON),
            str(VIS_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--checkpoint",
            str(output_dir / "best_model.pt"),
            "--report-json",
            str(output_dir / "training_report.json"),
            "--output-dir",
            str(visuals_dir),
            "--split",
            "test",
        ]
        run_command(vis_cmd)

        report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))
        summary = summarize(report, output_dir)
        summaries.append(summary)
        (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final = {
        "base_checkpoint": str(BASE_CHECKPOINT),
        "total_runs": len(summaries),
        "best_by_test_macro_f1": sorted(summaries, key=lambda item: item["test_macro_f1"], reverse=True),
        "best_by_test_iletim_f1": sorted(summaries, key=lambda item: item["test_iletim_f1"], reverse=True),
        "all_runs": summaries,
    }
    (OUTPUT_ROOT / "fp_finetune_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    json.dump(final, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
