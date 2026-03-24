from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path("/home/alper/ekg")
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_uc_sinif_bucket_aware.py"
VIS_SCRIPT = REPO_ROOT / "scripts" / "render_confusion_and_gradcam.py"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "uc_sinif_bucket_aware_refined"


RUN_CONFIGS = [
    {
        "run_name": "rba01_fn15_fr15_light",
        "focus_normal_weight": 1.5,
        "focus_ritim_weight": 1.5,
        "augment": "light",
        "weight_decay": 1e-4,
    },
    {
        "run_name": "rba02_fn15_fr20_light",
        "focus_normal_weight": 1.5,
        "focus_ritim_weight": 2.0,
        "augment": "light",
        "weight_decay": 1e-4,
    },
    {
        "run_name": "rba03_fn20_fr15_light",
        "focus_normal_weight": 2.0,
        "focus_ritim_weight": 1.5,
        "augment": "light",
        "weight_decay": 1e-4,
    },
    {
        "run_name": "rba04_fn175_fr175_light",
        "focus_normal_weight": 1.75,
        "focus_ritim_weight": 1.75,
        "augment": "light",
        "weight_decay": 1e-4,
    },
    {
        "run_name": "rba05_fn15_fr15_none",
        "focus_normal_weight": 1.5,
        "focus_ritim_weight": 1.5,
        "augment": "none",
        "weight_decay": 1e-4,
    },
    {
        "run_name": "rba06_fn15_fr20_light_wd5e5",
        "focus_normal_weight": 1.5,
        "focus_ritim_weight": 2.0,
        "augment": "light",
        "weight_decay": 5e-5,
    },
    {
        "run_name": "rba07_fn175_fr15_light",
        "focus_normal_weight": 1.75,
        "focus_ritim_weight": 1.5,
        "augment": "light",
        "weight_decay": 1e-4,
    },
    {
        "run_name": "rba08_fn15_fr175_light",
        "focus_normal_weight": 1.5,
        "focus_ritim_weight": 1.75,
        "augment": "light",
        "weight_decay": 1e-4,
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
    confusion = test["confusion_matrix"]
    return {
        "run_dir": str(output_dir),
        "best_epoch": report["best_epoch"],
        "focus_normal_weight": report["focus_normal_weight"],
        "focus_ritim_weight": report["focus_ritim_weight"],
        "sampler_stats": report["sampler_stats"],
        "val_macro_f1": validation["macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "test_iletim_f1": test["per_class_f1"]["iletim"],
        "normal_to_ritim": confusion[0][1],
        "ritim_to_normal": confusion[1][0],
        "priority_fp_sum": confusion[0][1] + confusion[1][0],
        "test_confusion_matrix": confusion,
        "confusion_matrix_test_png": str(output_dir / "visuals" / "confusion_matrix_test.png"),
        "gradcam_summary_test_json": str(output_dir / "visuals" / "gradcam_summary_test.json"),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = {"total_runs": len(RUN_CONFIGS), "runs": RUN_CONFIGS}
    (OUTPUT_ROOT / "bucket_aware_refined_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
            "50",
            "--batch-size",
            "64",
            "--lr",
            "0.0005",
            "--weight-decay",
            str(config["weight_decay"]),
            "--dropout",
            "0.2",
            "--label-smoothing",
            "0.05",
            "--scheduler",
            "cosine",
            "--min-lr",
            "1e-5",
            "--augment",
            str(config["augment"]),
            "--sampler",
            "weighted",
            "--class-weight-mode",
            "none",
            "--focus-normal-weight",
            str(config["focus_normal_weight"]),
            "--focus-ritim-weight",
            str(config["focus_ritim_weight"]),
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
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    final = {
        "total_runs": len(summaries),
        "best_by_test_macro_f1": sorted(summaries, key=lambda item: item["test_macro_f1"], reverse=True),
        "best_by_test_iletim_f1": sorted(summaries, key=lambda item: item["test_iletim_f1"], reverse=True),
        "best_by_priority_fp_sum": sorted(summaries, key=lambda item: item["priority_fp_sum"]),
    }
    (OUTPUT_ROOT / "bucket_aware_refined_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
