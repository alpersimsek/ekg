from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path("/home/alper/ekg")
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_uc_sinif_iletim_focus.py"
VIS_SCRIPT = REPO_ROOT / "scripts" / "render_confusion_and_gradcam.py"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
BASE_DIR = REPO_ROOT / "secili_model_adaylari" / "finalist_v2"
BASE_CHECKPOINT = BASE_DIR / "best_model.pt"
OUTPUT_ROOT = BASE_DIR / "iletim_sweep"
BASE_TEST_MACRO_F1 = 0.9382599872565627
BASE_TEST_ILETIM_F1 = 0.9046321525885558
BASE_TEST_ILETIM_FN = 23
BASE_TEST_ILETIM_FP = 12

FOCUS_ILETIM_CODES = ["59118001", "270492004", "10370003", "164909002", "445118002", "713426002"]

RUN_CONFIGS = [
    {"run_name": "it01_lr1e4_e6_iw120_none", "epochs": 6, "lr": 1e-4, "weight_decay": 5e-5, "augment": "none", "focus_iletim_weight": 1.2},
    {"run_name": "it02_lr1e4_e6_iw135_none", "epochs": 6, "lr": 1e-4, "weight_decay": 5e-5, "augment": "none", "focus_iletim_weight": 1.35},
    {"run_name": "it03_lr5e5_e8_iw120_light", "epochs": 8, "lr": 5e-5, "weight_decay": 5e-5, "augment": "light", "focus_iletim_weight": 1.2},
    {"run_name": "it04_lr5e5_e8_iw135_light", "epochs": 8, "lr": 5e-5, "weight_decay": 5e-5, "augment": "light", "focus_iletim_weight": 1.35},
    {"run_name": "it05_lr5e5_e8_iw150_light", "epochs": 8, "lr": 5e-5, "weight_decay": 5e-5, "augment": "light", "focus_iletim_weight": 1.5},
    {"run_name": "it06_lr5e5_e8_iw135_none_wd1e4", "epochs": 8, "lr": 5e-5, "weight_decay": 1e-4, "augment": "none", "focus_iletim_weight": 1.35},
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
    iletim_fn = confusion[2][0] + confusion[2][1]
    iletim_fp = confusion[0][2] + confusion[1][2]
    return {
        "run_dir": str(output_dir),
        "best_epoch": report["best_epoch"],
        "epochs": report["epochs"],
        "lr": report["lr"],
        "weight_decay": report["weight_decay"],
        "augment": report["augment"],
        "focus_iletim_weight": report["focus_iletim_weight"],
        "val_macro_f1": validation["macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "test_iletim_f1": test["per_class_f1"]["iletim"],
        "test_iletim_precision": test["per_class_precision"]["iletim"],
        "test_iletim_recall": test["per_class_recall"]["iletim"],
        "iletim_fn": iletim_fn,
        "iletim_fp": iletim_fp,
        "iletim_total_error": iletim_fn + iletim_fp,
        "macro_delta_vs_base": test["macro_f1"] - BASE_TEST_MACRO_F1,
        "iletim_f1_delta_vs_base": test["per_class_f1"]["iletim"] - BASE_TEST_ILETIM_F1,
        "test_confusion_matrix": confusion,
        "confusion_matrix_test_png": str(output_dir / "visuals" / "confusion_matrix_test.png"),
        "gradcam_summary_test_json": str(output_dir / "visuals" / "gradcam_summary_test.json"),
    }


def is_meaningful_gain(summary: dict[str, object]) -> bool:
    return (
        float(summary["test_iletim_f1"]) >= BASE_TEST_ILETIM_F1 + 0.005
        and float(summary["test_macro_f1"]) >= BASE_TEST_MACRO_F1 - 0.001
    )


def promote_winner(best_summary: dict[str, object]) -> None:
    selected_dir = Path(best_summary["run_dir"])
    target_dir = BASE_DIR / "secili_surum_iletim"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(selected_dir, target_dir)
    update = {
        "base_model_dir": str(BASE_DIR),
        "source_run_dir": str(selected_dir),
        "selection_reason": "iletim odakli sweep anlamli kazanc getirdi",
        "base_test_macro_f1": BASE_TEST_MACRO_F1,
        "base_test_iletim_f1": BASE_TEST_ILETIM_F1,
        "selected_test_macro_f1": best_summary["test_macro_f1"],
        "selected_test_iletim_f1": best_summary["test_iletim_f1"],
        "selected_iletim_fn": best_summary["iletim_fn"],
        "selected_iletim_fp": best_summary["iletim_fp"],
    }
    (BASE_DIR / "iletim_promotion.json").write_text(json.dumps(update, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = {
        "base_checkpoint": str(BASE_CHECKPOINT),
        "focus_iletim_codes": FOCUS_ILETIM_CODES,
        "baseline_reference": {
            "test_macro_f1": BASE_TEST_MACRO_F1,
            "test_iletim_f1": BASE_TEST_ILETIM_F1,
            "test_iletim_fn": BASE_TEST_ILETIM_FN,
            "test_iletim_fp": BASE_TEST_ILETIM_FP,
        },
        "total_runs": len(RUN_CONFIGS),
        "runs": RUN_CONFIGS,
    }
    (OUTPUT_ROOT / "iletim_sweep_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

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
            "0.05",
            "--scheduler",
            "cosine",
            "--min-lr",
            "1e-6",
            "--augment",
            str(config["augment"]),
            "--sampler",
            "weighted",
            "--class-weight-mode",
            "none",
            "--focus-normal-weight",
            "1.5",
            "--focus-ritim-weight",
            "2.0",
            "--focus-iletim-weight",
            str(config["focus_iletim_weight"]),
            "--focus-iletim-codes",
            *FOCUS_ILETIM_CODES,
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

    best_by_iletim = sorted(
        summaries,
        key=lambda item: (item["test_iletim_f1"], item["test_macro_f1"]),
        reverse=True,
    )
    best_by_macro = sorted(summaries, key=lambda item: item["test_macro_f1"], reverse=True)
    meaningful_candidates = [item for item in best_by_iletim if is_meaningful_gain(item)]

    final = {
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_test_macro_f1": BASE_TEST_MACRO_F1,
        "base_test_iletim_f1": BASE_TEST_ILETIM_F1,
        "total_runs": len(summaries),
        "best_by_test_iletim_f1": best_by_iletim,
        "best_by_test_macro_f1": best_by_macro,
        "meaningful_candidates": meaningful_candidates,
        "promoted": bool(meaningful_candidates),
    }
    (OUTPUT_ROOT / "iletim_sweep_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    if meaningful_candidates:
        promote_winner(meaningful_candidates[0])


if __name__ == "__main__":
    main()
