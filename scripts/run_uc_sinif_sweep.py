from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/home/alper/ekg")
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_uc_sinif_baseline.py"
VIS_SCRIPT = REPO_ROOT / "scripts" / "render_confusion_and_gradcam.py"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "uc_sinif_sweep50"


PARAMETER_SPACE = {
    "model": ["baseline", "resnet_se"],
    "epochs": [50],
    "lr": [1e-3, 5e-4, 3e-4, 1e-4],
    "dropout": [0.2, 0.3, 0.4],
    "label_smoothing": [0.0, 0.05, 0.1],
    "scheduler": ["cosine"],
    "min_lr": [1e-6, 1e-5],
    "augment": ["none", "light"],
    "sampler": ["none", "weighted"],
    "class_weight_mode": ["balanced", "none"],
}

PARAMETER_KEYS = list(PARAMETER_SPACE.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3 sinifli EKG icin otomatik hyperparameter sweep.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Sweep sonuc klasoru.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=24,
        help="Teorik grid icinden secilecek maksimum kosu sayisi.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["even", "random"],
        default="even",
        help="Grid icinden kosu secme yontemi.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgele secim gerekiyorsa kullanilacak seed.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Uretilen kosu index baslangici.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Egitim baslatmadan yalniz secilen kosulari yaz.",
    )
    return parser.parse_args()


def format_float(value: float) -> str:
    text = f"{value:.10g}"
    return text.replace(".", "").replace("-", "m")


def slugify_value(key: str, value: object) -> str:
    if isinstance(value, float):
        if key == "lr":
            return f"lr{format_float(value)}"
        if key == "dropout":
            return f"do{format_float(value)}"
        if key == "label_smoothing":
            return f"ls{format_float(value)}"
        if key == "min_lr":
            return f"min{format_float(value)}"
        return f"{key}{format_float(value)}"
    return f"{key}{str(value).replace('_', '')}"


def build_run_name(index: int, config: dict[str, object]) -> str:
    parts = [f"run{index:03d}"]
    parts.append(str(config["model"]).replace("_", ""))
    parts.append(slugify_value("lr", config["lr"]))
    parts.append(slugify_value("dropout", config["dropout"]))
    parts.append(slugify_value("label_smoothing", config["label_smoothing"]))
    parts.append(str(config["scheduler"]))
    parts.append(slugify_value("min_lr", config["min_lr"]))
    parts.append(f"aug{config['augment']}")
    parts.append(f"sampler{config['sampler']}")
    parts.append(f"weight{config['class_weight_mode']}")
    return "_".join(parts)


def generate_all_configs() -> list[dict[str, object]]:
    values = [PARAMETER_SPACE[key] for key in PARAMETER_KEYS]
    all_configs: list[dict[str, object]] = []
    for combo in itertools.product(*values):
        config = {key: value for key, value in zip(PARAMETER_KEYS, combo, strict=True)}
        all_configs.append(config)
    return all_configs


def select_configs(
    all_configs: list[dict[str, object]],
    max_runs: int,
    selection_mode: str,
    seed: int,
) -> list[dict[str, object]]:
    if max_runs <= 0:
        raise ValueError("--max-runs pozitif olmali.")
    if max_runs >= len(all_configs):
        return list(all_configs)

    if selection_mode == "random":
        rng = random.Random(seed)
        indices = list(range(len(all_configs)))
        rng.shuffle(indices)
        chosen_indices = sorted(indices[:max_runs])
        return [all_configs[index] for index in chosen_indices]

    if max_runs == 1:
        return [all_configs[0]]

    step = (len(all_configs) - 1) / (max_runs - 1)
    chosen_indices: list[int] = []
    used: set[int] = set()
    for i in range(max_runs):
        candidate = int(round(i * step))
        while candidate in used and candidate + 1 < len(all_configs):
            candidate += 1
        if candidate in used:
            candidate = max(idx for idx in range(len(all_configs)) if idx not in used)
        used.add(candidate)
        chosen_indices.append(candidate)
    return [all_configs[index] for index in chosen_indices]


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Komut basarisiz oldu: {' '.join(command)}")


def summarize_run(report: dict[str, object], output_dir: Path) -> dict[str, object]:
    test = report["test"]
    validation = report["validation"]
    return {
        "run_dir": str(output_dir),
        "model": report["model"],
        "epochs": report["epochs"],
        "best_epoch": report["best_epoch"],
        "lr": report["lr"],
        "dropout": report["dropout"],
        "label_smoothing": report["label_smoothing"],
        "scheduler": report["scheduler"],
        "min_lr": report["min_lr"],
        "augment": report["augment"],
        "sampler": report["sampler"],
        "class_weight_mode": report["class_weight_mode"],
        "val_macro_f1": validation["macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "val_accuracy": validation["accuracy"],
        "test_accuracy": test["accuracy"],
        "val_iletim_f1": validation["per_class_f1"]["iletim"],
        "test_iletim_f1": test["per_class_f1"]["iletim"],
        "confusion_matrix_test_png": str(output_dir / "visuals" / "confusion_matrix_test.png"),
        "gradcam_summary_test_json": str(output_dir / "visuals" / "gradcam_summary_test.json"),
    }


def main() -> None:
    args = parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(f"Python ortami bulunamadi: {PYTHON}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_configs = generate_all_configs()
    selected_configs = select_configs(all_configs, args.max_runs, args.selection_mode, args.seed)

    generated_runs: list[dict[str, object]] = []
    for offset, config in enumerate(selected_configs, start=args.start_index):
        run_config = dict(config)
        run_config["run_name"] = build_run_name(offset, run_config)
        generated_runs.append(run_config)

    index = {
        "parameter_space": PARAMETER_SPACE,
        "parameter_order": PARAMETER_KEYS,
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "requested_max_runs": args.max_runs,
        "total_possible_runs": len(all_configs),
        "total_runs": len(generated_runs),
        "runs": generated_runs,
    }
    (args.output_root / "sweep_parameter_space.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        json.dump(index, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    run_summaries: list[dict[str, object]] = []
    for run_config in generated_runs:
        output_dir = args.output_root / str(run_config["run_name"])
        output_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            str(PYTHON),
            str(TRAIN_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--output-dir",
            str(output_dir),
            "--model",
            str(run_config["model"]),
            "--epochs",
            str(run_config["epochs"]),
            "--lr",
            str(run_config["lr"]),
            "--dropout",
            str(run_config["dropout"]),
            "--label-smoothing",
            str(run_config["label_smoothing"]),
            "--scheduler",
            str(run_config["scheduler"]),
            "--min-lr",
            str(run_config["min_lr"]),
            "--augment",
            str(run_config["augment"]),
            "--sampler",
            str(run_config["sampler"]),
            "--class-weight-mode",
            str(run_config["class_weight_mode"]),
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
        summary = summarize_run(report, output_dir)
        run_summaries.append(summary)
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    best_by_val = sorted(run_summaries, key=lambda item: item["val_macro_f1"], reverse=True)
    best_by_test = sorted(run_summaries, key=lambda item: item["test_macro_f1"], reverse=True)
    best_by_iletim = sorted(run_summaries, key=lambda item: item["test_iletim_f1"], reverse=True)
    final_summary = {
        "parameter_space": PARAMETER_SPACE,
        "parameter_order": PARAMETER_KEYS,
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "requested_max_runs": args.max_runs,
        "total_possible_runs": len(all_configs),
        "total_runs": len(run_summaries),
        "best_by_validation_macro_f1": best_by_val[:5],
        "best_by_test_macro_f1": best_by_test[:5],
        "best_by_test_iletim_f1": best_by_iletim[:5],
        "all_runs": run_summaries,
    }
    (args.output_root / "sweep_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    json.dump(final_summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
