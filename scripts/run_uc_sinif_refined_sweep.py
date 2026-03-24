from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/home/alper/ekg")
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_uc_sinif_baseline.py"
VIS_SCRIPT = REPO_ROOT / "scripts" / "render_confusion_and_gradcam.py"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "uc_sinif_refined_sweep"


REFINED_SPACE = {
    "model": ["baseline"],
    "epochs": [50],
    "batch_size": [64],
    "lr": [1e-3, 5e-4, 3e-4],
    "weight_decay": [1e-4, 5e-4],
    "dropout": [0.2, 0.3],
    "label_smoothing": [0.0, 0.05, 0.1],
    "scheduler": ["cosine"],
    "min_lr": [1e-6, 1e-5],
    "augment": ["none", "light"],
    "sampler": ["none", "weighted"],
    "class_weight_mode": ["none", "balanced"],
}

PARAMETER_KEYS = list(REFINED_SPACE.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ilk 28 kosudan cikan bolgeyi tarayan daraltilmis sweep.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-runs", type=int, default=48)
    parser.add_argument("--selection-mode", choices=["even", "random", "first"], default="even")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Komut basarisiz oldu: {' '.join(command)}")


def compact_float(value: float) -> str:
    return f"{value:.10g}".replace(".", "").replace("-", "m")


def config_slug(config: dict[str, object]) -> str:
    return "_".join(
        [
            str(config["model"]),
            f"ep{config['epochs']}",
            f"bs{config['batch_size']}",
            f"lr{compact_float(float(config['lr']))}",
            f"wd{compact_float(float(config['weight_decay']))}",
            f"do{compact_float(float(config['dropout']))}",
            f"ls{compact_float(float(config['label_smoothing']))}",
            str(config["scheduler"]),
            f"min{compact_float(float(config['min_lr']))}",
            f"aug{config['augment']}",
            f"sampler{config['sampler']}",
            f"weight{config['class_weight_mode']}",
        ]
    )


def is_reasonable(config: dict[str, object]) -> bool:
    # Ilk 28 kosudaki sinyale gore zayif kombinasyonlari hafif buda.
    if config["sampler"] == "weighted" and config["class_weight_mode"] == "balanced":
        if float(config["lr"]) == 3e-4 and float(config["dropout"]) == 0.3:
            return False
    if config["augment"] == "light" and float(config["label_smoothing"]) == 0.1:
        if float(config["weight_decay"]) == 5e-4:
            return False
    return True


def generate_all_configs() -> list[dict[str, object]]:
    values = [REFINED_SPACE[key] for key in PARAMETER_KEYS]
    configs: list[dict[str, object]] = []
    for combo in itertools.product(*values):
        config = {key: value for key, value in zip(PARAMETER_KEYS, combo, strict=True)}
        if is_reasonable(config):
            configs.append(config)
    return configs


def select_configs(configs: list[dict[str, object]], max_runs: int, selection_mode: str, seed: int) -> list[dict[str, object]]:
    if max_runs <= 0 or max_runs >= len(configs):
        return list(configs)
    if selection_mode == "first":
        return configs[:max_runs]
    if selection_mode == "random":
        indices = list(range(len(configs)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        chosen = sorted(indices[:max_runs])
        return [configs[index] for index in chosen]
    if max_runs == 1:
        return [configs[0]]
    step = (len(configs) - 1) / (max_runs - 1)
    chosen_indices: list[int] = []
    used: set[int] = set()
    for i in range(max_runs):
        idx = int(round(i * step))
        while idx in used and idx + 1 < len(configs):
            idx += 1
        if idx in used:
            idx = max(candidate for candidate in range(len(configs)) if candidate not in used)
        used.add(idx)
        chosen_indices.append(idx)
    return [configs[index] for index in chosen_indices]


def summary_from_report(report: dict[str, object], output_dir: Path) -> dict[str, object]:
    validation = report["validation"]
    test = report["test"]
    return {
        "run_dir": str(output_dir),
        "model": report["model"],
        "epochs": report["epochs"],
        "best_epoch": report["best_epoch"],
        "lr": report["lr"],
        "weight_decay": report["weight_decay"],
        "dropout": report["dropout"],
        "label_smoothing": report["label_smoothing"],
        "scheduler": report["scheduler"],
        "min_lr": report["min_lr"],
        "augment": report["augment"],
        "sampler": report["sampler"],
        "class_weight_mode": report["class_weight_mode"],
        "val_macro_f1": validation["macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "val_iletim_f1": validation["per_class_f1"]["iletim"],
        "test_iletim_f1": test["per_class_f1"]["iletim"],
        "confusion_matrix_test_png": str(output_dir / "visuals" / "confusion_matrix_test.png"),
        "gradcam_summary_test_json": str(output_dir / "visuals" / "gradcam_summary_test.json"),
    }


def config_to_command(config: dict[str, object], output_dir: Path) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--manifest",
        str(MANIFEST),
        "--output-dir",
        str(output_dir),
        "--model",
        str(config["model"]),
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--dropout",
        str(config["dropout"]),
        "--label-smoothing",
        str(config["label_smoothing"]),
        "--scheduler",
        str(config["scheduler"]),
        "--min-lr",
        str(config["min_lr"]),
        "--augment",
        str(config["augment"]),
        "--sampler",
        str(config["sampler"]),
        "--class-weight-mode",
        str(config["class_weight_mode"]),
    ]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_configs = generate_all_configs()
    selected_configs = select_configs(all_configs, args.max_runs, args.selection_mode, args.seed)

    indexed_configs: list[dict[str, object]] = []
    for run_index, config in enumerate(selected_configs, start=args.start_index):
        item = dict(config)
        item["run_name"] = f"run{run_index:03d}_{config_slug(item)}"
        indexed_configs.append(item)

    plan = {
        "refined_space": REFINED_SPACE,
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "requested_max_runs": args.max_runs,
        "total_possible_runs": len(all_configs),
        "total_selected_runs": len(indexed_configs),
        "runs": indexed_configs,
    }
    (args.output_root / "refined_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        json.dump(plan, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    run_summaries: list[dict[str, object]] = []
    for config in indexed_configs:
        output_dir = args.output_root / str(config["run_name"])
        report_path = output_dir / "training_report.json"
        gradcam_summary = output_dir / "visuals" / "gradcam_summary_test.json"
        if args.skip_completed and report_path.exists() and gradcam_summary.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            run_summaries.append(summary_from_report(report, output_dir))
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        run_command(config_to_command(config, output_dir))

        visuals_dir = output_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
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
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = summary_from_report(report, output_dir)
        run_summaries.append(summary)
        (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_summary = {
        "refined_space": REFINED_SPACE,
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "requested_max_runs": args.max_runs,
        "total_possible_runs": len(all_configs),
        "total_selected_runs": len(run_summaries),
        "best_by_validation_macro_f1": sorted(run_summaries, key=lambda item: item["val_macro_f1"], reverse=True)[:10],
        "best_by_test_macro_f1": sorted(run_summaries, key=lambda item: item["test_macro_f1"], reverse=True)[:10],
        "best_by_test_iletim_f1": sorted(run_summaries, key=lambda item: item["test_iletim_f1"], reverse=True)[:10],
        "all_runs": run_summaries,
    }
    (args.output_root / "refined_summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    json.dump(final_summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
