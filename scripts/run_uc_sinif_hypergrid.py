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
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_uc_sinif_hparam.py"
VIS_SCRIPT = REPO_ROOT / "scripts" / "render_confusion_and_gradcam.py"
MANIFEST = REPO_ROOT / "artifacts" / "uc_sinif" / "manifest_uc_sinif_temiz.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "uc_sinif_hypergrid"


BASE_SPACE = {
    "model": ["baseline", "resnet_se"],
    "epochs": [50],
    "batch_size": [64],
    "lr": [1e-3, 5e-4, 3e-4, 1e-4],
    "weight_decay": [1e-4, 5e-4],
    "dropout": [0.2, 0.3, 0.4],
    "label_smoothing": [0.0, 0.05, 0.1],
    "augment": ["none", "light"],
    "sampler": ["none", "weighted"],
    "class_weight_mode": ["balanced", "none"],
}

SCHEDULER_SPACE = {
    "none": [{}],
    "cosine": [
        {"min_lr": 1e-6},
        {"min_lr": 1e-5},
    ],
    "plateau": [
        {"min_lr": 1e-6, "plateau_factor": 0.5, "plateau_patience": 3},
        {"min_lr": 1e-6, "plateau_factor": 0.5, "plateau_patience": 5},
        {"min_lr": 1e-5, "plateau_factor": 0.3, "plateau_patience": 4},
    ],
    "onecycle": [
        {"min_lr": 1e-6, "onecycle_pct_start": 0.2},
        {"min_lr": 1e-6, "onecycle_pct_start": 0.3},
        {"min_lr": 1e-5, "onecycle_pct_start": 0.4},
    ],
    "cosine_warm_restarts": [
        {"min_lr": 1e-6, "warm_restart_t0": 10, "warm_restart_tmult": 2},
        {"min_lr": 1e-6, "warm_restart_t0": 5, "warm_restart_tmult": 2},
        {"min_lr": 1e-5, "warm_restart_t0": 10, "warm_restart_tmult": 1},
    ],
    "step": [
        {"scheduler_gamma": 0.5, "scheduler_step_size": 10},
        {"scheduler_gamma": 0.3, "scheduler_step_size": 15},
    ],
    "multistep": [
        {"scheduler_gamma": 0.5, "scheduler_milestones": [15, 30, 40]},
        {"scheduler_gamma": 0.3, "scheduler_milestones": [10, 20, 35]},
    ],
}

BASE_KEYS = list(BASE_SPACE.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genel hiperparametre grid sistemi.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-runs", type=int, default=64)
    parser.add_argument("--selection-mode", choices=["even", "random", "first"], default="even")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
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
    parts = [
        str(config["model"]).replace("_", ""),
        f"ep{config['epochs']}",
        f"bs{config['batch_size']}",
        f"lr{compact_float(float(config['lr']))}",
        f"wd{compact_float(float(config['weight_decay']))}",
        f"do{compact_float(float(config['dropout']))}",
        f"ls{compact_float(float(config['label_smoothing']))}",
        str(config["scheduler"]),
        f"aug{config['augment']}",
        f"sampler{config['sampler']}",
        f"weight{config['class_weight_mode']}",
    ]
    if "min_lr" in config:
        parts.append(f"min{compact_float(float(config['min_lr']))}")
    if "scheduler_gamma" in config:
        parts.append(f"gamma{compact_float(float(config['scheduler_gamma']))}")
    if "scheduler_step_size" in config:
        parts.append(f"step{config['scheduler_step_size']}")
    if "scheduler_milestones" in config:
        joined = "-".join(str(x) for x in config["scheduler_milestones"])
        parts.append(f"ms{joined}")
    if "plateau_factor" in config:
        parts.append(f"pf{compact_float(float(config['plateau_factor']))}")
    if "plateau_patience" in config:
        parts.append(f"pp{config['plateau_patience']}")
    if "onecycle_pct_start" in config:
        parts.append(f"oc{compact_float(float(config['onecycle_pct_start']))}")
    if "warm_restart_t0" in config:
        parts.append(f"t0{config['warm_restart_t0']}")
    if "warm_restart_tmult" in config:
        parts.append(f"tm{config['warm_restart_tmult']}")
    return "_".join(parts)


def is_reasonable(config: dict[str, object]) -> bool:
    if config["sampler"] == "weighted" and config["class_weight_mode"] == "balanced":
        if config["model"] == "baseline" and float(config["lr"]) == 1e-4:
            return False
    if config["augment"] == "light" and float(config["dropout"]) == 0.4 and config["model"] == "baseline":
        if config["scheduler"] in {"plateau", "multistep"}:
            return False
    return True


def generate_all_configs() -> list[dict[str, object]]:
    base_values = [BASE_SPACE[key] for key in BASE_KEYS]
    configs: list[dict[str, object]] = []
    for base_combo in itertools.product(*base_values):
        base_config = {key: value for key, value in zip(BASE_KEYS, base_combo, strict=True)}
        for scheduler_name, scheduler_variants in SCHEDULER_SPACE.items():
            for scheduler_params in scheduler_variants:
                config = dict(base_config)
                config["scheduler"] = scheduler_name
                config.update(scheduler_params)
                if is_reasonable(config):
                    configs.append(config)
    return configs


def shard_configs(configs: list[dict[str, object]], shard_count: int, shard_index: int) -> list[dict[str, object]]:
    if shard_count <= 0:
        raise ValueError("--shard-count pozitif olmali.")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index [0, shard-count) araliginda olmali.")
    return [config for idx, config in enumerate(configs) if idx % shard_count == shard_index]


def select_configs(
    configs: list[dict[str, object]],
    max_runs: int,
    selection_mode: str,
    seed: int,
) -> list[dict[str, object]]:
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


def config_to_train_args(config: dict[str, object], output_dir: Path) -> list[str]:
    command = [
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
        "--augment",
        str(config["augment"]),
        "--sampler",
        str(config["sampler"]),
        "--class-weight-mode",
        str(config["class_weight_mode"]),
    ]
    optional_keys = {
        "min_lr": "--min-lr",
        "scheduler_gamma": "--scheduler-gamma",
        "scheduler_step_size": "--scheduler-step-size",
        "plateau_factor": "--plateau-factor",
        "plateau_patience": "--plateau-patience",
        "onecycle_pct_start": "--onecycle-pct-start",
        "warm_restart_t0": "--warm-restart-t0",
        "warm_restart_tmult": "--warm-restart-tmult",
    }
    for key, flag in optional_keys.items():
        if key in config:
            command.extend([flag, str(config[key])])
    if "scheduler_milestones" in config:
        command.append("--scheduler-milestones")
        command.extend(str(item) for item in config["scheduler_milestones"])
    return command


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_configs = generate_all_configs()
    sharded_configs = shard_configs(all_configs, args.shard_count, args.shard_index)
    selected_configs = select_configs(sharded_configs, args.max_runs, args.selection_mode, args.seed)

    indexed_configs: list[dict[str, object]] = []
    for run_index, config in enumerate(selected_configs, start=args.start_index):
        item = dict(config)
        item["run_name"] = f"run{run_index:04d}_{config_slug(item)}"
        indexed_configs.append(item)

    plan = {
        "base_space": BASE_SPACE,
        "scheduler_space": SCHEDULER_SPACE,
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "requested_max_runs": args.max_runs,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "total_possible_runs": len(all_configs),
        "total_runs_after_shard": len(sharded_configs),
        "total_selected_runs": len(indexed_configs),
        "runs": indexed_configs,
    }
    (args.output_root / "hypergrid_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        json.dump(plan, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    run_summaries: list[dict[str, object]] = []
    for config in indexed_configs:
        output_dir = args.output_root / str(config["run_name"])
        report_path = output_dir / "training_report.json"
        visuals_path = output_dir / "visuals" / "gradcam_summary_test.json"
        if args.skip_completed and report_path.exists() and visuals_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            run_summaries.append(summary_from_report(report, output_dir))
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        run_command(config_to_train_args(config, output_dir))

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
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    final_summary = {
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "requested_max_runs": args.max_runs,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "total_possible_runs": len(all_configs),
        "total_runs_after_shard": len(sharded_configs),
        "total_selected_runs": len(run_summaries),
        "best_by_validation_macro_f1": sorted(run_summaries, key=lambda item: item["val_macro_f1"], reverse=True)[:10],
        "best_by_test_macro_f1": sorted(run_summaries, key=lambda item: item["test_macro_f1"], reverse=True)[:10],
        "best_by_test_iletim_f1": sorted(run_summaries, key=lambda item: item["test_iletim_f1"], reverse=True)[:10],
        "all_runs": run_summaries,
    }
    (args.output_root / "hypergrid_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    json.dump(final_summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
