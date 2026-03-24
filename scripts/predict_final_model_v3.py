from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_model_v3_inference import AUX_CHECKPOINT, BASE_CHECKPOINT, REPORT_JSON, FinalModelV3Predictor, load_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="final_model_v3 icin tek EKG kaydi tahmini uretir.")
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--aux-checkpoint", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FinalModelV3Predictor.load(
        base_checkpoint=args.base_checkpoint or BASE_CHECKPOINT,
        aux_checkpoint=args.aux_checkpoint or AUX_CHECKPOINT,
        report_json=args.report_json or REPORT_JSON,
    )
    result = predictor.predict_tensor(load_signal(args.mat_path))
    result["mat_path"] = str(args.mat_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
