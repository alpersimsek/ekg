from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from final_model_v3_inference import FINAL_MODEL_ROOT, FinalModelV3Predictor, load_rows, load_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="final_model_v3 icin toplu tahmin uretir.")
    parser.add_argument("--manifest", type=Path, default=Path("/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv"))
    parser.add_argument("--split", choices=["egitim", "dogrulama", "test", "all"], default="test")
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FinalModelV3Predictor.load()
    rows = load_rows(args.manifest, args.split)
    outputs = []
    for row in rows:
        pred = predictor.predict_tensor(load_signal(Path(row["mat_yolu"])))
        outputs.append(
            {
                "kayit_id": row["kayit_id"],
                "mat_yolu": row["mat_yolu"],
                "gercek_sinif": row.get("target_class", ""),
                "base_tahmin_sinif": pred["base_predicted_class"],
                "tahmin_sinif": pred["predicted_class"],
                "override_type": pred["override_type"],
                "normal_olasilik": pred["base_probabilities"]["normal"],
                "ritim_olasilik": pred["base_probabilities"]["ritim"],
                "iletim_olasilik": pred["base_probabilities"]["iletim"],
                "aux_iletim_olasilik": pred["aux_probabilities"]["iletim"],
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0].keys()))
        writer.writeheader()
        writer.writerows(outputs)
    print(json.dumps({"output_csv": str(args.output_csv), "rows": len(outputs), "model_root": str(FINAL_MODEL_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
