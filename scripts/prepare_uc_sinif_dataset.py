from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scipy.io import loadmat


RNG_SEED = 42
NORMAL_CODE = "426783006"
MAIN_CLASSES = ("normal", "ritim", "iletim")
EXTRA_CLASSES = ("morfolojik_iskemik", "exclude")
ALL_CLASSES = MAIN_CLASSES + EXTRA_CLASSES


@dataclass(frozen=True)
class RecordMeta:
    record_id: str
    hea_path: Path
    mat_path: Path
    num_leads: int
    sampling_rate: int
    num_samples: int
    lead_names: tuple[str, ...]
    diagnoses: tuple[str, ...]
    age: str | None
    sex: str | None
    malformed_first_line: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3 sinif EKG veri kumesi icin Sprint 2 manifest ve split hazirlar."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/alper/ekg/data/data_3_Sinif"),
        help="WFDB .hea/.mat dosyalarinin bulundugu veri koku.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("/home/alper/ekg/yedek/20260323_182919/yeni_yaklasım/siniflar_final.csv"),
        help="SNOMED -> kategori esleme CSV dosyasi.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/alper/ekg/artifacts/uc_sinif"),
        help="Manifest ve ozet dosyalarinin yazilacagi dizin.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
        help="Split olusturma tohumu.",
    )
    parser.add_argument(
        "--tensor-audit-sample-size",
        type=int,
        default=256,
        help="Rastgele secilip .mat tensori yuklenerek kontrol edilecek kayit sayisi.",
    )
    return parser.parse_args()


def load_mapping(mapping_csv: Path) -> dict[str, str]:
    with mapping_csv.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["snomed_code"].strip(): row["category"].strip()
            for row in csv.DictReader(handle)
            if row["snomed_code"].strip()
        }


def parse_header(hea_path: Path) -> RecordMeta:
    lines = hea_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Bos header: {hea_path}")

    first_parts = lines[0].split()
    if len(first_parts) < 4:
        raise ValueError(f"Beklenmeyen ilk satir: {hea_path}")

    record_id = hea_path.stem
    num_leads = int(first_parts[1])
    sampling_rate = int(float(first_parts[2]))
    malformed_first_line = False
    first_line_signal_tokens: list[str] | None = None

    try:
        num_samples = int(first_parts[3])
        if len(first_parts) > 4 and first_parts[4].endswith(".mat"):
            first_line_signal_tokens = first_parts[4:]
    except ValueError:
        num_samples = 5000
        malformed_first_line = True

    mat_path: Path | None = None
    lead_names: list[str] = []

    if first_line_signal_tokens:
        signal_lines = [" ".join(first_line_signal_tokens)] + lines[1:num_leads]
    elif malformed_first_line:
        mat_path = hea_path.with_name(f"{record_id}.mat")
        signal_lines = lines[1:num_leads]
    else:
        signal_lines = lines[1 : 1 + num_leads]

    for line in signal_lines:
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Beklenmeyen sinyal tanimi: {hea_path} -> {line!r}")
        if mat_path is None:
            mat_token = parts[0] if parts[0].endswith(".mat") else f"{record_id}.mat"
            mat_path = hea_path.with_name(mat_token)
        lead_names.append(parts[-1])

    metadata_start = num_leads if (first_line_signal_tokens or malformed_first_line) else 1 + num_leads
    metadata: dict[str, str] = {}
    for line in lines[metadata_start:]:
        if not line.startswith("#") or ":" not in line:
            continue
        key, value = line[1:].split(":", 1)
        metadata[key.strip().lower()] = value.strip()

    if mat_path is None:
        raise ValueError(f".mat yolu cozulmedi: {hea_path}")

    diagnoses = tuple(dx.strip() for dx in metadata.get("dx", "").split(",") if dx.strip())
    return RecordMeta(
        record_id=record_id,
        hea_path=hea_path,
        mat_path=mat_path,
        num_leads=num_leads,
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        lead_names=tuple(lead_names),
        diagnoses=diagnoses,
        age=metadata.get("age"),
        sex=metadata.get("sex"),
        malformed_first_line=malformed_first_line or bool(first_line_signal_tokens),
    )


def classify_record(meta: RecordMeta, mapping: dict[str, str]) -> dict[str, object]:
    flags = {name: 0 for name in ALL_CLASSES}
    unmapped_codes: list[str] = []

    if meta.diagnoses and set(meta.diagnoses) == {NORMAL_CODE}:
        flags["normal"] = 1
    else:
        for code in meta.diagnoses:
            if code == NORMAL_CODE:
                continue
            category = mapping.get(code)
            if category in flags:
                flags[category] = 1
            else:
                unmapped_codes.append(code)

    main_hits = [name for name in MAIN_CLASSES if flags[name] == 1]
    exclusion_reason = ""
    target_class = ""

    if unmapped_codes:
        exclusion_reason = "unmapped_code"
    elif flags["exclude"] == 1:
        exclusion_reason = "technical_exclude"
    elif flags["morfolojik_iskemik"] == 1:
        exclusion_reason = "morfolojik_iskemik"
    elif len(main_hits) == 0:
        exclusion_reason = "no_target_class"
    elif len(main_hits) > 1:
        exclusion_reason = "multiple_main_classes"
    else:
        target_class = main_hits[0]

    return {
        **flags,
        "target_class": target_class,
        "main_hit_count": len(main_hits),
        "unmapped_codes": ",".join(unmapped_codes),
        "exclusion_reason": exclusion_reason,
    }


def stratified_split(rows: list[dict[str, str]], seed: int) -> None:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = {name: [] for name in MAIN_CLASSES}
    for row in rows:
        by_class[row["target_class"]].append(row)

    for class_rows in by_class.values():
        rng.shuffle(class_rows)
        total = len(class_rows)
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)
        for index, row in enumerate(class_rows):
            if index < train_end:
                row["split"] = "egitim"
            elif index < val_end:
                row["split"] = "dogrulama"
            else:
                row["split"] = "test"


def audit_tensor_shapes(clean_rows: list[dict[str, str]], sample_size: int, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    sample_rows = clean_rows if len(clean_rows) <= sample_size else rng.sample(clean_rows, sample_size)
    issues: list[dict[str, object]] = []

    for row in sample_rows:
        mat_path = Path(row["mat_yolu"])
        try:
            payload = loadmat(mat_path)
            shape = tuple(payload["val"].shape)
            if shape != (12, 5000):
                issues.append({"record_id": row["kayit_id"], "shape": shape})
        except Exception as exc:  # pragma: no cover - defensive audit path
            issues.append({"record_id": row["kayit_id"], "error": type(exc).__name__})

    return {
        "sample_size": len(sample_rows),
        "issue_count": len(issues),
        "issues": issues[:20],
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_rows(dataset_root: Path, mapping: dict[str, str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    hea_files = sorted(dataset_root.glob("*.hea"))
    mat_stems = {path.stem for path in dataset_root.glob("*.mat")}
    zero_byte_files = [
        path.name
        for path in list(dataset_root.glob("*.hea")) + list(dataset_root.glob("*.mat"))
        if path.stat().st_size == 0
    ]

    all_rows: list[dict[str, object]] = []
    format_counter: Counter[tuple[int, int, int]] = Counter()
    malformed_headers: list[str] = []
    missing_mat: list[str] = []

    for hea_path in hea_files:
        meta = parse_header(hea_path)
        if meta.record_id not in mat_stems:
            missing_mat.append(meta.record_id)
        format_counter[(meta.num_leads, meta.sampling_rate, meta.num_samples)] += 1
        if meta.malformed_first_line:
            malformed_headers.append(meta.record_id)

        classification = classify_record(meta, mapping)
        row = {
            "kayit_id": meta.record_id,
            "hasta_id": meta.record_id,
            "hea_yolu": str(meta.hea_path),
            "mat_yolu": str(meta.mat_path),
            "tani_kodlari": ",".join(meta.diagnoses),
            "yas": meta.age or "",
            "cinsiyet": meta.sex or "",
            "kanal_sayisi": meta.num_leads,
            "ornekleme_hizi": meta.sampling_rate,
            "ornek_sayisi": meta.num_samples,
            "lead_names": ",".join(meta.lead_names),
            "normal": classification["normal"],
            "ritim": classification["ritim"],
            "iletim": classification["iletim"],
            "morfolojik_iskemik": classification["morfolojik_iskemik"],
            "exclude": classification["exclude"],
            "target_class": classification["target_class"],
            "unmapped_codes": classification["unmapped_codes"],
            "exclusion_reason": classification["exclusion_reason"],
            "split": "",
        }
        all_rows.append(row)

    audit = {
        "toplam_kayit": len(all_rows),
        "hea_sayisi": len(hea_files),
        "mat_sayisi": len(mat_stems),
        "eksik_mat_sayisi": len(missing_mat),
        "eksik_mat_ornekleri": missing_mat[:20],
        "sifir_byte_dosya_sayisi": len(zero_byte_files),
        "sifir_byte_ornekleri": zero_byte_files[:20],
        "header_format_dagilimi": {
            f"{key[0]}_{key[1]}_{key[2]}": value for key, value in sorted(format_counter.items())
        },
        "malformed_header_sayisi": len(malformed_headers),
        "malformed_header_ornekleri": malformed_headers[:20],
    }
    return all_rows, audit


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping_csv)
    all_rows, audit = build_rows(args.dataset_root, mapping)
    clean_rows = [row for row in all_rows if row["target_class"]]
    stratified_split(clean_rows, seed=args.seed)

    split_counts: dict[str, dict[str, int]] = {
        split: {class_name: 0 for class_name in MAIN_CLASSES}
        for split in ("egitim", "dogrulama", "test")
    }
    for row in clean_rows:
        split_counts[row["split"]][row["target_class"]] += 1

    class_counts = Counter(row["target_class"] for row in clean_rows)
    total_clean = len(clean_rows)
    class_weights = {
        class_name: total_clean / (len(MAIN_CLASSES) * max(count, 1))
        for class_name, count in class_counts.items()
    }
    exclusion_counts = Counter(row["exclusion_reason"] for row in all_rows if row["exclusion_reason"])

    tensor_audit = audit_tensor_shapes(clean_rows, args.tensor_audit_sample_size, seed=args.seed)

    fieldnames = [
        "kayit_id",
        "hasta_id",
        "hea_yolu",
        "mat_yolu",
        "tani_kodlari",
        "yas",
        "cinsiyet",
        "kanal_sayisi",
        "ornekleme_hizi",
        "ornek_sayisi",
        "lead_names",
        "normal",
        "ritim",
        "iletim",
        "morfolojik_iskemik",
        "exclude",
        "target_class",
        "unmapped_codes",
        "exclusion_reason",
        "split",
    ]
    write_csv(args.output_dir / "manifest_tum_kayitlar.csv", all_rows, fieldnames)
    write_csv(args.output_dir / "manifest_uc_sinif_temiz.csv", clean_rows, fieldnames)

    summary = {
        "dataset_root": str(args.dataset_root),
        "mapping_csv": str(args.mapping_csv),
        "seed": args.seed,
        "audit": audit,
        "temiz_kohort_toplami": total_clean,
        "sinif_sayilari": dict(class_counts),
        "sinif_agirliklari": class_weights,
        "split_sayilari": split_counts,
        "dislama_nedenleri": dict(exclusion_counts),
        "on_isleme_sozlesmesi": {
            "normalizasyon": "lead_bazli_zscore",
            "filtreleme": "varsayilan degil, deneysel olarak dogrulanirsa eklenecek",
            "kanal_duzeni": "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6",
            "hedef_tensor_sekli": [12, 5000],
        },
        "tensor_audit": tensor_audit,
    }
    (args.output_dir / "hazirlik_ozeti.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
