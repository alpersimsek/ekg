# EKG 3-Sınıf Sınıflandırma Projesi

Bu repo, 12 derivasyonlu EKG sinyallerinden tek etiketli 3-sınıf tahmin üretmek için hazırlanmıştır.

Sınıflar:

- `normal`
- `ritim`
- `iletim`

Ana hedef metrik:

- `Macro F1`

## Veri

Kaynak veri kümesi:

- `/home/alper/ekg/data/data_3_Sinif`

Temiz manifest:

- [manifest_uc_sinif_temiz.csv](/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv)

## Seçilen Nihai Model

Aktif model kökü:

- [final_model_v3](/home/alper/ekg/final_model_v3)

Bu yapı tek checkpoint değildir:

- ana model: [base_best_model.pt](/home/alper/ekg/final_model_v3/base_best_model.pt)
- yardımcı binary model: [best_model.pt](/home/alper/ekg/final_model_v3/aux_binary/best_model.pt)
- karar raporu: [v3_two_stage_report.json](/home/alper/ekg/final_model_v3/v3_two_stage_report.json)

Seçilen iki aşamalı kural:

- `positive_threshold = 0.83`
- `negative_threshold = 0.21`

Test sonucu:

- `Macro F1 = 0.939617`
- `Accuracy = 0.953495`
- `iletim F1 = 0.908602`

## Hızlı Başlangıç

Tek kayıt tahmini:

```bash
.venv/bin/python scripts/predict_final_model_v3.py --mat-path /home/alper/ekg/data/data_3_Sinif/JS00002.mat
```

Batch tahmin:

```bash
.venv/bin/python scripts/predict_final_model_v3_batch.py --split test --output-csv /home/alper/ekg/final_model_v3/inference/test_predictions.csv
```

## Önemli Dizinler

- [scripts](/home/alper/ekg/scripts): eğitim, analiz, inference ve servis scriptleri
- [dokumanlar](/home/alper/ekg/dokumanlar): sprint raporları ve teknik özetler
- [final_model_v3](/home/alper/ekg/final_model_v3): seçilen model, raporlar ve görseller

## Ana Dokümanlar

- [FINAL_DURUM_RAPORU.md](/home/alper/ekg/final_model_v3/FINAL_DURUM_RAPORU.md)
- [INFERENCE_SERVIS_REHBERI.md](/home/alper/ekg/final_model_v3/INFERENCE_SERVIS_REHBERI.md)
- [kanit_bazli_proje_ozeti.md](/home/alper/ekg/dokumanlar/rapor/kanit_bazli_proje_ozeti.md)
