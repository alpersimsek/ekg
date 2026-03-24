# Final Model V3 Inference ve Servis Rehberi

Bu model icin yeni inference katmani eklendi.

Scriptler:

- [final_model_v3_inference.py](/home/alper/ekg/scripts/final_model_v3_inference.py)
- [predict_final_model_v3.py](/home/alper/ekg/scripts/predict_final_model_v3.py)
- [predict_final_model_v3_batch.py](/home/alper/ekg/scripts/predict_final_model_v3_batch.py)
- [serve_final_model_v3.py](/home/alper/ekg/scripts/serve_final_model_v3.py)

## Tek Kayit Tahmini

Komut:

```bash
.venv/bin/python scripts/predict_final_model_v3.py --mat-path /home/alper/ekg/data/data_3_Sinif/JS00002.mat
```

Bu komut test edildi.

## Batch Tahmin

Komut:

```bash
.venv/bin/python scripts/predict_final_model_v3_batch.py --split test --output-csv /home/alper/ekg/final_model_v3/inference/test_predictions.csv
```

Bu komut test edildi.

Uretilen cikti:

- [test_predictions.csv](/home/alper/ekg/final_model_v3/inference/test_predictions.csv)

## Servis Dosyasi

FastAPI servis girisi hazir:

- [serve_final_model_v3.py](/home/alper/ekg/scripts/serve_final_model_v3.py)

Beklenen calistirma ornegi:

```bash
.venv/bin/python -m uvicorn scripts.serve_final_model_v3:app --host 0.0.0.0 --port 8000
```

Not:

- bu ortamda `fastapi` ve `uvicorn` kurulu degil
- bu nedenle servis dosyasi yazildi ama burada ayağa kaldirilip test edilmedi

## Karar Mantigi

Tahmin akisi:

- base model 3-sinif olasiliklarini uretir
- aux binary model `iletim` olasiligini uretir
- secilen kural uygulanir:
- `aux_iletim >= 0.83` ise tahmin `iletim`
- base model `iletim` dese bile `aux_iletim < 0.21` ise tahmin `normal/ritim` icinde en guclu olana geri cekilir
