# Final Model V3

Bu klasor, artik devam edilecek aktif model yapisini tutar.

Model tipi:

- iki-asamali sistem
- `base 3-sinif model + aux binary iletim modeli + threshold kuralı`

Ana dosyalar:

- [base_best_model.pt](/home/alper/ekg/final_model_v3/base_best_model.pt)
- [aux_binary/best_model.pt](/home/alper/ekg/final_model_v3/aux_binary/best_model.pt)
- [v3_two_stage_report.json](/home/alper/ekg/final_model_v3/v3_two_stage_report.json)
- [FINAL_DURUM_RAPORU.md](/home/alper/ekg/final_model_v3/FINAL_DURUM_RAPORU.md)
- [INFERENCE_SERVIS_REHBERI.md](/home/alper/ekg/final_model_v3/INFERENCE_SERVIS_REHBERI.md)

Seçilen kural:

- `positive_threshold = 0.83`
- `negative_threshold = 0.21`

Ana test sonucu:

- test `Macro F1 = 0.939617`
- test `iletim F1 = 0.908602`
