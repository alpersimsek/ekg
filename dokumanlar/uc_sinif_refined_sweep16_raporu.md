# Uc Sinif Refined Sweep16 Raporu

Tarih: 24 Mart 2026

Bu rapor, `/home/alper/ekg/artifacts/uc_sinif_refined_sweep16` altinda kosulan daraltilmis ikinci tur pilot sweep sonucunu ozetler.

## 1. Genel Durum

Tamamlanan kosu sayisi:

- `16 / 16`

Her kosu icin olusan ana dosyalar:

- `training_report.json`
- `run_summary.json`
- `visuals/confusion_matrix_test.png`
- `visuals/gradcam_summary_test.json`

Bu tur:

- `baseline` mimarisi ile sinirlandi
- `cosine` scheduler ile sinirlandi
- ilk 28 kosudan guclu gorunen hiperparametre bolgesine odaklandi

## 2. Ana Sonuclar

Validation lideri:

- [run006_baseline_ep50_bs64_lr0001_wd00005_do03_ls01_cosine_min1em06_augnone_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run006_baseline_ep50_bs64_lr0001_wd00005_do03_ls01_cosine_min1em06_augnone_samplernone_weightnone)
- validation `Macro F1 = 0.930251`
- test `Macro F1 = 0.925708`

Toplam test lideri:

- [run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced)
- test `Macro F1 = 0.934861`
- test `iletim F1 = 0.893151`

`iletim` lideri:

- [run007_baseline_ep50_bs64_lr00005_wd00001_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run007_baseline_ep50_bs64_lr00005_wd00001_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone)
- test `Macro F1 = 0.934375`
- test `iletim F1 = 0.911602`

## 3. Teknik Gozlem

Bu turdan cikan guclu sinyaller:

- `baseline` mimarisi ikinci kez dogrulandi
- `lr = 1e-3` ve `5e-4` hala guclu bolgeler
- `weight_decay = 5e-4` bazi kosularda anlamli katkı verdi
- `weighted sampler` toplam skoru her zaman bozmadı
- aksine `iletim` tarafinda bazen belirgin fayda verdi
- `label_smoothing = 0.05` ve `0.1` halen rekabetci

## 4. Ara Karar

Bu pilot turun sonunda iki ana aday ortaya cikti:

- toplam kalite odakli aday: `run004`
- azinlik sinif duyarliligi odakli aday: `run007`

Sonraki karar mekanizmasi bu iki kosu arasinda netlesmelidir.
