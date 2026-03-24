# Run004 ve Run007 Karsilastirma Karar Notu

Tarih: 24 Mart 2026

Bu not, refined sweep16 sonunda one cikan iki adayi karsilastirmak icin hazirlanmistir.

## 1. Adaylar

Toplam test lideri:

- [run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced)

`iletim` lideri:

- [run007_baseline_ep50_bs64_lr00005_wd00001_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run007_baseline_ep50_bs64_lr00005_wd00001_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone)

## 2. Parametre Farki

`run004`:

- `lr = 1e-3`
- `weight_decay = 5e-4`
- `dropout = 0.2`
- `label_smoothing = 0.0`
- `cosine`
- `min_lr = 1e-6`
- `augment = light`
- `sampler = none`
- `class_weight_mode = balanced`

`run007`:

- `lr = 5e-4`
- `weight_decay = 1e-4`
- `dropout = 0.2`
- `label_smoothing = 0.05`
- `cosine`
- `min_lr = 1e-5`
- `augment = none`
- `sampler = weighted`
- `class_weight_mode = none`

## 3. Performans Farki

`run004`:

- validation `Macro F1 = 0.923560`
- test `Macro F1 = 0.934861`
- test `iletim F1 = 0.893151`

`run007`:

- validation `Macro F1 = 0.922401`
- test `Macro F1 = 0.934375`
- test `iletim F1 = 0.911602`

## 4. Yorum

Toplam skor farki:

- `run004` toplam test `Macro F1` tarafinda biraz daha iyi
- fark cok buyuk degil

Azinlik sinif farki:

- `run007`, `iletim F1` tarafinda belirgin sekilde daha iyi
- bu fark pratikte daha anlamli olabilir

Sezgisel yorum:

- eger amac genel en yuksek toplam skor ise `run004`
- eger amac `iletim` sinifini daha guvenli yakalamak ise `run007`

## 5. Oneri

Varsayilan uretim adayi olarak:

- `run007`

Gerekce:

- toplam `Macro F1` farki kucuk
- `iletim F1` kazanci anlamli
- 3 sinifli bu problemde azinlik sinif kalitesi yalniz toplam skordan daha kritik olabilir

Ama eger kurum ici karar “tek KPI toplam Macro F1” ise:

- `run004`

## 6. Sonraki Adim

Karar verilmeden once bu iki kosu icin su dosyalar yan yana incelenmeli:

- [run004 confusion matrix](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced/visuals/confusion_matrix_test.png)
- [run007 confusion matrix](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run007_baseline_ep50_bs64_lr00005_wd00001_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone/visuals/confusion_matrix_test.png)
- [run004 Grad-CAM summary](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced/visuals/gradcam_summary_test.json)
- [run007 Grad-CAM summary](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run007_baseline_ep50_bs64_lr00005_wd00001_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone/visuals/gradcam_summary_test.json)
