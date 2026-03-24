# 64 Kosuluk Sweep Ilk 28 Kosu Ara Raporu

Tarih: 24 Mart 2026

Bu rapor, `/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64` altinda baslatilan 64 kosuluk sweep turunun ilk tamamlanan 28 kosusunu ozetler.

## 1. Temizlik Durumu

Yarida kalan kosu:

- `run029_baseline_lr00001_do03_ls01_cosine_min1em06_augnone_samplernone_weightbalanced`

Durum:

- bu kosuda tam `run_summary.json` olusmamistı
- yarim kalan klasor silindi

Mevcut sayi:

- tamamlanmis kosu sayisi: `28`

## 2. Genel Ozet

Ilk 28 kosunun toplu ortalama degerleri:

- validation `Macro F1` ortalama: `0.914311`
- test `Macro F1` ortalama: `0.920392`
- validation `Macro F1` medyan: `0.914272`
- test `Macro F1` medyan: `0.920929`

Yorum:

- ilk 28 kosunun genel kalite seviyesi yuksek
- skorlarin ortalama ve medyan degerlerinin birbirine yakin olmasi, ilk turdaki secimlerin tutarli oldugunu gosteriyor

## 3. Validation Tarafinda En Iyi 5 Kosu

1. [run005_baseline_lr0001_do03_ls005_cosine_min1em05_augnone_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run005_baseline_lr0001_do03_ls005_cosine_min1em05_augnone_samplernone_weightnone)
   validation `Macro F1 = 0.931182`
   test `Macro F1 = 0.925141`

2. [run003_baseline_lr0001_do02_ls01_cosine_min1em06_auglight_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run003_baseline_lr0001_do02_ls01_cosine_min1em06_auglight_samplernone_weightnone)
   validation `Macro F1 = 0.930532`
   test `Macro F1 = 0.930148`

3. [run021_baseline_lr00003_do03_ls005_cosine_min1em05_auglight_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run021_baseline_lr00003_do03_ls005_cosine_min1em05_auglight_samplernone_weightnone)
   validation `Macro F1 = 0.923312`
   test `Macro F1 = 0.930090`

4. [run004_baseline_lr0001_do03_ls0_cosine_min1em06_auglight_samplerweighted_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run004_baseline_lr0001_do03_ls0_cosine_min1em06_auglight_samplerweighted_weightnone)
   validation `Macro F1 = 0.922376`
   test `Macro F1 = 0.931321`

5. [run001_baseline_lr0001_do02_ls0_cosine_min1em06_augnone_samplernone_weightbalanced](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run001_baseline_lr0001_do02_ls0_cosine_min1em06_augnone_samplernone_weightbalanced)
   validation `Macro F1 = 0.921244`
   test `Macro F1 = 0.930715`

## 4. Test Tarafinda En Iyi 5 Kosu

1. [run004_baseline_lr0001_do03_ls0_cosine_min1em06_auglight_samplerweighted_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run004_baseline_lr0001_do03_ls0_cosine_min1em06_auglight_samplerweighted_weightnone)
   test `Macro F1 = 0.931321`

2. [run019_baseline_lr00003_do02_ls01_cosine_min1em05_augnone_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run019_baseline_lr00003_do02_ls01_cosine_min1em05_augnone_samplernone_weightnone)
   test `Macro F1 = 0.930895`

3. [run010_baseline_lr00005_do02_ls005_cosine_min1em06_auglight_samplernone_weightbalanced](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run010_baseline_lr00005_do02_ls005_cosine_min1em06_auglight_samplernone_weightbalanced)
   test `Macro F1 = 0.930757`

4. [run001_baseline_lr0001_do02_ls0_cosine_min1em06_augnone_samplernone_weightbalanced](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run001_baseline_lr0001_do02_ls0_cosine_min1em06_augnone_samplernone_weightbalanced)
   test `Macro F1 = 0.930715`

5. [run003_baseline_lr0001_do02_ls01_cosine_min1em06_auglight_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run003_baseline_lr0001_do02_ls01_cosine_min1em06_auglight_samplernone_weightnone)
   test `Macro F1 = 0.930148`

## 5. Iletim Sinifi Tarafinda En Iyi 5 Kosu

1. [run019_baseline_lr00003_do02_ls01_cosine_min1em05_augnone_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run019_baseline_lr00003_do02_ls01_cosine_min1em05_augnone_samplernone_weightnone)
   test `iletim F1 = 0.891365`

2. [run011_baseline_lr00005_do02_ls01_cosine_min1em06_auglight_samplerweighted_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run011_baseline_lr00005_do02_ls01_cosine_min1em06_auglight_samplerweighted_weightnone)
   test `iletim F1 = 0.890080`

3. [run001_baseline_lr0001_do02_ls0_cosine_min1em06_augnone_samplernone_weightbalanced](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run001_baseline_lr0001_do02_ls0_cosine_min1em06_augnone_samplernone_weightbalanced)
   test `iletim F1 = 0.888889`

4. [run009_baseline_lr00005_do02_ls0_cosine_min1em06_augnone_samplerweighted_weightbalanced](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run009_baseline_lr00005_do02_ls0_cosine_min1em06_augnone_samplerweighted_weightbalanced)
   test `iletim F1 = 0.885942`

5. [run004_baseline_lr0001_do03_ls0_cosine_min1em06_auglight_samplerweighted_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run004_baseline_lr0001_do03_ls0_cosine_min1em06_auglight_samplerweighted_weightnone)
   test `iletim F1 = 0.885870`

## 6. Gozlem

Ilk 28 kosuda dikkat ceken nokta:

- tum lider kosular `baseline` mimarisi altindan geldi
- `lr = 1e-3` ve `lr = 3e-4` bandi guclu gorunuyor
- `class_weight_mode = none` bircok ust kosuda iyi sonuc verdi
- `augment = light` bazi kosularda faydali oldu
- `label_smoothing = 0.05` ve `0.1` bazi kosularda olumlu katkı verdi

Bu nedenle ilk 28 kosu, onceki sweep liderlerinden farkli olarak daha guclu bir `baseline` bolgesi olduguna isaret ediyor.

## 7. Gorsel Ciktilar

Her tamamlanan kosuda:

- `training_report.json`
- `run_summary.json`
- `visuals/confusion_matrix_test.png`
- `visuals/gradcam_summary_test.json`

bulunmaktadir.

Ornek lider kosu gorselleri:

- [run004 confusion matrix](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run004_baseline_lr0001_do03_ls0_cosine_min1em06_auglight_samplerweighted_weightnone/visuals/confusion_matrix_test.png)
- [run004 Grad-CAM summary](/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64/run004_baseline_lr0001_do03_ls0_cosine_min1em06_auglight_samplerweighted_weightnone/visuals/gradcam_summary_test.json)

## 8. Ara Karar

Bu sweep yarida durdurulmus olsa da ilk 28 kosu kullanisli bir sinyal verdi.

Su anki ara liderler:

- toplam test `Macro F1` odakli: `run004`
- validation odakli: `run005`
- `iletim F1` odakli: `run019`

Eger devam edilirse sonraki adim mantigi su olabilir:

- kalan kosulari `--skip-completed` ile tamamlamak
- veya ilk 28 kosuda iyi gorunen bolgeye daha dar bir ikinci tur sweep acmak
