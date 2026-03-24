# Run007 FP Fine-Tuning Sweep Raporu

Tarih: 24 Mart 2026

Bu rapor, [run007 final aday modeli](/home/alper/ekg/secili_model_adaylari/run007_referans) uzerinde false positive azaltma hedefiyle kosulan `8` adet fine-tuning denemesini ozetler.

Fine-tuning cikti kok klasoru:

- [uc_sinif_run007_fp_finetune](/home/alper/ekg/artifacts/uc_sinif_run007_fp_finetune)

## 1. Baslangic Referansi

Referans model:

- [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans)

Referans test metrikleri:

- test `Macro F1 = 0.934375`
- test `iletim F1 = 0.911602`

Referans kritik FP hucreleri:

- `normal -> ritim = 81`
- `ritim -> normal = 82`
- toplam = `163`

## 2. Fine-Tuning Turu

Kosulan toplam deneme:

- `8`

Temel strateji:

- `run007` checkpoint’inden resume
- dusuk LR
- kisa ek epoch
- sinirli sayida `augment`, `sampler`, `label_smoothing`, `weight_decay`, `min_lr` varyasyonu

## 3. En Iyi FP Toplami

En dusuk `normal -> ritim + ritim -> normal` toplami:

- [ft04](/home/alper/ekg/artifacts/uc_sinif_run007_fp_finetune/ft04_lr5e5_e8_min1e5_none_sampler_none_weight_none_ls010)

Sonuc:

- `normal -> ritim = 76`
- `ritim -> normal = 79`
- toplam `= 155`

Yani:

- referans `163` iken
- `ft04` bunu `155`’e indirdi

Ama bedeli:

- test `Macro F1 = 0.928625`
- test `iletim F1 = 0.891365`

Bu nedenle:

- FP azaldi
- ama toplam kalite ve `iletim` performansi fazla zarar gordu

## 4. En Dengeli Fine-Tune Adayi

En dengeli aday:

- [ft05](/home/alper/ekg/artifacts/uc_sinif_run007_fp_finetune/ft05_lr1e4_e6_min1e6_light_sampler_none_weight_none_ls005)

Sonuc:

- test `Macro F1 = 0.931857`
- test `iletim F1 = 0.903581`
- `normal -> ritim = 69`
- `ritim -> normal = 92`
- toplam `= 161`

Yorum:

- referansa gore FP toplaminda kucuk iyilesme var
- `iletim F1` hala kabul edilebilir bantta
- ama toplam skor yine referansin altinda

## 5. En Iyi Test Macro F1 Fine-Tune Sonucu

Fine-tuning turundaki en yuksek test `Macro F1`:

- [ft05](/home/alper/ekg/artifacts/uc_sinif_run007_fp_finetune/ft05_lr1e4_e6_min1e6_light_sampler_none_weight_none_ls005)
- test `Macro F1 = 0.931857`

Bu deger:

- referans `run007` test `Macro F1 = 0.934375`

Yani:

- fine-tuning tavan test skorunu asamadi

## 6. Acceptance Criteria Degerlendirmesi

Kriterler:

- test `Macro F1 >= 0.932`
- test `iletim F1 >= 0.900`
- `normal -> ritim + ritim -> normal` toplami `163` altina inmeli
- ideal olarak `<= 150`

Sonuc:

- hicbir kosu ideal bandi net bicimde saglamadi
- `ft04` FP tarafinda iyi ama kaliteyi fazla dusurdu
- `ft05` kaliteyi daha iyi korudu ama kazanc sinirli kaldi

Ara karar:

- bu turda `run007`yi net gecen bir fine-tune kosusu cikmadi

## 7. Teknik Yorum

Bu turun ogrettigi sey:

- `run007` zaten guclu bir optimuma yakin
- dusuk LR ile kisa fine-tuning kolayca ciddi ekstra kazanc getirmiyor
- FP azaltmaya calisirken modelin mevcut ayrim dengesi bozulabiliyor

Ozellikle gorulen patern:

- `normal <-> ritim` gecislerini azaltinca
- bazen genel ayrim keskinligi de azaliyor
- bu da toplam `Macro F1` dusuruyor

## 8. Sonuc

Bu fine-tuning sweep sonunda:

- ana aday model halen [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans)

En dikkate deger fine-tune denemeleri:

- FP dusurme acisindan: [ft04](/home/alper/ekg/artifacts/uc_sinif_run007_fp_finetune/ft04_lr5e5_e8_min1e5_none_sampler_none_weight_none_ls010)
- dengeli yaklasim acisindan: [ft05](/home/alper/ekg/artifacts/uc_sinif_run007_fp_finetune/ft05_lr1e4_e6_min1e6_light_sampler_none_weight_none_ls005)

Ama genel karar:

- referans `run007` korunmali
