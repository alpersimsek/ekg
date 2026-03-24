# 50 Epoch Sweep Toplu Raporu

Tarih: 24 Mart 2026

Bu rapor, `/home/alper/ekg/artifacts/uc_sinif_sweep50` altında koşturulan 12 adet 50-epoch deneyin toplu sonucunu özetler. Her koşu için eğitim tamamlandıktan sonra test split üzerinde confusion matrix ve Grad-CAM çıktıları üretilmiştir.

## 1. Amaç

Amaç, 12 kanallı EKG sinyalinden `normal`, `ritim`, `iletim` sınıflarından birini tahmin eden 3 sınıflı model için genişletilmiş hiperparametre taraması yapmaktı.

Ana optimizasyon metriği:

- `Macro F1`

Ek izlenen metrikler:

- accuracy
- sınıf bazlı precision / recall / F1
- özellikle `iletim` sınıfı F1

## 2. Sweep Kapsamı

Tarama alanı aşağıdaki parametre kümeleri üzerinden tanımlandı:

- `model`: `["baseline", "resnet_se"]`
- `epochs`: `[50]`
- `lr`: `[1e-3, 5e-4, 3e-4, 1e-4]`
- `dropout`: `[0.2, 0.3, 0.4]`
- `label_smoothing`: `[0.0, 0.05, 0.1]`
- `scheduler`: `["cosine"]`
- `min_lr`: `[1e-6, 1e-5]`
- `augment`: `["none", "light"]`
- `sampler`: `["none", "weighted"]`
- `class_weight_mode`: `["balanced", "none"]`

Not:

- Bu alanın tamamı doğrudan tam kartesyen çarpım olarak koşturulmadı.
- Bunun yerine bu parametre havuzundan seçilmiş 12 adet yüksek sinyal değerli koşu planlandı.
- Her koşu tam `50 epoch` sürdü. İyi skor geldi diye hiçbir koşu yarıda kesilmedi.

## 3. Üretilen Çıktılar

Her koşu klasörü altında şu çıktılar üretildi:

- `training_report.json`
- `best_model.pt`
- `visuals/confusion_matrix_test.png`
- `visuals/gradcam_summary_test.json`
- örnek Grad-CAM görselleri

Toplu özet:

- [sweep_summary.json](/home/alper/ekg/artifacts/uc_sinif_sweep50/sweep_summary.json)

Doğrulama:

- tamamlanan koşu sayısı: `12 / 12`
- confusion matrix üretilen koşu sayısı: `12 / 12`
- Grad-CAM summary üretilen koşu sayısı: `12 / 12`

## 4. En İyi Toplam Model

Validation ve test sonuçlarının birlikte en güçlü adayı:

- model yolu: [run12_resnetse_lr1e3_do03_ls000_cosine_min1e6_augnone_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50/run12_resnetse_lr1e3_do03_ls000_cosine_min1e6_augnone_samplernone_weightnone)
- model tipi: `resnet_se`
- epoch: `50`
- best epoch: `44`
- `lr = 1e-3`
- `dropout = 0.3`
- `label_smoothing = 0.0`
- `scheduler = cosine`
- `min_lr = 1e-6`
- `augment = none`
- `sampler = none`
- `class_weight_mode = none`

Skorlar:

- validation `Macro F1 = 0.936866`
- test `Macro F1 = 0.937777`
- validation accuracy `= 0.958239`
- test accuracy `= 0.959977`
- validation `iletim F1 = 0.886486`
- test `iletim F1 = 0.887052`

Görseller:

- confusion matrix: [confusion_matrix_test.png](/home/alper/ekg/artifacts/uc_sinif_sweep50/run12_resnetse_lr1e3_do03_ls000_cosine_min1e6_augnone_samplernone_weightnone/visuals/confusion_matrix_test.png)
- Grad-CAM summary: [gradcam_summary_test.json](/home/alper/ekg/artifacts/uc_sinif_sweep50/run12_resnetse_lr1e3_do03_ls000_cosine_min1e6_augnone_samplernone_weightnone/visuals/gradcam_summary_test.json)

## 5. En İyi Test Macro F1 Sıralaması

İlk 5 koşu:

1. `run12` `resnet_se`, test `Macro F1 = 0.937777`
2. `run08` `resnet_se`, test `Macro F1 = 0.932541`
3. `run11` `resnet_se`, test `Macro F1 = 0.931898`
4. `run01` `baseline`, test `Macro F1 = 0.931823`
5. `run09` `resnet_se`, test `Macro F1 = 0.931464`

Bu tablo, sweep sonunda toplam skor açısından `resnet_se` mimarisinin genel olarak öne çıktığını gösteriyor.

## 6. En İyi Iletim F1 Gözlemi

Toplam lider model `run12` olsa da, `iletim` sınıfında en güçlü koşulardan biri farklı bir ayarda geldi:

- model yolu: [run01_baseline_lr1e3_do02_ls000_cosine_min1e6_augnone_samplernone_weightbalanced](/home/alper/ekg/artifacts/uc_sinif_sweep50/run01_baseline_lr1e3_do02_ls000_cosine_min1e6_augnone_samplernone_weightbalanced)
- test `iletim F1 = 0.893048`

Bu önemli çünkü:

- en iyi toplam model ile en iyi sınıf-özel model aynı olmayabiliyor
- `iletim` azınlık sınıfı olduğu için küçük karar sınırı farkları F1 üzerinde daha belirgin etki yapıyor
- deployment seçiminde hedef yalnız toplam `Macro F1` değilse, sınıf-özel karar stratejisi de değerlendirilebilir

## 7. Dikkat Çeken Teknik Sonuçlar

- `resnet_se` mimarisi sweep sonunda toplam performansta öne geçti.
- `lr = 1e-3` beklenenden daha güçlü çıktı; özellikle `run12` bu ayarda lider oldu.
- `sampler = weighted` her zaman toplam skoru artırmadı.
- `class_weight_mode = none` bazı güçlü koşularda `balanced` seçeneğini geçti.
- `augment = light` bazı koşularda faydalı olsa da genel lider kombinasyonu oluşturmadı.
- `label_smoothing = 0.05` bazı `resnet_se` koşularında iyi sonuç verdi ama en iyi toplam modelde gerekli olmadı.

## 8. Nihai Karar

Bu sweep turu sonunda ana aday model:

- [run12_resnetse_lr1e3_do03_ls000_cosine_min1e6_augnone_samplernone_weightnone](/home/alper/ekg/artifacts/uc_sinif_sweep50/run12_resnetse_lr1e3_do03_ls000_cosine_min1e6_augnone_samplernone_weightnone)

Eğer amaç:

- en yüksek toplam kalite ise `run12`
- `iletim` sınıfını ayrı hassasiyetle optimize etmek ise `run01` de ikinci aday olarak korunmalı

## 9. Sonraki Adımlar

- final inference scriptini lider model için sabitlemek
- test confusion matrix ve Grad-CAM örneklerini sunum formatına çevirmek
- `run12` ve `run01` için sınıf bazlı karşılaştırmalı kısa teknik rapor hazırlamak
- gerekiyorsa `iletim` odaklı calibration veya threshold analizi yapmak
