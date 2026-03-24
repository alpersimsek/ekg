# Final Model V3 Durum Raporu

Tarih: 24 Mart 2026

Bu klasor, proje boyunca denenmis tum adaylar arasinda aktif calisma modeli olarak secilen `finalist_v3` yapisini toplar.

## Secilen Mimari

Bu model tek bir checkpoint degildir.

Yapi:

- ana model: 3-sinif EKG classifier
- yardimci model: `iletim vs diger` binary classifier
- karar kurali: iki-esikli override mekanizmasi

Dosyalar:

- [base_best_model.pt](/home/alper/ekg/final_model_v3/base_best_model.pt)
- [aux_binary/best_model.pt](/home/alper/ekg/final_model_v3/aux_binary/best_model.pt)

## Secilen Karar Kurali

Secilen esikler:

- `positive_threshold = 0.83`
- `negative_threshold = 0.21`

Kural:

- yardimci model `iletim` olasiligi `>= 0.83` ise nihai tahmin `iletim`
- ana model `iletim` dese bile yardimci model `iletim` olasiligi `< 0.21` ise tahmin `normal/ritim` icinde en guclu sinifa geri cekilir

## Son Test Sonucu

Ana rapor:

- [v3_two_stage_report.json](/home/alper/ekg/final_model_v3/v3_two_stage_report.json)

Test metrikleri:

- test `Macro F1 = 0.939617`
- test accuracy `= 0.953495`
- test `iletim F1 = 0.908602`

Sinif bazli test F1:

- `normal = 0.949082`
- `ritim = 0.961168`
- `iletim = 0.908602`

Confusion matrix:

- `[[1370, 66, 7], [65, 1844, 7], [9, 11, 169]]`

Iletim etkisi:

- baz finalist `iletim FN = 23`
- secilen V3 model `iletim FN = 20`
- baz finalist `iletim FP = 12`
- secilen V3 model `iletim FP = 14`

Yani:

- biraz daha fazla `iletim` false positive uretildi
- ama daha fazla gercek `iletim` yakalandi
- toplam `iletim F1` ve `Macro F1` birlikte iyilesti

## Gorseller

Confusion matrix:

- [confusion_matrix_test.png](/home/alper/ekg/final_model_v3/visuals/confusion_matrix_test.png)

Grad-CAM ozeti:

- [gradcam_summary_test.json](/home/alper/ekg/final_model_v3/visuals/gradcam_summary_test.json)

Ornek gorseller:

- `iletim`e zorlanan ornek:
  [A0330 base](/home/alper/ekg/final_model_v3/visuals/base_gradcam_A0330_normal_to_normal.png)
  [A0330 aux](/home/alper/ekg/final_model_v3/visuals/aux_gradcam_A0330_normal_to_iletim.png)
- `iletim`den geri cekilen ornek:
  [A2606 base](/home/alper/ekg/final_model_v3/visuals/base_gradcam_A2606_iletim_to_iletim.png)
  [A2606 aux](/home/alper/ekg/final_model_v3/visuals/aux_gradcam_A2606_iletim_to_normal.png)

## Neden Bu Modelde Kaldik

Bu model:

- `finalist_v2`yi gecti
- fusion adayi `finalist_v4`u gecti
- hardmine binary adayi `finalist_v5`i gecti
- daha guclu binary mimari `finalist_v6`yi gecti

Yani su anki en iyi pratik denge bu yapida.

## Bundan Sonra

Bu klasor artik ana gelistirme tabanidir.

Bir sonraki mantikli adimlar:

- inference akisini bu iki-asamali yapıya gore sabitlemek
- deployment icin tek girisli ve batch tahmin scriptlerini bu yapıya uyarlamak
- gerekirse bu modeli `servis modeli` olarak versiyonlamak
