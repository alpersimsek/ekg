# Run007 Final Aday ve FP Fine-Tuning Plani

Tarih: 24 Mart 2026

## 1. Final Adayin Sabitlenmesi

Final adaya ayrilan klasor:

- [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans)

Bu klasore birlikte alinan dosyalar:

- [best_model.pt](/home/alper/ekg/secili_model_adaylari/run007_referans/best_model.pt)
- [training_report.json](/home/alper/ekg/secili_model_adaylari/run007_referans/training_report.json)
- [run_summary.json](/home/alper/ekg/secili_model_adaylari/run007_referans/run_summary.json)
- [confusion_matrix_test.png](/home/alper/ekg/secili_model_adaylari/run007_referans/visuals/confusion_matrix_test.png)
- [gradcam_summary_test.json](/home/alper/ekg/secili_model_adaylari/run007_referans/visuals/gradcam_summary_test.json)

Bu aday modelin mevcut test ozeti:

- test `Macro F1 = 0.934375`
- test precision:
  - `normal = 0.935862`
  - `ritim = 0.951169`
  - `iletim = 0.953757`
- test recall:
  - `normal = 0.940402`
  - `ritim = 0.955637`
  - `iletim = 0.873016`
- test `iletim F1 = 0.911602`

## 2. Neden FP Odakli Fine-Tuning?

Run007 guclu bir aday olsa da confusion matrix sunu gosteriyor:

- `normal -> ritim = 81`
- `ritim -> normal = 82`
- `normal -> iletim = 5`
- `ritim -> iletim = 3`

Buradan cikan yorum:

- model `iletim` yakalamada guclu
- ama `normal` ve `ritim` arasinda gereksiz gecisler hala var
- yani siradaki mantikli adim, “daha agresif recall” yerine “daha temiz precision / daha az false positive” uzerine gitmek

## 3. Fine-Tuning Hedefi

Ana hedef:

- false positive sayisini azaltmak

Ozellikle odaklanilacak hatalar:

- `normal -> ritim`
- `ritim -> normal`
- ikinci oncelik olarak:
  - `normal -> iletim`
  - `ritim -> iletim`

Basari tanimi:

- test `Macro F1` korunurken veya hafif artarken
- `normal` ve `ritim` precision tarafinda iyilesme gormek
- `iletim F1`’yi ciddi bozmamak

## 4. Planlanan Fine-Tuning Stratejisi

Bu tur sifirdan egitim degil, secilmis checkpoint uzerinde devam egitimidir.

Temel prensipler:

- kisa ek epoch
- daha dusuk LR
- daha yumusak ince ayar
- mevcut optimumu bozmayacak kadar kontrollu hareket

Denenecek ana eksenler:

- `resume checkpoint = run007`
- `lr = 1e-4` veya `5e-5`
- `epochs = 6` veya `8`
- `label_smoothing = 0.05` veya `0.1`
- `min_lr = 1e-6` veya `1e-5`
- `weight_decay = 1e-4` veya `5e-4`
- `augment = none` agirlikli
- `sampler = none` agirlikli
- karsilastirma icin sinirli sayida `weighted sampler`

## 5. Neden Bu Parametreler?

`lr` dusurme:

- buyuk adimlar yerine mevcut optimum etrafinda ince ayar yapar

`label_smoothing`:

- asiri emin yanlis tahminleri yumusatabilir
- false positive baskisini azaltmaya yardim edebilir

`weight_decay`:

- modeli daha konservatif hale getirebilir
- gereksiz ayri sinif patlamalarini azaltabilir

`sampler`:

- `none` daha konservatif davranis verebilir
- `weighted` ise `iletim` performansini koruma acisindan kontrol kosulu olarak tutulur

## 6. Ayrı Fine-Tuning Sweep Scripti

Bu plan icin yazilan script:

- [run_uc_sinif_run007_fp_finetune_sweep.py](/home/alper/ekg/scripts/run_uc_sinif_run007_fp_finetune_sweep.py)

Bu script:

- mevcut `train_uc_sinif_baseline.py` trainer’ini kullanir
- `run007` checkpoint’inden resume eder
- her kosu sonunda confusion matrix ve Grad-CAM uretir
- toplam `8` adet kucuk fine-tune kosusu tanimlar

Plan dosyasi ve cikti kok klasoru:

- plan/sweep kok: [uc_sinif_run007_fp_finetune](/home/alper/ekg/artifacts/uc_sinif_run007_fp_finetune)

## 7. Kosu Aileleri

Bu fine-tune turunda kabaca dort grup vardir:

- dusuk LR + sampler yok + smoothing varyasyonu
- dusuk LR + daha uzun epoch + conservative ayar
- az miktarda augment ile dayanıklılık kontrolu
- az sayida `balanced` veya `weighted` kontrol kosusu

Ama ana omurga:

- `sampler = none`
- `label_smoothing = 0.05 / 0.1`
- `lr = 1e-4 / 5e-5`

## 8. Beklenen Karar Mekanizmasi

Bu tur sonunda su tabloya bakilacak:

- test `Macro F1`
- test precision:
  - `normal`
  - `ritim`
  - `iletim`
- test confusion matrix
- `normal -> ritim` sayisi
- `ritim -> normal` sayisi
- `iletim F1`

Karar mantigi:

- eger FP’ler azalip toplam skor korunursa yeni aday secilir
- eger FP azalirken `iletim` ciddi zarar gorurse run007 korunur

## 9. Calistirma Komutu

Bu turu baslatmak icin:

```bash
.venv/bin/python scripts/run_uc_sinif_run007_fp_finetune_sweep.py
```

## 10. Beklenen Sonuc

Bu turdan beklenen:

- buyuk bir skor sıçramasi degil
- daha temiz bir confusion matrix
- daha konservatif ve daha kullanisli bir final aday
