# Uc Sinif Ikinci Tur Daraltilmis Sweep Plani

Tarih: 24 Mart 2026

Bu plan, `/home/alper/ekg/artifacts/uc_sinif_sweep50_auto64` altindaki ilk 28 kosunun ara sonucuna gore hazirlanmistir.

## 1. Neden Daraltilmis Ikinci Tur?

Ilk 28 kosudan cikan sinyal:

- ust siradaki tum kosular `baseline` mimarisinden geldi
- en guclu bolge `cosine` scheduler ile calisti
- `lr` olarak en cok `1e-3` ve `3e-4` tekrar etti, `5e-4` de yararli kaldı
- `dropout` tarafinda `0.2` ve `0.3` on plana cikti
- `label_smoothing` `0.0`, `0.05`, `0.1` ucunun de kullanisli oldugu goruldu
- `augment = none` ve `augment = light` ikisi de ust kosularda yer aldi
- `sampler = none` daha sik ust siralarda yer alsa da `weighted` tamamen elenmedi
- `class_weight_mode = none` daha guclu bir sinyal verdi, ama `balanced` tamamen dislanmadi

Bu nedenle ikinci turda:

- mimariyi daraltiyoruz
- iyi gorunmeyen alanlari buyuk olcude disliyoruz
- ama belirsiz kalan parametreleri hala yari rekabetci tutuyoruz

## 2. Rapor Kaynagi

Ara rapor:

- [uc_sinif_sweep50_auto64_ilk_28_kosu_ara_raporu.md](/home/alper/ekg/dokumanlar/uc_sinif_sweep50_auto64_ilk_28_kosu_ara_raporu.md)

## 3. Daraltilmis Parametre Uzayi

Ikinci turda kullanilacak alan:

- `model = ["baseline"]`
- `epochs = [50]`
- `batch_size = [64]`
- `lr = [1e-3, 5e-4, 3e-4]`
- `weight_decay = [1e-4, 5e-4]`
- `dropout = [0.2, 0.3]`
- `label_smoothing = [0.0, 0.05, 0.1]`
- `scheduler = ["cosine"]`
- `min_lr = [1e-6, 1e-5]`
- `augment = ["none", "light"]`
- `sampler = ["none", "weighted"]`
- `class_weight_mode = ["none", "balanced"]`

Bu alanin amaci:

- iyi sinyal gelen bolgede daha yogun arama yapmak
- gereksiz yere `resnet_se` veya zayif scheduler varyantlarini tekrar koşturmamak

## 4. Neleri Bilerek Disari Biraktik?

Ikinci turda su alanlari bilincli olarak disarda tutuyoruz:

- `resnet_se`
  - ilk 28 kosuda ust siralarda yer almadi
- `scheduler = none`
  - su anki en guclu kosular `cosine` altinda geldi
- diger schedulerlar
  - once genel laboratuvar scriptiyle ayri olarak denenebilir
- `dropout = 0.4`
  - ilk 28 kosuda daha zayif sinyal verdi
- `lr = 1e-4`
  - bu bolgede en guclu adaylar arasinda baskin degildi

## 5. Hedef

Ikinci turun hedefi:

- toplam `Macro F1` liderini daha netlestirmek
- `iletim F1` ile toplam skor arasindaki en iyi dengeyi bulmak
- final model secimi icin daha guclu ve daha dar bir karar alani yaratmak

## 6. Yeni Script

Bu tur icin ayri script:

- [run_uc_sinif_refined_sweep.py](/home/alper/ekg/scripts/run_uc_sinif_refined_sweep.py)

Bu script:

- sadece daraltilmis ikinci tur alanini kullanir
- mevcut genel hypergrid sisteminden ayridir
- mevcut calisan temel scripti bozmadan calisir
- her kosu sonunda confusion matrix ve Grad-CAM uretir

## 7. Onerilen Kosu Stratejisi

Once kucuk dry-run:

```bash
.venv/bin/python scripts/run_uc_sinif_refined_sweep.py --dry-run --max-runs 16
```

Sonra pilot tur:

```bash
.venv/bin/python scripts/run_uc_sinif_refined_sweep.py \
  --max-runs 16 \
  --selection-mode even \
  --output-root /home/alper/ekg/artifacts/uc_sinif_refined_sweep16
```

Sonra ana ikinci tur:

```bash
.venv/bin/python scripts/run_uc_sinif_refined_sweep.py \
  --max-runs 48 \
  --selection-mode even \
  --output-root /home/alper/ekg/artifacts/uc_sinif_refined_sweep48
```

Alternatif cesitlilik turu:

```bash
.venv/bin/python scripts/run_uc_sinif_refined_sweep.py \
  --max-runs 48 \
  --selection-mode random \
  --seed 42 \
  --output-root /home/alper/ekg/artifacts/uc_sinif_refined_sweep48_random42
```

## 8. Beklenen Karar Mekanizmasi

Ikinci tur sonunda karar verirken bakilacak seyler:

- validation `Macro F1`
- test `Macro F1`
- test `iletim F1`
- confusion matrix dengesi
- `normal -> iletim` ve `ritim -> iletim` yanlislarinin yogunlugu
- Grad-CAM orneklerinin klinik sezgiyle ne kadar uyumlu oldugu

## 9. Beklenen Cikti

Her kosu icin:

- `best_model.pt`
- `training_report.json`
- `run_summary.json`
- `visuals/confusion_matrix_test.png`
- `visuals/gradcam_summary_test.json`

Tur sonunda:

- `refined_plan.json`
- `refined_summary.json`

## 10. Ara Sonuc Beklentisi

Bu turdan beklenen sey, yeni bir dev kazanc degil; daha cok:

- ilk 28 kosuda gorulen en iyi bolgenin teyidi
- final model secimini daha guvenli hale getirmek
- son karari `run004 / run005 / run019` benzeri tekil adaylar arasinda daraltmak
