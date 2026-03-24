# EKG Projesi – Kanıt Bazlı "Gerçekten Ne Yaptık" Raporu

**Tarih:** 24 Mart 2026  
**Yöntem:** Bu rapor dosya sistemi taraması, artifact doğrulaması, kod incelemesi ve rapor karşılaştırması ile hazırlanmıştır. Her madde için kanıt dosyası verilmiştir.

---

## 1. Veri Tarafında Gerçekten Yapılanlar

### 1.1 Veri Kaynağı

**✅ Kesin yapıldı.**

- Kaynak: `/home/alper/ekg/data/data_3_Sinif`
- Dosya sistemi doğrulaması: **62.985 `.hea`** + **62.985 `.mat`** dosyası mevcut
- WFDB formatında 12 kanallı, 500 Hz, 10 saniyelik EKG kayıtları
- Kanıt: `hazirlik_ozeti.json` → `toplam_kayit: 62985`, `hea_sayisi: 62985`, `mat_sayisi: 62985`

### 1.2 Format Kontrolü

**✅ Kesin yapıldı.**

- Tüm kayıtlarda `12 kanal / 500 Hz / 5000 örnek` formatı doğrulandı
  - Kanıt: `hazirlik_ozeti.json` → `header_format_dagilimi: {"12_500_5000": 62985}`
- Eksik `.mat`: 0, Sıfır byte dosya: 0
- 1 adet bozuk header tespit edildi: `JS23074` → toleranslı parser ile ele alındı
  - Kanıt: `hazirlik_ozeti.json` → `malformed_header_sayisi: 1`, `malformed_header_ornekleri: ["JS23074"]`
- 256 örneklem üzerinde tensor şekli `(12, 5000)` doğrulandı
  - Kanıt: `sprint_2_hazirlik_ve_split_raporu.md` satır 132–136

### 1.3 Etiket Eşleme

**✅ Kesin yapıldı.**

- Etiket kaynağı: `/home/alper/ekg/yedek/20260323_182919/yeni_yaklasım/siniflar_final.csv` (dosya mevcut, 8609 byte)
- 116 SNOMED kodu → `normal / ritim / iletim / morfolojik_iskemik / exclude` eşlendi
- Eşlenmeyen tıbbi kod: 0
  - Kanıt: `sprint_1_veri_denetimi_raporu.md` satır 161–165

### 1.4 Sınıf Dağılımı

**✅ Kesin yapıldı.**

- Temiz 3 sınıflı kohort: **35.458** kayıt (manifest dosyasında 35.459 satır = 35.458 veri + 1 header)
- Dışlanan: 27.523 morfolojik_iskemik + 4 exclude
- Kanıt: `manifest_uc_sinif_temiz.csv` (7.3 MB, 35.459 satır)

| Sınıf | Kayıt | Oran |
|---|---:|---:|
| Normal | 14.419 | %40,67 |
| Ritim | 19.155 | %54,02 |
| İletim | 1.884 | %5,31 |

- Kanıt: `hazirlik_ozeti.json` → `sinif_sayilari`

### 1.5 Split Yöntemi

**✅ Kesin yapıldı.**

- **Stratified split**, seed=42 ile deterministik
- Oranlar: %80 eğitim / %10 doğrulama / %10 test
- Kanıt: `hazirlik_ozeti.json`, `manifest_uc_sinif_temiz.csv` (bolum sütunu mevcut)
- **⚠ Hasta bazlı split yapılmadı** — veri setinde hasta ID bulunmuyor, kayıt bazlı yapıldı
  - Kanıt: `sprint_2_hazirlik_ve_split_raporu.md` satır 143–153: "Bu veri kümesinde açık bir `patient_id` alanı bulunmadığı için split şu anda kayıt-bazlı stratified olarak üretilmiştir."

### 1.6 Ön İşleme Adımları

**✅ Gerçekten uygulanan:**
- Lead bazlı z-score normalizasyon (her kanal kendi ortalaması ve standart sapması ile)
- Kanıt: `sprint_2_hazirlik_ve_split_raporu.md` satır 121: `lead_bazli_zscore`, `base_training_report.json` içinde kullanılan tensor formatı

**✅ Bazı koşularda uygulanan:**
- Hafif augmentasyon (`augment=light`): sinyal üzerinde Gaussian gürültü ve zaman öteleme
- Kanıt: `base_training_report.json` → `"augment": "light"`

**❌ Uygulanmayan:**
- Bant geçiren filtre → planlanan ama uygulanmadı ("deneysel olarak fayda sağlıyorsa" şartıyla)
- Baz çizgisi düzeltme → planlandı ama hiçbir raporda uygulandığına dair kanıt yok

### 1.7 Veri Hazırlık Betiği

- Script: `scripts/prepare_uc_sinif_dataset.py` (13.100 byte, mevcut)
- Üretilen dosyalar:
  - `artifacts/uc_sinif/manifest_tum_kayitlar.csv` (13,7 MB, mevcut)
  - `artifacts/uc_sinif/manifest_uc_sinif_temiz.csv` (7,3 MB, mevcut)
  - `artifacts/uc_sinif/hazirlik_ozeti.json` (mevcut)

---

## 2. Model Tarafında Gerçekten Denenenler

### 2.1 Baseline CNN

**✅ Kesin uygulandı ve tüm proje boyunca kullanıldı.**

- 1D CNN: conv stem → iki aşamalı kanal genişletme → global average pooling → FC head
- Kod: `scripts/train_uc_sinif_baseline.py` satır ~100+ (BaselineECGNet sınıfı)
- **Final model bu mimariye dayanmaktadır.**
- Kanıt: 83 adet `training_report.json` ve `best_model.pt` dosyası mevcut

### 2.2 ResNet-SE (Squeeze-and-Excitation)

**✅ Kesin uygulandı, denendi, ancak final modelde kullanılmadı.**

- Gerçek 1D residual bloklar + SE (kanal dikkat) mekanizması
- Kod: `scripts/train_uc_sinif_baseline.py` satır 158–261
  - `SEBlock1D` sınıfı (squeeze-excitation)
  - `ResNetSEBlock1D` sınıfı (artık bağlantılı blok)
  - `ResNetSEECGNet` sınıfı (tam model)
- Eğitildi ve artifact üretildi:
  - `artifacts/uc_sinif_test50_resnet_se/` (mevcut, best_model.pt + training_report.json)
  - 12-koşu sweep'te birçok ResNet-SE koşusu yapıldı
- **En iyi sweep sonucu (run12):** Test Macro F1 = 0.938
- **Neden final olmadı:** İkinci tur daraltılmış sweep'te baseline daha iyi sonuç verdi; run007 ve rba02 gibi finalistler baseline mimarisinde

### 2.3 İletim Binary Classifier (İki Aşamalı Sistem)

**✅ Kesin uygulandı ve final modelin parçası.**

- `iletim vs diğer` binary classifier olarak eğitildi
- Kod: `scripts/train_iletim_aux_binary.py` (11.919 byte, mevcut)
- Model dosyası: `final_model_v3/aux_binary/best_model.pt` (1,5 MB, mevcut)
- Eğitim raporu: `aux_binary/training_report.json` → 12 epoch, test accuracy = 0.9873
- İki eşikli override mekanizması ile ana modelle birleştirildi
  - Kanıt: `v3_two_stage_report.json` → `positive_threshold: 0.83`, `negative_threshold: 0.21`

### 2.4 CBAM

**❌ Uygulanmadı.**

- Dosya sistemi grep sonucu: `CBAM` veya `ChannelAttention/SpatialAttention` ifadesi hiçbir script'te bulunamadı
- Sadece SE (Squeeze-and-Excitation) blok var, CBAM yok

### 2.5 Transformer / Attention Head

**❌ Uygulanmadı.**

- `ayar_envanteri_ve_en_iyi_degerler_raporu.md` içinde "temporal attention veya transformer-lite başlık" **öneri** olarak geçiyor
- Kodda transformer veya multi-head attention implementasyonu yok

### 2.6 Fusion / Ensemble

**✅ Denendi ama final modelde kullanılmadı.**

- `finalist_v4_fusion_aday/` klasörü mevcut
- `scripts/run_finalist_v4_fusion_sweep.py` (7.192 byte, mevcut)
- `FINAL_DURUM_RAPORU.md`: "fusion adayi finalist_v4'u geçti" → yani iki aşamalı model fusion'dan daha iyi çıktı

### 2.7 Denenip Bırakılan Model Varyantları

| Varyant | Durum | Kanıt |
|---|---|---|
| `finalist_v4_fusion_aday` | Denendi, geçildi | Klasör mevcut |
| `finalist_v5_binary_hardmine` | Denendi, geçildi | Klasör mevcut |
| `finalist_v6_binary_resnet` | Denendi, geçildi | Klasör mevcut |

---

## 3. Eğitim ve Hiperparametre Tarafında Gerçekten Yapılanlar

### 3.1 Denenen Değerler (Kanıtlı)

Aşağıdaki tablo, `training_report.json` dosyalarından doğrulanan gerçek değerlerdir:

| Parametre | Denenen Değerler | Kanıt |
|---|---|---|
| **Batch size** | `64` (tüm koşularda sabit) | Tüm training_report.json dosyaları |
| **Learning rate** | `2e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3` | training_report.json dosyaları |
| **Epoch** | `1, 3, 5, 6, 8, 12, 50` | training_report.json dosyaları |
| **Dropout** | `0.2, 0.3` (kesin), `0.4` (sweep'te) | training_report.json dosyaları |
| **Class weight** | `balanced, none` | training_report.json → `class_weight_mode` |
| **Scheduler** | `none, cosine` (kesin kullanılan) | training_report.json → `scheduler` |
| **Augmentation** | `none, light` | training_report.json → `augment` |
| **Label smoothing** | `0.0, 0.05, 0.1` | training_report.json → `label_smoothing` |
| **Weight decay** | `1e-4, 5e-5, 5e-4` | training_report.json → `weight_decay` |
| **Sampler** | `none, weighted` | training_report.json → `sampler` |
| **Min LR** | `None, 1e-6, 1e-5` | training_report.json → `min_lr` |
| **Model** | `baseline, resnet_se` | training_report.json → `model` |

### 3.2 Scheduler Detayı

**Gerçekten kullanılanlar:**
- `none` (Sprint 3, ilk denemeler)
- `cosine` (Sprint 5 ve sonrası, tüm güçlü modeller)

**Hypergrid scriptinde tanımlı ama koşulduğuna dair artifact kanıtı bulamadığım:**
- `plateau`, `onecycle`, `cosine_warm_restarts`, `step`, `multistep`
- Bu scheduler'lar `train_uc_sinif_hparam.py` kodunda tanımlı ama `artifacts/uc_sinif_hypergrid/` altında tamamlanmış koşu yok
- Kanıt: `hypergrid_kullanim_kilavuzu.md` bunları desteklenen olarak listeler, ama koşu raporu yoktur

### 3.3 Early Stopping

**❌ Yapısal early stopping kullanılmadı.**

- `train_uc_sinif_hparam.py` kodunda `early` kelimesi geçiyor (muhtemelen patience parametresi)
- Ancak `train_uc_sinif_baseline.py` ve `train_uc_sinif_bucket_aware.py` dosyalarında early stopping implementasyonu yok
- Tüm training_report.json dosyalarında `epochs` kadar epoch koşulmuş, erken durma uygulanmamış
- Bunun yerine **en iyi epoch checkpoint'i kaydedilmiş** (best_epoch seçimi)
- Kanıt: `base_training_report.json` → `epochs: 8`, `best_epoch: 7` (tüm 8 epoch koşuldu)

### 3.4 Fine-Tuning

**✅ Kesin uygulandı.**

- Resume checkpoint desteği `train_uc_sinif_baseline.py`'ye eklendi
- Birden fazla fine-tuning turu yapıldı:
  1. `uc_sinif_opt_exp3_e12_cosine` üzerinden 1 ve 5 epoch fine-tune → mevcut modeli geçemedi
  2. `rba02_final_aday` üzerinden 8 epoch fine-tune → **ft06 final model** olarak seçildi
  3. `run007` üzerinden FP-odaklı 8 deneme → run007'yi geçemedi
- Kanıt: `fine_tuning_deneme_raporu.md`, `run007_fp_finetune_sweep_raporu.md`, `base_training_report.json` → `resume_checkpoint` alanı dolu

### 3.5 Toplam Koşu Sayısı

- **83 adet best_model.pt** dosyası mevcut (dosya sistemi doğrulaması)
- **83 adet training_report.json** dosyası mevcut
- **83 adet confusion_matrix_test.png** dosyası mevcut
- **83 adet gradcam_summary_test.json** dosyası mevcut
- Tüm koşular GPU üzerinde (NVIDIA RTX 3090) tamamlanmıştır

### 3.6 Bucket-Aware Eğitim

**✅ Kesin uygulandı.**

- Özel script: `scripts/train_uc_sinif_bucket_aware.py` (13.154 byte)
- `focus_normal_codes` ve `focus_ritim_codes` ile SNOMED kodu bazlı ağırlıklı örnekleme
- Denenen ağırlıklar: focus_normal_weight: 1.5–2.0, focus_ritim_weight: 1.5–3.0
- Kanıt: `base_training_report.json` → `focus_normal_weight: 1.5, focus_ritim_weight: 2.0`

### 3.7 Temperature Scaling (Kalibrasyon)

**✅ Kesin uygulandı.**

- Script: `scripts/calibrate_uc_sinif_temperature.py` (7.684 byte, mevcut)
- Sonuç: temperature = 0.881767, test ECE: 0.0217 → 0.0107
- Kanıt: `FINALIST_V2_DURUM.md` satır 36–40
- Tahmin sınıflarını değiştirmedi, sadece olasılık güvenini iyileştirdi

### 3.8 Sınıf-Özel Kalibrasyon

**✅ Denendi ama bırakıldı.**

- Script: `scripts/calibrate_uc_sinif_iletim_class_specific.py` (9.250 byte)
- 275 kombinasyon tarandı, test'te kazanç elde edilemedi
- Kanıt: `FINALIST_V2_DURUM.md` satır 104–135

---

## 4. Sonuçlar

### 4.1 Sprint Bazlı Metrik Gelişimi

| Aşama | Test Macro F1 | Test İletim F1 | NR+RN |
|---|---:|---:|---:|
| Sprint 3 – Baz CNN 3 epoch | 0.845 | 0.772 | ~350 |
| Sprint 5 – 6 epoch, lr=5e-4 | 0.891 | 0.833 | ~220 |
| Sprint 5 ek – 12 epoch + cosine | 0.921 | 0.881 | ~180 |
| 50ep ResNet-SE | 0.930 | 0.874 | ~115 |
| Sweep run12 ResNet-SE | 0.938 | 0.887 | ~100 |
| Refined run007 (iletim şampiyonu) | 0.934 | 0.912 | 163 |
| RBA02 (bucket-aware dengeli) | 0.938 | 0.906 | 137 |
| **ft06 baz model** | **0.938** | 0.905 | **131** |
| **Final V3 (iki aşamalı)** | **0.940** | **0.909** | 131 |

### 4.2 Final Test Macro F1

- **Tek model (ft06):** 0.9383
- **İki aşamalı (V3):** 0.9396

### 4.3 Sınıf Bazlı F1 (Final V3 – Test)

| Sınıf | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0.9488 | 0.9494 | 0.9491 |
| Ritim | 0.9599 | 0.9624 | 0.9612 |
| İletim | 0.9235 | 0.8942 | 0.9086 |

### 4.4 Confusion Matrix Ana Hataları (Final V3)

```
              Tahmin
           normal  ritim  iletim
Gerçek
normal      1370     66       7
ritim         65   1844       7
iletim         9     11     169
```

- **En büyük hata:** Normal ↔ Ritim karışması (131 adet, tüm hataların %79'u)
- **En güçlü sınıf:** Ritim (F1: 0.961)
- **En zayıf sınıf:** İletim (F1: 0.909) — azınlık sınıf olmasına rağmen güçlü performans

---

## 5. Açıklanabilirlik (Explainability)

### 5.1 Grad-CAM

**✅ Kesin uygulandı. Kapsamlı biçimde.**

- Script: `scripts/render_confusion_and_gradcam.py` (9.001 byte)
  - Kod incelemesi: `compute_gradcam()` fonksiyonu, PyTorch hook ile son konvolüsyon katmanının aktivasyonlarını alıyor
- **612 adet Grad-CAM PNG görüntüsü** üretildi (dosya sistemi doğrulaması)
- **83 adet `gradcam_summary_test.json`** dosyası mevcut (her koşu için)
- Final modelde hem ana model hem yardımcı binary model için ayrı Grad-CAM üretildi:
  - `final_model_v3/visuals/base_gradcam_*.png` (6 adet)
  - `final_model_v3/visuals/aux_gradcam_*.png` (6 adet)
- Her koşuda 6 örnek seçildi: 3 doğru tahmin (her sınıftan 1) + 3 hata örneği
- Kanıt: `50_epoch_testler_confusion_gradcam_raporu.md`, `render_confusion_and_gradcam.py` satır 47–172

### 5.2 Saliency Map

**❌ Ayrı bir saliency map yöntemi uygulanmadı.**

- Kodda `saliency` kelimesi geçiyor ama bu Grad-CAM implementasyonu kapsamında (`attention_map` aramalarının da kaynak kodu Grad-CAM ile ilgili)
- Vanilla gradient saliency, SmoothGrad veya Integrated Gradients gibi ayrı yöntemler uygulanmadı

### 5.3 SHAP / LIME

**❌ Uygulanmadı.**

- Kodda `shap` veya `lime` referansı bulunamadı

### 5.4 Benzer Örnek Analizi

**❌ Uygulanmadı.**

- Nearest neighbor, embedding similarity veya prototype-based açıklama yöntemi kullanılmadı

### 5.5 Hata Alt Grup Analizi

**✅ Kesin uygulandı.**

- Script: `scripts/analyze_run007_priority_errors.py` (13.876 byte)
- Çıktı: `artifacts/run007_priority_error_analysis/` (CSV dosyaları mevcut)
- İçerik: Kayıt bazlı hata listesi, SNOMED kodu dağılımı, model güven skoru analizi
- `iletim` odaklı analiz: `scripts/analyze_iletim_focus.py` (8.437 byte)
- Kanıt: `run007_oncelikli_hata_analizi_raporu.md`

---

## 6. Kesinlik Sınıflandırması

### ✅ Kesin Yaptıklarımız (Artifact ile doğrulanmış)

1. **Veri denetimi:** 62.985 kayıdın format, bütünlük ve etiket kontrolü → `hazirlik_ozeti.json`
2. **Etiket eşleme:** `siniflar_final.csv` ile 116 SNOMED kodu eşlendi → `sprint_1_veri_denetimi_raporu.md`
3. **Temiz kohort çıkarma:** 35.458 kayıtlık 3 sınıflı alt küme → `manifest_uc_sinif_temiz.csv`
4. **Stratified split:** seed=42, %80/%10/%10 → `manifest_uc_sinif_temiz.csv`
5. **Lead bazlı z-score normalizasyon** → Tüm training scriptlerinde mevcut
6. **1D CNN (baseline) eğitimi:** Çok sayıda epoch/LR/scheduler varyasyonu → 83 training_report.json
7. **ResNet-SE eğitimi:** SE bloklu artık ağ → `train_uc_sinif_baseline.py` satır 158–261, artifact'lar mevcut
8. **Cosine scheduler:** Sprint 5'ten itibaren standart → training_report.json'larda doğrulanıyor
9. **Weighted sampler (genel):** Birçok koşuda kullanıldı → rapor ve json'larda doğrulanıyor
10. **Bucket-aware sampling:** SNOMED kodu bazlı ağırlıklı örnekleme → `train_uc_sinif_bucket_aware.py`
11. **Fine-tuning (resume checkpoint):** Birden fazla turda uygulandı → `base_training_report.json`
12. **İki aşamalı classifier:** Ana model + binary yardımcı → `final_model_v3/aux_binary/`, `v3_two_stage_report.json`
13. **Confusion matrix üretimi:** 83 koşuda → 83 adet `confusion_matrix_test.png`
14. **Grad-CAM üretimi:** 83 koşuda → 612 adet `gradcam_*.png`, 83 `gradcam_summary_test.json`
15. **Temperature scaling:** Uygulandı → `calibration/temperature_calibration.json`
16. **Hata alt grup analizi:** SNOMED kodu ve güven bazlı → `run007_priority_error_analysis/`
17. **Inference scriptleri:** Tek kayıt ve batch tahmin → `predict_final_model_v3.py`, `predict_final_model_v3_batch.py`
18. **Hiperparametre tarama sistemi (Hypergrid):** Script yazıldı ve kısmen koşuldu → `run_uc_sinif_hypergrid.py`

### 🟡 Yüksek Olasılıkla Yaptıklarımız (Kanıt dolaylı veya kısmi)

1. **Hafif augmentasyon (light):** Kodda mevcut, birçok raporda geçiyor, ama tam olarak ne uygulandığı (Gaussian noise miktarı, zaman öteleme aralığı vb.) user facing raporda tam açıklanmamış. `base_training_report.json`'da `augment: light` yazıyor.
2. **Label smoothing:** Bazı koşularda 0.05 ve 0.1 ile doğrulanıyor (ama etkisi raporda "karışık" olarak nitelenmiş).
3. **Hypergrid ile alternatif scheduler denemeleri:** `train_uc_sinif_hparam.py`'de `plateau, onecycle, step, multistep, cosine_warm_restarts` tanımlı. `hypergrid_kullanim_kilavuzu.md`'de destekleniyor deniliyor. Ama `artifacts/uc_sinif_hypergrid/` altında kaç koşunun tamamlandığı kesin doğrulanmadı (karışık durumda olabilir).

### ❌ Sadece Öneri / Plan Olarak Kalanlar

1. **CBAM (Channel + Spatial Attention):** Kodda yok, hiçbir raporda yapıldığına dair kanıt yok
2. **Transformer / multi-head attention:** Öneri olarak `ayar_envanteri_ve_en_iyi_degerler_raporu.md`'de geçiyor ama uygulanmadı
3. **Focal loss / margin-based loss:** Öneri olarak geçiyor, kodda implementasyonu yok
4. **EMA (Exponential Moving Average):** Öneri, uygulanmadı
5. **Gradient clipping:** Öneri, hiçbir raporda uygulandığına dair kanıt yok
6. **Hasta bazlı split:** Veri setinde hasta ID olmadığı için yapılamadı (planlandı ama mümkün değildi)
7. **SHAP / LIME / Integrated Gradients:** Hiç uygulanmadı
8. **Benzer örnek analizi / prototype-based açıklama:** Hiç uygulanmadı
9. **Bant geçiren filtre / baz çizgisi düzeltme:** Planlandı ama uygulanmadı
10. **Hard example mining (aktif):** `run007_oncelikli_iyilestirme_plani.md`'de önerildi ama sistematik uygulama yapılmadı (bucket-aware sampling bunun yumuşak bir versiyonu)
11. **Warmup + cosine scheduler:** Önerildi ama training_report'larda sadece `cosine` var, ayrı warmup aşaması yok
12. **Multi-scale kernel CNN:** Önerildi, uygulanmadı
13. **Lead-wise attention / lead grouping:** Önerildi, uygulanmadı

---

## Dosya Referans Dizini

| Dosya/Klasör | Tür | İçerik |
|---|---|---|
| `scripts/train_uc_sinif_baseline.py` | Kod | Ana eğitim betiği (CNN + ResNet-SE) |
| `scripts/train_uc_sinif_bucket_aware.py` | Kod | Bucket-aware eğitim |
| `scripts/train_uc_sinif_hparam.py` | Kod | Hypergrid eğitim betiği |
| `scripts/train_iletim_aux_binary.py` | Kod | Binary yardımcı classifier |
| `scripts/render_confusion_and_gradcam.py` | Kod | Confusion matrix + Grad-CAM üretici |
| `scripts/analyze_run007_priority_errors.py` | Kod | Hata alt grup analizi |
| `scripts/calibrate_uc_sinif_temperature.py` | Kod | Temperature scaling |
| `scripts/predict_final_model_v3.py` | Kod | Tek kayıt inference |
| `scripts/predict_final_model_v3_batch.py` | Kod | Batch inference |
| `artifacts/uc_sinif/manifest_uc_sinif_temiz.csv` | Veri | 35.458 kayıtlık temiz manifest |
| `artifacts/uc_sinif/hazirlik_ozeti.json` | Veri | Veri denetim özeti |
| `final_model_v3/` | Model | Final iki aşamalı model |
| `final_model_v3/v3_two_stage_report.json` | Rapor | Final test sonuçları |
| `final_model_v3/base_training_report.json` | Rapor | Baz model eğitim logları |
| `dokumanlar/sprint_*.md` | Rapor | Sprint raporları |
| `dokumanlar/rapor/` | Rapor | Özet raporlar |
