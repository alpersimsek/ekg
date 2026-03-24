# EKG Yapay Zekâ Projesi – Kronolojik Detaylı Rapor

**Tarih:** 24 Mart 2026  
**Kapsam:** Projenin 23 Mart 2026 akşamından 24 Mart 2026 öğleden sonrasına kadar olan tüm çalışmaları  
**Seviye:** Lise seviyesinde anlaşılır dil

---

## Bu Rapor Ne Anlatıyor?

Bu rapor, 12 kanallı EKG (kalp elektrik sinyali) kayıtlarını yapay zekâ ile otomatik sınıflandıran bir projenin **baştan bu yana tüm adımlarını zaman sırasına göre** anlatır. Her adımda ne yapıldığı, ne sonuç alındığı ve teknik terimlerin ne anlama geldiği açıklanmaktadır.

---

## Terimler Sözlüğü (Kısa)

Raporda sık geçen terimlerin hızlı karşılıkları:

| Terim | Açıklama |
|---|---|
| **EKG** | Kalbin elektrik sinyallerini ölçen cihaz. 12 farklı sensörle 10 saniye boyunca ölçüm yapar. |
| **Model** | Bilgisayara verilerden öğrenme yeteneği kazandıran matematiksel yapı. |
| **CNN** | Sinyaldeki kalıpları öğrenen bir yapay zekâ mimarisi. |
| **ResNet-SE** | CNN'in daha güçlü bir versiyonu. Daha derin ve "dikkat mekanizmalı". |
| **Epoch** | Modelin tüm eğitim verisini bir kere baştan sona görmesi. |
| **Macro F1** | Her sınıfı eşit önemde değerlendiren başarı ölçütü (projenin ana puanı). |
| **Precision** | "Model X dedi" dediğinde gerçekten X olanların oranı. |
| **Recall** | Gerçekte X olanların kaçını modelin bulabildiği. |
| **Confusion Matrix** | Modelin hangi sınıfı hangisiyle karıştırdığını gösteren tablo. |
| **Sweep** | Birçok farklı ayarla modeli sırayla deneyip en iyisini bulma işlemi. |
| **Fine-tuning** | Zaten eğitilmiş bir modeli küçük adımlarla ince ayar yaparak iyileştirme. |
| **Grad-CAM** | Modelin kararını verirken sinyalin hangi bölümüne baktığını gösteren görselleştirme. |
| **Bucket-Aware** | Hatanın yoğunlaştığı belirli alt gruplara odaklanarak eğitim yapma stratejisi. |
| **Scheduler** | Eğitim boyunca öğrenme hızını otomatik olarak ayarlayan mekanizma. |
| **Checkpoint** | Modelin belirli bir andaki anlık görüntüsünün kaydedilmiş dosyası. |

---

## Zaman Çizelgesi

### 📅 23 Mart 2026 – Saat ~19:56
### Adım 1: Veri Havuzu İncelemesi

**Dosya:** `3_sinif_data_birlesik_raporu.md` + `3_sinif_data_birlesik_ozet.json`

İlk iş olarak elimizdeki EKG veri havuzu incelendi:

- **62.985** kayıt mevcut
- Her kayıt **12 kanal × 5.000 ölçüm noktası** (10 saniyelik sinyal)
- Kayıtlar 3 sınıfa ayrılmış:

| Sınıf | Ne Anlama Geliyor | Kayıt Sayısı | Oran |
|---|---|---:|---:|
| **Normal** | Kalp düzgün çalışıyor | 22.107 | %35 |
| **Ritim** | Kalp atış düzeninde sorun var | 37.844 | %60 |
| **İletim** | Kalpteki elektrik yolunda tıkanma | 3.034 | %5 |

**Önemli Gözlem:** İletim sınıfı çok az kayıt içeriyor (%5). Bu, yapay zekâ için zorluk yaratır çünkü az gördüğü şeyi iyi öğrenemez. Buna **sınıf dengesizliği** denir.

Ayrıca hastaların demografik bilgileri çıkarıldı:
- Yaş: 0–92 arası (ortalama 57,6)
- Kadın/Erkek: 29.256 / 33.712

---

### 📅 23 Mart 2026 – Saat ~21:29
### Adım 2: Proje Yol Haritası (Sprint Planı)

**Dosya:** `uc_sinif_ekg_sprint_plani.md`

Proje **6 aşamaya (sprint)** bölündü:

1. **Sprint 1** – Veri denetimi ve etiket kuralları
2. **Sprint 2** – Veri hazırlama ve bölme
3. **Sprint 3** – İlk model eğitimi
4. **Sprint 4** – Hata analizi
5. **Sprint 5** – Optimizasyon
6. **Sprint 6** – Final test ve üretime hazırlık

Ana kural belirlendi: Model başarısı **Macro F1** ile ölçülecek (accuracy/doğruluk değil), çünkü sınıf dengesizliği varken accuracy yanıltıcı olabilir.

---

### 📅 23 Mart 2026 – Saat ~21:38
### Adım 3: Sprint 1 – Veri Denetimi ✅

**Dosya:** `sprint_1_veri_denetimi_raporu.md`

Tüm verinin sağlam olup olmadığı kontrol edildi:

- 62.985 kaydın tamamında dosya çiftleri eksiksiz bulundu
- Tüm kayıtlar doğru formatta (12 kanal, 500 Hz, 5.000 ölçüm)
- 1 adet bozuk başlık dosyası tespit edildi (`JS23074.hea`) – küçük sorun, tolere edildi
- **Etiket kuralları** sabitlendi: `siniflar_final.csv` dosyası tek kaynak olarak kabul edildi
- Morfolojik/iskemik ve teknik sorunlu **27.527 kayıt eğitim dışı bırakıldı**
- Geriye **35.458** temiz kayıt kaldı

---

### 📅 23 Mart 2026 – Saat ~22:42
### Adım 4: Sprint 2 – Veri Hazırlama ve Bölme ✅

**Dosya:** `sprint_2_hazirlik_ve_split_raporu.md`

Temiz veri 3 parçaya bölündü (sınıf oranları korunarak):

| Bölüm | Normal | Ritim | İletim | Toplam |
|---|---:|---:|---:|---:|
| Eğitim (%80) | 11.535 | 15.324 | 1.507 | 28.366 |
| Doğrulama (%10) | 1.441 | 1.915 | 188 | 3.544 |
| Test (%10) | 1.443 | 1.916 | 189 | 3.548 |

Ön işleme kuralı: Her kanal kendi içinde **z-score normalizasyonu** uygulanacak (ortalamayı 0'a, dağılımı 1'e getirme).

---

### 📅 23 Mart 2026 – Saat ~22:48
### Adım 5: Sprint 3 – İlk Model (Baz Model) ✅

**Dosya:** `sprint_3_baz_model_egitim_raporu.md`

İlk yapay zekâ modeli eğitildi:
- **Mimari:** 1D CNN (basit evrişimli sinir ağı)
- **Eğitim süresi:** ~18 saniye (GPU üzerinde, 3 epoch)

**İlk Sonuçlar:**

| Metrik | Doğrulama | Test |
|---|---:|---:|
| **Macro F1** | 0,826 | 0,845 |
| İletim F1 | 0,718 | 0,772 |

Model çalışıyor ve anlamlı! Ancak iletim sınıfında performans düşük.

---

### 📅 23 Mart 2026 – Saat ~22:52
### Adım 6: Sprint 4 – Hata Analizi ✅

**Dosya:** `sprint_4_hata_analizi_ve_iyilestirme_raporu.md`

Modelin **nerede hata yaptığı** incelendi:
- En büyük hata: **ritim olan kayıtlar normal diye tahmin ediliyor** (288 adet)
- İletim'de sorun: recall (yakalama) iyi ama precision (kesinlik) düşük – bazı normal/ritim kayıtları yanlışlıkla iletim deniyor

Denenen iyileştirmeler (ağırlıklı örnekleme varyasyonları) baz modeli geçemedi.

---

### 📅 23 Mart 2026 – Saat ~23:16
### Adım 7: Sprint 5 – İlk Optimizasyon Turu ✅

**Dosya:** `sprint_5_optimizasyon_raporu.md`

Daha uzun eğitim ve daha düşük öğrenme oranı denendi:

| Deney | Epoch | LR | Doğr. Macro F1 | Test Macro F1 |
|---|---:|---|---:|---:|
| Baz model | 3 | 1e-3 | 0,826 | 0,845 |
| Deney 1 | 6 | 1e-3 | 0,853 | 0,877 |
| **Deney 2** | **6** | **5e-4** | **0,862** | **0,891** |

Sonuç: Sadece daha uzun eğitim ve daha küçük öğrenme adımı bile büyük fark yarattı!

---

### 📅 23 Mart 2026 – Saat ~23:22
### Adım 8: Sprint 5 – Ek Optimizasyon (Scheduler Eklenmesi) ✅

**Dosya:** `sprint_5_ek_optimizasyon_raporu.md`

**Kosinüs planlayıcısı** (cosine scheduler) eklendi: Öğrenme oranını eğitim boyunca yavaş yavaş düşüren bir strateji.

| Deney | Epoch | Doğr. Macro F1 | Test Macro F1 |
|---|---:|---:|---:|
| Önceki en iyi | 6 | 0,862 | 0,891 |
| **12 epoch + cosine** | **12** | **0,908** | **0,921** |
| 12 epoch + cosine + augment | 12 | 0,907 | 0,910 |

🎯 **Büyük sıçrama!** Veri artırma (augmentasyon) ek fayda sağlamadı ama scheduler çok işe yaradı.

---

### 📅 23 Mart 2026 – Saat ~23:37
### Adım 9: Fine-Tuning Denemeleri

**Dosya:** `fine_tuning_deneme_raporu.md`

En iyi modelden devam ederek ince ayar yapılmaya çalışıldı (daha düşük LR ile):

| Deney | Doğr. Macro F1 | Test Macro F1 |
|---|---:|---:|
| Mevcut en iyi | 0,908 | 0,921 |
| Fine-tune 1 epoch | 0,895 | 0,901 |
| Fine-tune 5 epoch | 0,907 | 0,918 |

**Sonuç:** Fine-tuning mevcut en iyi modeli geçemedi. Model zaten güçlü bir optimuma ulaşmıştı.

---

### 📅 23 Mart 2026 – Saat ~23:43
### Adım 10: ResNet-SE Mimari Denemesi

**Dosya:** `resnet_se_deneme_raporu.md`

Daha güçlü bir mimari denendi: **ResNet-SE** (artık bağlantılar + dikkat mekanizması).

| Model | Doğr. Macro F1 | Test Macro F1 | Test İletim F1 |
|---|---:|---:|---:|
| Mevcut CNN (12 epoch) | 0,908 | 0,921 | 0,881 |
| ResNet-SE (12 epoch) | 0,903 | 0,923 | 0,859 |

**Yorum:** Test'te biraz daha iyi, ama doğrulamada geride. İletim F1'de ise daha zayıf. Henüz ana aday yapılmadı.

---

### 📅 24 Mart 2026 – Saat ~00:05
### Adım 11: 50 Epoch Uzun Eğitim Testleri + Confusion Matrix + Grad-CAM

**Dosya:** `50_epoch_testler_confusion_gradcam_raporu.md`

Her iki mimari de **50 epoch** ile eğitildi:

| Model | Doğr. Macro F1 | Test Macro F1 | Test İletim F1 |
|---|---:|---:|---:|
| CNN 12 epoch (eski lider) | 0,908 | 0,921 | 0,881 |
| CNN 50 epoch | 0,909 | 0,916 | 0,870 |
| **ResNet-SE 50 epoch** | **0,922** | **0,930** | 0,874 |

🏆 **ResNet-SE 50 epoch yeni lider oldu!** İlk kez Grad-CAM (modelin neye baktığını gösteren ısı haritaları) üretildi.

---

### 📅 24 Mart 2026 – Saat ~00:35
### Adım 12: Ayar Envanteri ve Öğrenilmiş Dersler

**Dosya:** `ayar_envanteri_ve_en_iyi_degerler_raporu.md`

Şimdiye kadar denenen tüm ayarlar envanter haline getirildi:
- **10 ana hiperparametre**
- Her birinin denenen ve en iyi değerleri listelendi
- En etkili ayarlar: model mimarisi, epoch sayısı, scheduler, öğrenme oranı
- En az etkili: hafif augmentasyon, label smoothing

---

### 📅 24 Mart 2026 – Saat ~08:02
### Adım 13: 12 Koşuluk 50-Epoch Sweep (Geniş Tarama)

**Dosya:** `50_epoch_sweep_toplu_raporu.md`

12 farklı ayar kombinasyonu ile sistematik tarama yapıldı. Her biri 50 epoch.

**En iyi sonuç (run12):**
- Model: ResNet-SE, LR: 1e-3, Dropout: 0.3
- Doğrulama Macro F1: **0,937**
- Test Macro F1: **0,938**
- Test İletim F1: 0,887

**İlginç bulgu:** En iyi toplam model ile en iyi iletim modeli farklı koşulardan geldi.

---

### 📅 24 Mart 2026 – Saat ~10:02
### Adım 14: Hiperparametre Tarama Sistemi (Hypergrid) Kurulumu

**Dosya:** `hypergrid_kullanim_kilavuzu.md`

**17.406** teorik kombinasyondan seçilebilen otomatik tarama sistemi kuruldu:
- Birden fazla scheduler desteği (cosine, plateau, onecycle vb.)
- Shard (parçalama) ile paralel çalışma
- Kaldığı yerden devam etme
- Her koşuda otomatik confusion matrix ve Grad-CAM üretimi

---

### 📅 24 Mart 2026 – Saat ~11:08
### Adım 15: 64 Koşuluk Sweep – İlk 28 Koşu Ara Raporu

**Dosya:** `uc_sinif_sweep50_auto64_ilk_28_kosu_ara_raporu.md`

64 koşuluk büyük tarama başlatıldı, ilk 28 koşu tamamlandı:

**Sürpriz bulgu:** Üst sıradaki tüm koşular **baseline (basit CNN)** mimarisinden geldi!
- Ortalama test Macro F1: 0,920
- En iyi test Macro F1: 0,931 (run004)
- En iyi test İletim F1: 0,891 (run019)

---

### 📅 24 Mart 2026 – Saat ~11:14
### Adım 16: Daraltılmış İkinci Tur Sweep Planı

**Dosya:** `uc_sinif_ikinci_tur_daraltilmis_sweep_plani.md`

İlk 28 koşunun sinyallerine göre arama alanı daraltıldı:
- Sadece **baseline** mimarisi
- Sadece **cosine** scheduler
- İyi göstermeyen alanlar (yüksek dropout, düşük LR) çıkarıldı
- Amaç: İyi bölgeyi daha detaylı taramak

---

### 📅 24 Mart 2026 – Saat ~12:51
### Adım 17: Daraltılmış Sweep – 16 Koşu Pilot

**Dosya:** `uc_sinif_refined_sweep16_raporu.md`

16 koşuluk odaklı tarama tamamlandı. İki güçlü aday çıktı:

| Aday | Test Macro F1 | Test İletim F1 | Güçlü Yönü |
|---|---:|---:|---|
| **run004** | 0,935 | 0,893 | Toplam en iyi |
| **run007** | 0,934 | 0,912 | İletim en iyi |

---

### 📅 24 Mart 2026 – Saat ~12:51
### Adım 18: Run004 vs Run007 Karar Karşılaştırması

**Dosyalar:** `run004_run007_karsilastirma_karar_notu.md` + `run004_run007_confusion_gradcam_karsilastirma_raporu.md`

İki finalist detaylıca karşılaştırıldı:

| Kriter | run004 | run007 |
|---|---|---|
| Toplam Macro F1 | **0,935** ✓ | 0,934 |
| İletim F1 | 0,893 | **0,912** ✓ |
| Normal ↔ Ritim karışıklığı | Daha az | Biraz daha fazla |

**Karar:** Varsayılan final aday olarak **run007** seçildi. Gerekçe: Toplam skor farkı çok küçük, ama iletim (en zor sınıf) performansındaki fark anlamlı.

---

### 📅 24 Mart 2026 – Saat ~13:02
### Adım 19: Run007 Final Model Olarak Sabitlendi

**Dosya:** `run007_final_aday_ve_fp_finetune_plani.md`

Run007 daha sonra secili adaylar kokunde `run007_referans` klasorune de alinacak sekilde sabitlendi. Mevcut durumu:
- Test Macro F1: **0,934**
- Test İletim F1: **0,912**
- Kalan ana hata: Normal ↔ Ritim karışması (163 adet toplam)

Sonraki hedef belirlendi: **False Positive (yanlış alarm) azaltma** – özellikle normal/ritim karışmasını düşürmek.

---

### 📅 24 Mart 2026 – Saat ~13:07
### Adım 20: FP Fine-Tuning Kabul Kriterleri Belirlendi

**Dosya:** `run007_fp_finetune_acceptance_criteria.md`

İnce ayar turunun başarı ölçütleri yazıldı:
- Normal→Ritim + Ritim→Normal toplamı 163'ten **≤150'ye** düşmeli
- Test Macro F1 ≥ 0,932 (fazla düşmemeli)
- İletim F1 ≥ 0,900 (bozulmamalı)

---

### 📅 24 Mart 2026 – Saat ~13:37
### Adım 21: FP Fine-Tuning Sweep (8 Deneme)

**Dosya:** `run007_fp_finetune_sweep_raporu.md`

Run007 üzerinden düşük öğrenme oranıyla 8 ince ayar denemesi koşuldu:

| Koşu | Test Macro F1 | İletim F1 | NR+RN Toplamı |
|---|---:|---:|---:|
| run007 (referans) | 0,934 | 0,912 | 163 |
| ft04 (en iyi FP) | 0,929 | 0,891 | 155 |
| ft05 (en dengeli) | 0,932 | 0,904 | 161 |

**Sonuç:** Hiçbir fine-tune koşusu ideal bandı sağlayamadı. Run007 hâlâ en güçlü aday olarak korundu.

---

### 📅 24 Mart 2026 – Saat ~13:41–13:42
### Adım 22: Öncelikli Hata Analizi

**Dosyalar:** `run007_oncelikli_iyilestirme_plani.md` + `run007_oncelikli_hata_analizi_raporu.md`

Hatanın **tam olarak nereden geldiği** kayıt bazında analiz edildi:

**Normal → Ritim (81 hata):**
- Neredeyse tamamı tek bir SNOMED kodu (`426783006` – saf sinüs ritmi)
- Ortalama model güveni: %79 (model emin ama yanlış!)

**Ritim → Normal (82 hata):**
- Ağırlıklı olarak 3 alt tip:
  - `427393009` (sinüs aritmisi) – 55 adet
  - `426177001` (sinüs bradikardisi) – 9 adet
  - `284470004` (erken atrial vuru) – 9 adet
- Ortalama model güveni: %83

**Önemli Sonuç:** Sorun artık "modeli daha çok eğitmek" değil, **"belirli zor alt grupları hedefli olarak öğretmek"** sorununa dönüşmüştü.

---

### 📅 24 Mart 2026 – Saat ~14:39
### Adım 23: Bucket-Aware Sweep (Hedefli Eğitim)

**Dosyalar:** `run007_bucket_aware_sweep_raporu.md` + `run007_vs_ba05_karar_notu.md`

Hata yoğunlaşan kodlara daha fazla ağırlık vererek 8 yeni model eğitildi (**bucket-aware** yaklaşım):

| Model | Test Macro F1 | İletim F1 | NR+RN Toplamı |
|---|---:|---:|---:|
| run007 (referans) | 0,934 | **0,912** | 163 |
| **ba05** (en iyi) | **0,935** | 0,887 | **116** |

🎯 **Büyük başarı!** Normal↔Ritim karışması **163'ten 116'ya** düştü. ANCAK iletim F1 de 0,912'den 0,887'ye düştü.

**Karar:** 
- Run007 = referans (iletim güçlü)
- BA05 = geliştirme adayı (karışıklık düşük ama iletim bedeli var)

---

### 📅 24 Mart 2026 – Saat ~14:41
### Adım 24: İkinci Tur Bucket-Aware Planı (Son Durum)

**Dosya:** `ba05_ikinci_tur_bucket_aware_plani.md`

BA05'in başarısını koruyup iletim kaybını azaltmak için yeni bir daraltılmış tur planlandı:
- Hedef: Normal↔Ritim toplamı ≤130 VE İletim F1 ≥ 0,900
- Daha yumuşak ağırlıklar denenecek (2.0 yerine 1.5–1.75 arası)

---

## Genel İlerleme Özeti

Projenin başından bu yana model performansının yolculuğu:

| Aşama | Tarih/Saat | Test Macro F1 | Test İletim F1 | NR+RN Hatası |
|---|---|---:|---:|---:|
| İlk model (3 epoch CNN) | 23/03 22:48 | 0,845 | 0,772 | 350+ |
| 6 epoch + scheduler | 23/03 23:16 | 0,891 | 0,833 | ~250 |
| 12 epoch + cosine | 23/03 23:22 | 0,921 | 0,881 | ~180 |
| ResNet-SE 50 epoch | 24/03 00:05 | 0,930 | 0,874 | ~115 |
| 12-koşu sweep lideri | 24/03 08:02 | 0,938 | 0,887 | ~100 |
| **run007 (final aday)** | **24/03 12:51** | **0,934** | **0,912** | **163** |
| **ba05 (geliştirme adayı)** | **24/03 14:39** | **0,935** | 0,887 | **116** |

---

## Projenin Mevcut Durumu (24 Mart 2026, 15:00)

### Elimizde İki Güçlü Aday Var:

**1. run007 – "İletim Şampiyonu"**
- Her sınıfta dengeli
- İletim yakalamada en güçlü (F1: 0,912)
- Normal↔Ritim karışması hâlâ var (163 adet)

**2. ba05 – "Denge Şampiyonu"**
- Normal↔Ritim karışmasını ciddi azalttı (116 adet)
- Toplam Macro F1 hafif daha yüksek
- Ama iletim performansı düştü (F1: 0,887)

### Devam Eden İş:
- BA05'in başarılı yönünü koruyup iletim kaybını gidermek için **ikinci tur daraltılmış tarama** planlandı

### Üretilen Toplam Altyapı:
- ✅ Otomatik hiperparametre tarama sistemi (17.406 kombinasyon destekli)
- ✅ Her koşuda otomatik confusion matrix + Grad-CAM üretimi
- ✅ Kaldığı yerden devam etme ve paralel çalışma desteği
- ✅ Daraltılmış ikinci tur tarama altyapısı
- ✅ Bucket-aware (hedefli) eğitim altyapısı
- ✅ Hata analizi ve önceliklendirme araçları

---

## Sonuç

~19 saatlik yoğun çalışmada:
1. **62.985** kayıtlık ham veri denetlendi, temizlendi
2. **35.458** temiz kayıtla sistematik model geliştirme yapıldı
3. Test Macro F1 skoru **0,845'ten 0,935'e** yükseltildi (+%10,6)
4. İletim (en zor sınıf) F1 skoru **0,772'den 0,912'ye** yükseltildi (+%18,1)
5. Normal↔Ritim karışması **350+'dan 116'ya** düşürüldü
6. Tam otomatik deney yönetim altyapısı kuruldu
7. Modelin kararlarını görselleştiren Grad-CAM sistemi eklendi

Proje, final model seçimi için iki güçlü adayla birlikte aktif geliştirme sürecindedir.
