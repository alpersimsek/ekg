# EKG Projesi – Teknik Metrik Özeti ve Hata Analizi Raporu

**Tarih:** 24 Mart 2026  
**Kapsam:** Tüm sprint raporları, sweep sonuçları, fine-tuning denemeleri ve final model seçimi  
**Amaç:** Macro F1 başta olmak üzere raporlanan metrikleri, elde edilme koşullarını, sınıf karışımlarını ve öğrenme eğrileri üzerinden aşırı/yetersiz öğrenme işaretlerini özetlemek

---

## 1. Model Evrimi – Macro F1 Tablosu

Aşağıdaki tablo, projedeki her önemli modelin **test Macro F1** skorunu, elde edilme koşullarını ve kritik hata sayılarını kronolojik sırada gösterir.

| # | Model Adı | Mimari | Epoch | LR | Scheduler | Test Macro F1 | Test İletim F1 | NR+RN Hatası | Koşul Notları |
|---|---|---|---:|---|---|---:|---:|---:|---|
| 1 | Sprint 3 Baz | CNN | 3 | 1e-3 | yok | 0.845 | 0.772 | ~350 | İlk baseline |
| 2 | Opt Exp1 | CNN | 6 | 1e-3 | yok | 0.877 | 0.788 | ~250 | Daha uzun eğitim |
| 3 | Opt Exp2 | CNN | 6 | 5e-4 | yok | 0.891 | 0.833 | ~220 | Düşük LR |
| 4 | Opt Exp3 | CNN | 12 | 5e-4 | cosine | **0.921** | 0.881 | ~180 | Scheduler eklendi |
| 5 | Opt Exp4 | CNN | 12 | 5e-4 | cosine | 0.910 | 0.838 | — | +augment+label smooth |
| 6 | ResNet-SE 12ep | ResNet-SE | 12 | 5e-4 | cosine | 0.923 | 0.859 | — | Güçlü mimari |
| 7 | CNN 50ep | CNN | 50 | 5e-4 | cosine | 0.916 | 0.870 | — | Uzun baseline |
| 8 | ResNet-SE 50ep | ResNet-SE | 50 | 3e-4 | cosine | 0.930 | 0.874 | ~115 | Mimari+uzun eğitim |
| 9 | Sweep Run12 | ResNet-SE | 50 | 1e-3 | cosine | **0.938** | 0.887 | ~100 | class_weight=none |
| 10 | Sweep Run01 | CNN | 50 | 1e-3 | cosine | 0.932 | **0.893** | — | balanced weight |
| 11 | Auto64 Run004 | CNN | 50 | 1e-3 | cosine | 0.931 | 0.886 | — | +light aug+weighted |
| 12 | Refined Run004 | CNN | 50 | 1e-3 | cosine | 0.935 | 0.893 | — | wd=5e-4, light aug |
| 13 | Refined **Run007** | CNN | 50 | 5e-4 | cosine | 0.934 | **0.912** | 163 | weighted sampler, ls=0.05 |
| 14 | BA05 | CNN | 50 | 5e-4 | cosine | 0.935 | 0.887 | **116** | Bucket-aware, fn=2.0/fr=2.0 |
| 15 | **RBA02** | CNN | 50 | 5e-4 | cosine | 0.938 | 0.906 | 137 | Bucket-aware yumuşak (1.5/2.0) |
| 16 | RBA06 | CNN | 50 | 5e-4 | cosine | 0.938 | 0.898 | 122 | Bucket-aware yumuşak, düşük wd |
| 17 | **ft06 (Final V3 baz)** | CNN | 50+8 | 5e-5 | cosine | **0.938** | 0.905 | **131** | RBA02 üzerine fine-tune |
| 18 | **Final V3 (two-stage)** | CNN+Binary | — | — | — | **0.940** | 0.909 | 131 | İki aşamalı karar kuralı |

> **Not:** NR+RN Hatası = `normal→ritim` + `ritim→normal` yanlış tahmin toplamı

---

## 2. Şu Anki Final Model (V3) – Detaylı Metrikler

### 2.1 Yapısı

Final model tek bir checkpoint değildir:
- **Ana model:** 3-sınıf CNN classifier (ft06 – run007→rba02 üzerine fine-tune)
- **Yardımcı model:** İletim vs. diğer binary classifier
- **Karar kuralı:** İki eşikli override (positive_threshold=0.83, negative_threshold=0.21)

### 2.2 Ana Metrikler

| Metrik | Doğrulama | Test |
|---|---:|---:|
| **Macro F1** | 0.9295 | **0.9396** |
| Accuracy | 0.9517 | 0.9535 |
| Loss | 0.2883 | 0.2894 |

### 2.3 Sınıf Bazlı Metrikler (Test)

| Sınıf | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0.9482 | 0.9508 | 0.9494 |
| Ritim | 0.9589 | 0.9624 | 0.9607 |
| İletim | 0.9326 | 0.8783 | 0.9046 |

> İki aşamalı karar kuralı sonrası iletim F1: **0.9086** (baz modeldeki 0.9046'dan iyileşme)

### 2.4 Confusion Matrix (Test – İki Aşamalı)

```
              Tahmin
           normal  ritim  iletim
Gerçek
normal      1370     66       7
ritim         65   1844       7
iletim         9     11     169
```

### 2.5 Koşullar

| Parametre | Değer |
|---|---|
| Mimari | baseline (1D CNN) |
| Toplam epoch | 50 (ana) + 8 (fine-tune) |
| Başlangıç LR | 5e-5 (fine-tune aşaması) |
| Scheduler | cosine (→ 2.86e-6) |
| Batch size | 64 |
| Dropout | 0.2 |
| Augment | light |
| Sampler | weighted (bucket-aware) |
| Class weight | none |
| Focus weights | normal=1.5×, ritim=2.0× |
| Label smoothing | 0.05 |
| Weight decay | 5e-5 |
| GPU | NVIDIA RTX 3090 |
| Eğitim süresi | ~73 saniye (fine-tune aşaması) |

---

## 3. En Belirgin Hata Tipleri

### 3.1 Birincil Hata Ekseni: Normal ↔ Ritim Karışması

Bu proje boyunca **tüm modellerde** en baskın hata **normal** ve **ritim** sınıfları arasındaki yanlış geçişler olmuştur.

**Final modeldeki durum:**

| Hata Yönü | Adet | Oran (toplam test) |
|---|---:|---:|
| Normal → Ritim | 66 | %1.86 |
| Ritim → Normal | 65 | %1.83 |
| **Toplam** | **131** | **%3.69** |

**Proje boyunca bu hatanın evrimi:**

| Model Aşaması | NR+RN Toplamı |
|---|---:|
| Sprint 3 (baz) | ~350 |
| 12 epoch + cosine | ~180 |
| Run007 (referans) | 163 |
| BA05 (ilk bucket-aware) | 116 |
| **Final V3** | **131** |

> Hata %63 oranında azaltıldı (350→131) ama hâlâ ana hata kaynağı.

**Hata alt tipi analizi (run007 detaylı incelemesinden):**

| Hata | Baskın SNOMED Kodları | Yorum |
|---|---|---|
| Normal→Ritim (66) | `426783006` (sinüs ritmi) | Saf normal kayıtların bir kısmı hafif ritim gibi tahmin ediliyor |
| Ritim→Normal (65) | `427393009` (sinüs aritmisi, ~55), `426177001` (sinüs bradikardisi), `284470004` (prematür atrial) | Hafif ritim bozuklukları morfolojik olarak normale çok benziyor |

**Önemli gözlem:** Model bu hataları **yüksek güvenle** yapıyor (ortalama güven: %79–83). Bu, basit kalibrasyon ile çözülemeyecek bir **karar sınırı** sorunu olduğunu gösteriyor.

### 3.2 İkincil Hata Ekseni: İletim Kenarları

| Hata Yönü | Adet | Yorum |
|---|---:|---|
| İletim → Normal | 9 | Bazı hafif iletim bozuklukları kaçırılıyor |
| İletim → Ritim | 11 | Karışık desenli iletim kayıtları ritim olarak etiketleniyor |
| Normal → İletim | 7 | Düşük hacim, kabul edilebilir |
| Ritim → İletim | 7 | Düşük hacim, kabul edilebilir |

İletim false negative'lerde baskın kodlar:
- `59118001` (sağ dal bloğu)
- `270492004` (1. derece AV blok)
- `10370003` (pacing ritmi)
- `164909002` (sol dal bloğu)

> İletim sınıfı toplam 189 test kaydından 169'unu doğru yakalıyor (%89,4 recall). False positive sayısı da düşük (14 adet), yani precision yüksek (%92,3).

### 3.3 Hata Özet Matrisi

```
Hata Büyüklük Sıralaması:
1. Normal ↔ Ritim karışması  : 131 hata  (tüm hataların ~79%'u)
2. İletim → Normal/Ritim     : 20 hata   (tüm hataların ~12%'u)
3. Normal/Ritim → İletim     : 14 hata   (tüm hataların ~9%'u)
                                ─────
                  Toplam hata: 165 / 3548 test kaydı (%4.65)
```

---

## 4. Öğrenme Eğrileri ve Aşırı/Yetersiz Öğrenme Analizi

### 4.1 Final Model (ft06) Eğitim Logları

| Epoch | Train Loss | Val Loss | Val Macro F1 | LR |
|---:|---:|---:|---:|---:|
| 1 | 0.1871 | 0.2891 | 0.9249 | 5.00e-5 |
| 2 | 0.1857 | 0.2963 | 0.9267 | 4.81e-5 |
| 3 | 0.1854 | 0.2877 | 0.9218 | 4.28e-5 |
| 4 | 0.1845 | 0.2875 | 0.9250 | 3.49e-5 |
| 5 | 0.1839 | 0.2909 | 0.9266 | 2.55e-5 |
| 6 | 0.1833 | 0.2900 | 0.9282 | 1.61e-5 |
| **7** | **0.1827** | **0.2883** | **0.9295** | 8.18e-6 |
| 8 | 0.1834 | 0.2875 | 0.9286 | 2.86e-6 |

### 4.2 Aşırı Öğrenme (Overfitting) Değerlendirmesi

**Train Loss – Val Loss farkı:**
- Train loss: ~0.183
- Val loss: ~0.288
- Fark: ~0.105

Bu fark belirgin görünse de, modelin **fine-tuning** aşamasında olduğu ve zaten güçlü bir checkpoint'ten devam edildiği dikkate alınmalıdır. Ayrıca:

1. **Val loss kararlı:** 8 epoch boyunca val loss dar bir bantta salınıyor (0.287–0.296). Artış trendi yok.
2. **Val Macro F1 artıyor:** Epoch 3'te hafif düşüş dışında sürekli yükseliş trendi var.
3. **Tepe ile son fark minimal:** En iyi epoch (7) ile son epoch (8) arasındaki F1 farkı sadece 0.0009.

**Sonuç: Belirgin overfitting yok.** Model val metriklerinde kararlı ve tepe performansına yakın durarak eğitimi tamamlamıştır.

### 4.3 Yetersiz Öğrenme (Underfitting) Değerlendirmesi

- Val Macro F1 = 0.9295 (doğrulama) → Test Macro F1 = 0.9383 → Test genellemesi doğrulamadan bile iyi
- Bu, modelin veriyi yeterince öğrendiğini ve genelleme yapabildiğini gösterir

Ancak şu gözlemler "marjinal yetersiz öğrenme" olarak yorumlanabilir:

1. **Train loss çok az iniyor:** 0.187 → 0.183 (8 epoch boyunca sadece 0.004 düşüş). Model zaten güçlü başladığı için gradient çok küçük.
2. **Fine-tune turunun katkısı sınırlı:** Ana modelin (rba02, 50 epoch) üzerine eklenen 8 epoch fine-tune, val Macro F1'i sadece ~0.002 artırdı.

**Sonuç: Yapısal yetersiz öğrenme yok.** Ama model mevcut kapasitesi ile öğrenebileceğinin sınırına yaklaşmış durumda.

### 4.4 Erken Durdurma

- **Erken durdurma kullanılmadı** — tüm 8 epoch koşuldu
- En iyi checkpoint Epoch 7'den seçildi (validation Macro F1 tabanlı)
- Epoch 7→8 arası sadece 0.0009 F1 düşüşü → aşırı eğitim sinyali değil

---

## 5. Denenen Ama Başarısız / Sınırlı Kalan Stratejiler

| Strateji | Sonuç | Neden Başarısız |
|---|---|---|
| **Weighted sampler (tek başına)** | Val Macro F1 düştü | Aşırı telafi etkisi, iletim precision kaybı |
| **Hafif augmentasyon (tek başına)** | Ek fayda sınırlı | Bu veri setinde augment marjinal etki |
| **Label smoothing (0.05–0.1)** | Karışık sonuçlar | Bazı koşullarda faydalı, bazılarında zararlı |
| **Fine-tuning (düşük LR ile resume)** | Mevcut modeli geçemedi | Model zaten güçlü optimumda |
| **Sınıf-özel kalibrasyon (iletim bias)** | Validasyonda +, testte − | Overfitting to validation |
| **FP-odaklı fine-tune (8 deneme)** | NR+RN azaldı ama Macro F1 düştü | FP azaltırken genel ayrım keskinliği bozuldu |

---

## 6. Başarılı Stratejiler

| Strateji | Katkı |
|---|---|
| **Cosine scheduler** | En büyük tek sıçrama (Macro F1: 0.891 → 0.921) |
| **Daha uzun eğitim (50 epoch)** | Modelin tam kapasitesine ulaşmasını sağladı |
| **ResNet-SE mimarisi** | Toplam skorda CNN'i geçti (ancak final modelde CNN tercih edildi) |
| **Bucket-aware sampling** | Normal↔Ritim karışmasını 163→116'ya düşürdü |
| **Yumuşak bucket ağırlıkları (rba02)** | İletim kaybını geri alırken hata azalmasını korudu |
| **İki aşamalı classifier (V3)** | İletim FN'yi azaltırken toplam Macro F1'i de artırdı |

---

## 7. İyileştirme Yönleri

### Kısa Vadeli (Mevcut Altyapı İle)

1. **Normal↔Ritim sınırındaki spesifik kodlar için hedefli oversampling** — `427393009` (sinüs aritmisi) ve `426783006` (sinüs ritmi) arasındaki ayrımı güçlendirmek
2. **İki aşamalı modelin eşik optimizasyonu** — Daha geniş grid arama veya Bayesian optimizasyon
3. **Hard example mining** — Yüksek güvenle yanlış yapılan kayıtlarla ek eğitim turu

### Orta Vadeli

4. **Daha güçlü 1D mimari** — Multi-scale kernel CNN veya temporal attention
5. **Focal loss veya margin-based loss** — Sınıf sınırlarını keskinleştirmek
6. **EMA (Exponential Moving Average)** — Eğitim stabilitesi için

### Uzun Vadeli

7. **Hasta bazlı split** — Veri setinde hasta ID bulunursa split yeniden üretilmeli
8. **Ensemble** — Run007 (iletim güçlü) + RBA02 (dengeli) + ResNet-SE (toplam güçlü) birleşimi
9. **Etiket belirsizliği denetimi** — Normal ve hafif ritim arasındaki kayıtların gerçek etiketlerinin uzman doğrulaması

---

## 8. Sonuç

| Kategori | Durum |
|---|---|
| **Toplam başarı** | Test Macro F1 = **0.940** (iki aşamalı), **0.938** (tek model) |
| **En zayıf sınıf** | İletim (F1: 0.909 — dengesizlik nedeniyle en zor sınıf ama güçlü) |
| **Baskın hata** | Normal↔Ritim karışması (131/165 hatanın %79'u) |
| **Overfitting** | Yok — val metrikleri kararlı, test genellemesi güçlü |
| **Underfitting** | Yapısal değil, ama kapasite tavanına yaklaşılmış |
| **İyileştirme alanı** | Hedefli veri stratejileri > genel sweep; iki aşamalı model umut verici |
