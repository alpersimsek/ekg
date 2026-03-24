# Sprint 3 Baz Model Eğitim Raporu

## Amaç

Sprint 3 hedefi, Sprint 2’de hazırlanan temiz 3 sınıflı kohort üzerinde çalışan ilk bazal modeli eğitmek ve `Macro F1` odaklı ilk referans performansı üretmekti.

Bu sprint kapsamında:

- bazal eğitim betiği yazıldı
- 12 kanallı ham EKG sinyali üzerinden çalışan 1D CNN tabanlı model kuruldu
- weighted cross-entropy ile sınıf dengesizliği ele alındı
- validation ve test metrikleri üretildi
- en iyi epoch checkpoint olarak kaydedildi

## Üretilen Kod ve Artifact’lar

### Kod

- [train_uc_sinif_baseline.py](/home/alper/ekg/scripts/train_uc_sinif_baseline.py)

### Model ve Rapor Çıktıları

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_baseline/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_baseline/training_report.json)

## Kullanılan Veri

- Manifest: [manifest_uc_sinif_temiz.csv](/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv)
- Eğitim kayıt sayısı: `28366`
- Doğrulama kayıt sayısı: `3544`
- Test kayıt sayısı: `3548`

Sınıf düzeni:

- `normal`
- `ritim`
- `iletim`

## Model Yapısı

Kullanılan baz mimari:

- giriş: `12 x 5000`
- lead-bazlı z-score normalizasyon
- 1D CNN stem
- iki aşamalı kanal genişletmeli blok
- global average pooling
- tam bağlı sınıflandırma başlığı

Kayıp fonksiyonu:

- `weighted cross-entropy`

Sınıf ağırlıkları:

- `normal`: `0.819705`
- `ritim`: `0.617036`
- `iletim`: `6.273531`

## Eğitim Konfigürasyonu

- cihaz: `cuda`
- epoch: `3`
- batch size: `64`
- learning rate: `0.001`
- weight decay: `0.0001`
- en iyi epoch: `3`
- toplam süre: `18.08 saniye`

## Epoch Bazlı Sonuçlar

| Epoch | Train Loss | Val Loss | Val Macro F1 | Val Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | `0.4935` | `0.3390` | `0.8065` | `0.8660` |
| 2 | `0.3632` | `0.4000` | `0.7815` | `0.8448` |
| 3 | `0.3139` | `0.3438` | `0.8260` | `0.8694` |

Yorum:

- Eğitim kaybı düzenli biçimde düştü.
- En iyi validation sonucu `Epoch 3` sonunda alındı.
- `Epoch 2` performans düşüşü görüldü ancak `Epoch 3` ile toparlandı.

## Validation Sonuçları

### Ana Metrikler

- Validation Loss: `0.3438`
- Validation Accuracy: `0.8694`
- Validation Macro F1: `0.8260`

### Sınıf Bazlı Sonuçlar

| Sınıf | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `normal` | `0.8211` | `0.9493` | `0.8806` |
| `ritim` | `0.9680` | `0.8063` | `0.8798` |
| `iletim` | `0.5972` | `0.8989` | `0.7176` |

### Validation Confusion Matrix

Gerçek sınıf satırlarda, tahmin sütunlarda olacak şekilde:

| Gerçek \ Tahmin | normal | ritim | iletim |
| --- | ---: | ---: | ---: |
| `normal` | `1368` | `42` | `31` |
| `ritim` | `288` | `1544` | `83` |
| `iletim` | `10` | `9` | `169` |

Yorum:

- `iletim recall` yüksek: `0.8989`
- ancak `iletim precision` görece düşük: `0.5972`
- model bazı `normal` ve `ritim` örneklerini `iletim` olarak işaretliyor

## Test Sonuçları

### Ana Metrikler

- Test Loss: `0.3281`
- Test Accuracy: `0.8749`
- Test Macro F1: `0.8451`

### Sınıf Bazlı Sonuçlar

| Sınıf | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `normal` | `0.8154` | `0.9459` | `0.8758` |
| `ritim` | `0.9697` | `0.8173` | `0.8870` |
| `iletim` | `0.6680` | `0.9153` | `0.7723` |

### Test Confusion Matrix

| Gerçek \ Tahmin | normal | ritim | iletim |
| --- | ---: | ---: | ---: |
| `normal` | `1365` | `41` | `37` |
| `ritim` | `301` | `1566` | `49` |
| `iletim` | `8` | `8` | `173` |

Yorum:

- Test `Macro F1` validation’dan daha iyi geldi: `0.8451`
- `iletim` sınıfında recall güçlü kaldı: `0.9153`
- en belirgin hata paterni, bazı `ritim` örneklerinin `normal` olarak tahmin edilmesi

## Sprint 3 Sonucu

Sprint 3 tamamlandı.

Ana çıktı:

- ilk bazal model başarıyla eğitildi
- validation `Macro F1 = 0.8260`
- test `Macro F1 = 0.8451`

Bu sonuç, projede referans alınacak ilk sağlam baseline olarak kabul edilebilir.

## Teknik Değerlendirme

Güçlü taraflar:

- kısa sürede kararlı eğitim
- yüksek `normal` ve `ritim` F1
- `iletim` sınıfında yüksek recall
- Macro F1 odaklı anlamlı baz performans

Zayıf taraflar:

- `iletim precision` halen düşük
- `ritim` sınıfında recall artırılabilir
- mevcut split hasta-bazlı değil, kayıt-bazlı stratified

## Sprint 4 İçin Odak Noktaları

Sprint 4’te öncelikli analiz başlıkları:

1. `iletim` false positive örneklerini incelemek
2. `ritim -> normal` kayan örnekleri analiz etmek
3. balanced sampler ve loss varyasyonlarını karşılaştırmak
4. filtreleme ve hafif augmentasyonun validation `Macro F1` etkisini ölçmek

## Sonuç

Sprint 3 hedefi yerine getirildi. Mevcut baz model, 12 kanallı ham EKG sinyalinden `normal / ritim / iletim` tahmini yapabilen, GPU üzerinde hızlı çalışan ve `Macro F1` açısından savunulabilir ilk referans modeli sağlamıştır.
