# Ayar Envanteri ve En İyi Değerler Raporu

## Amaç

Bu rapor iki soruya cevap verir:

1. Modeli daha nasıl geliştirebiliriz?
2. Şu anda elimizde kaç ayar yüzeyi var ve bunların şimdiye kadar görülen en iyi değerleri neler?

Bu rapor, doğrudan [train_uc_sinif_baseline.py](/home/alper/ekg/scripts/train_uc_sinif_baseline.py) betiğindeki aktif ayarlar ve `artifacts/*/training_report.json` altındaki gerçek deney kayıtları üzerinden hazırlanmıştır.

## Kısa Cevap

Evet, model daha da geliştirilebilir. Ancak bundan sonraki gelişim büyük olasılıkla yalnızca `epoch` artırarak değil, şu alanlarda gelecektir:

- mimari kapasite ve mimari kararlılık
- optimizasyon planlayıcısı
- veri artırma stratejisinin daha dikkatli tasarımı
- `iletim` sınıfına özel hedefli iyileştirme
- karar kalibrasyonu ve sınıf eşiği mantığı

Şu anki en güçlü toplam model:

- koşu: `uc_sinif_test50_resnet_se`
- validation `Macro F1 = 0.921781`
- test `Macro F1 = 0.929923`

## Aktif Ayar Yüzeyi

Eğitim betiğinde doğrudan dışarıdan kontrol edilen `14` ana ayar vardır:

1. `manifest`
2. `output-dir`
3. `epochs`
4. `batch-size`
5. `lr`
6. `weight-decay`
7. `num-workers`
8. `seed`
9. `dropout`
10. `model`
11. `label-smoothing`
12. `scheduler`
13. `min-lr`
14. `augment`
15. `resume-checkpoint`
16. `sampler`
17. `class-weight-mode`

Pratik optimizasyon yüzeyi olarak etkili olanlar ise şunlardır:

- `model`
- `epochs`
- `lr`
- `dropout`
- `label-smoothing`
- `scheduler`
- `min-lr`
- `augment`
- `sampler`
- `class-weight-mode`

Yani gerçek hiperparametre optimizasyon yüzeyi yaklaşık `10` ana değişkenden oluşuyor.

## Şimdiye Kadar Denenen Değerler

Gerçek deney kayıtlarından çıkan benzersiz değerler:

| Ayar | Denenen Değerler |
| --- | --- |
| `model` | `baseline`, `resnet_se` |
| `epochs` | `1`, `3`, `5`, `6`, `12`, `50` |
| `batch_size` | `64` |
| `lr` | `2e-5`, `1e-4`, `3e-4`, `5e-4`, `1e-3` |
| `min_lr` | `None`, `1e-6`, `1e-5` |
| `weight_decay` | `1e-4` |
| `dropout` | `0.2`, `0.3` |
| `label_smoothing` | `0.0`, `0.05` |
| `scheduler` | `none`, `cosine` |
| `augment` | `none`, `light` |
| `sampler` | `none`, `weighted` |
| `class_weight_mode` | `balanced`, `none` |

## Şimdiye Kadarki En İyi Değerler

### En İyi Toplam Model

Kaynak koşu:

- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/training_report.json)

En iyi validation ve test `Macro F1` değerlerini veren ayar seti:

| Ayar | En İyi Değer |
| --- | --- |
| `model` | `resnet_se` |
| `epochs` | `50` |
| `batch_size` | `64` |
| `lr` | `3e-4` |
| `scheduler` | `cosine` |
| `min_lr` | `1e-6` |
| `dropout` | `0.3` |
| `label_smoothing` | `0.0` |
| `augment` | `none` |
| `sampler` | `none` |
| `class_weight_mode` | `balanced` |
| `weight_decay` | `1e-4` |

Bu koşunun sonuçları:

- validation `Macro F1 = 0.921781`
- test `Macro F1 = 0.929923`
- validation accuracy = `0.946953`
- test accuracy = `0.954340`
- validation `iletim F1 = 0.863158`
- test `iletim F1 = 0.873995`

### `iletim` Test F1 İçin En İyi Koşu

İlginç bir ayrıntı:

- en iyi toplam model `resnet_se 50 epoch`
- ama en iyi `test iletim F1` değeri başka bir koşuda

Kaynak koşu:

- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/training_report.json)

Bu koşuda:

- `model = baseline`
- `epochs = 12`
- `lr = 5e-4`
- `scheduler = cosine`
- `dropout = 0.2`
- `augment = none`
- `class_weight_mode = balanced`

Sonuç:

- test `iletim F1 = 0.881443`

Yorum:

- toplam lider model ile `iletim` özel lider model aynı değil
- eğer klinik öncelik `iletim` ise seçim ölçütü ayrıca tartışılabilir

## Hangi Ayarlar Ne Kadar Etki Etti

Şimdiye kadarki deneylere göre etki büyüklüğü sıralaması:

### Çok Yüksek Etki

- `model`
- `epochs`
- `scheduler`
- `lr`

### Orta Etki

- `dropout`
- `class-weight-mode`
- `sampler`

### Düşük veya Kararsız Etki

- `augment=light`
- `label_smoothing=0.05`

## Şu Ana Kadarki Öğrenim

Deneylerden çıkan en net teknik dersler:

1. `cosine scheduler` eklemek, erken koşulara göre büyük sıçrama sağladı.
2. Daha uzun eğitim özellikle `baseline` ve sonra `resnet_se` için kritik fark yarattı.
3. `weighted sampler` kısa vadede beklenen kazancı vermedi.
4. Hafif augmentasyon bu veri ve mimari üzerinde şu ana kadar net kazanç üretmedi.
5. Daha güçlü mimari olan `resnet_se`, sonunda toplam liderliği aldı.

## Daha Nasıl İyileştirebiliriz

Mevcut ayar yüzeyinden sonra en mantıklı geliştirme alanları:

### 1. Mimari Derinleştirme ve Varyantlar

- `resnet_se` için daha derin blok sayısı
- kernel boyutlarını çok ölçekli hale getirmek
- temporal attention veya transformer-lite başlık
- lead-wise attention ve lead grouping

### 2. Optimizasyon Ayarları

- `one-cycle` scheduler
- `warmup + cosine`
- `EMA` model ağırlıkları
- gradient clipping

### 3. Kayıp Fonksiyonu

- focal loss
- class-balanced focal loss
- margin-aware loss

### 4. Veri Tarafı

- daha kontrollü augmentasyon
- `ritim -> normal` karışan alt gruplara hedefli oversampling
- `iletim` için daha seçici zor örnek madenciliği

### 5. Değerlendirme ve Karar Katmanı

- sıcaklık kalibrasyonu
- sınıf bazlı karar mantığı
- `iletim` için precision/recall odaklı ayrı model seçimi

## Pratik Olarak Bir Sonraki En Mantıklı 5 Deneme

1. `resnet_se`, `epochs=70`, `lr=2e-4`, `cosine`, `dropout=0.3`
2. `resnet_se`, `epochs=70`, `lr=3e-4`, `warmup+cosine`
3. `resnet_se`, `epochs=50`, `focal loss`, `balanced`
4. `resnet_se`, `epochs=50`, `EMA` eklenmiş eğitim
5. `resnet_se`, `epochs=50`, `hard example` odaklı sampler

## Nihai Durum

Şu anda elimizde:

- optimize edilmiş bir `baseline` hattı
- daha güçlü bir `resnet_se` hattı
- confusion matrix ve Grad-CAM üretim altyapısı
- hiperparametre yüzeyinin önemli kısmı için gerçek deney sonuçları

Mevcut en iyi toplam ayar seti:

- `model = resnet_se`
- `epochs = 50`
- `batch_size = 64`
- `lr = 3e-4`
- `scheduler = cosine`
- `min_lr = 1e-6`
- `dropout = 0.3`
- `label_smoothing = 0.0`
- `augment = none`
- `sampler = none`
- `class_weight_mode = balanced`
- `weight_decay = 1e-4`

## Sonuç

Modeli geliştirmek için hâlâ alan var. Ancak artık en büyük kazançlar kaba `epoch` artırmadan değil, `resnet_se` hattı üzerinde daha sofistike optimizasyon, loss ve veri stratejilerinden gelecektir.
