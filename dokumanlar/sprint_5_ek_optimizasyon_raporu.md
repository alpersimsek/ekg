# Sprint 5 Ek Optimizasyon Raporu

## Kapsam

Bu rapor, Sprint 5 tamamlandıktan sonra kullanıcı talebiyle yapılan ek optimizasyon turunu özetler.

Bu turda amaç:

- epoch sayısını artırmak
- öğrenme oranı planlayıcısı eklemek
- hafif augmentasyon ve label smoothing gibi ek düzenlileştirmeleri denemek
- validation `Macro F1` üzerinden mevcut en iyi modeli daha ileri taşımak

## Yapılan Kod Güncellemeleri

Eğitim betiği aşağıdaki yeteneklerle genişletildi:

- `cosine` learning rate scheduler
- `dropout` parametresi
- `label_smoothing`
- hafif sinyal augmentasyonu
- `min_lr` desteği

Güncellenen betik:

- [train_uc_sinif_baseline.py](/home/alper/ekg/scripts/train_uc_sinif_baseline.py)

## Koşulan Yeni Deneyler

### Deney 1

- `epochs=12`
- `lr=5e-4`
- `scheduler=cosine`
- `augment=none`
- `label_smoothing=0.0`
- `dropout=0.2`

Artifact’lar:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/training_report.json)

### Deney 2

- `epochs=12`
- `lr=5e-4`
- `scheduler=cosine`
- `augment=light`
- `label_smoothing=0.05`
- `dropout=0.3`

Artifact’lar:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp4_e12_cosine_aug/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp4_e12_cosine_aug/training_report.json)

## Karşılaştırma

Referans olarak üç model karşılaştırıldı:

1. Sprint 3 baseline
2. Sprint 5’in ilk en iyi modeli: `6 epoch + lr=5e-4`
3. Yeni `12 epoch + cosine`
4. Yeni `12 epoch + cosine + augment + label smoothing`

| Model | Best Epoch | Val Macro F1 | Test Macro F1 | Val iletim F1 | Test iletim F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `3` | `0.825995` | `0.845055` | `0.717622` | `0.772321` |
| `6 epoch, lr=5e-4` | `5` | `0.861579` | `0.891354` | `0.763889` | `0.832536` |
| `12 epoch, cosine` | `12` | `0.908420` | `0.921371` | `0.850000` | `0.881443` |
| `12 epoch, cosine, aug` | `12` | `0.907110` | `0.909844` | `0.835749` | `0.838235` |

## En İyi Yeni Model

En iyi model:

- `12 epoch + cosine scheduler`
- validation `Macro F1 = 0.908420`
- test `Macro F1 = 0.921371`

Seçilen checkpoint:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/best_model.pt)

## En İyi Modelin Detayları

### Validation

- Loss: `0.226937`
- Accuracy: `0.933126`
- Macro F1: `0.908420`

| Sınıf | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `normal` | `0.918176` | `0.950035` | `0.933834` |
| `ritim` | `0.960348` | `0.923238` | `0.941427` |
| `iletim` | `0.801887` | `0.904255` | `0.850000` |

### Test

- Loss: `0.201934`
- Accuracy: `0.939121`
- Macro F1: `0.921371`

| Sınıf | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `normal` | `0.926230` | `0.939709` | `0.932921` |
| `ritim` | `0.957560` | `0.942067` | `0.949750` |
| `iletim` | `0.859296` | `0.904762` | `0.881443` |

## Önceki En İyi Modele Karşı Kazanım

`6 epoch + lr=5e-4` koşusuna göre yeni en iyi model:

- validation `Macro F1`: `0.861579 -> 0.908420`
- test `Macro F1`: `0.891354 -> 0.921371`
- validation `iletim F1`: `0.763889 -> 0.850000`
- test `iletim F1`: `0.832536 -> 0.881443`

Bu artış marjinal değil, açık biçimde anlamlıdır.

## Augmentasyonlu Koşunun Yorumu

`12 epoch + cosine + light augment + label smoothing` koşusu tamamen başarısız olmadı:

- validation `Macro F1 = 0.907110`
- test `Macro F1 = 0.909844`

Ancak:

- scheduler-only koşunun gerisinde kaldı
- özellikle test tarafında daha düşük kaldı
- bu veri ve mimari için hafif augmentasyonun ek faydası sınırlı göründü

Karar:

- şu aşamada augmentasyonlu model ana aday yapılmadı

## Teknik Yorum

Bu ek turun en önemli çıktısı şu:

- model hâlâ kapasite sınırına gelmemişti
- daha uzun eğitim ve scheduler eklemek, weighted sampler veya ağır dengeleme stratejilerinden çok daha etkili oldu

Bu da bize şunu söylüyor:

- mevcut veri hattı doğru
- ana kazanım veri dengesizliğinden çok optimizasyon dinamiğinden geldi

## Nihai Karar

Bu ek optimizasyon turu sonunda projedeki en iyi model artık:

- `12 epoch`
- `lr=5e-4`
- `cosine scheduler`
- `balanced class weights`
- `augmentasyon yok`

Sprint 6’ya taşınması gereken model budur.

## Sonuç

Evet, model daha da geliştirilebildi. Ek epoch ve scheduler ile validation `Macro F1` `0.908420`, test `Macro F1` ise `0.921371` seviyesine taşındı. Bu model, şu an proje içindeki en güçlü ve en savunulabilir adaydır.
