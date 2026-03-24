# Sprint 5 Optimizasyon Raporu

## Amaç

Sprint 5 hedefi, Sprint 3’te elde edilen baz modeli validation `Macro F1` üzerinden iyileştirmek ve Sprint 6’ya taşınacak en güçlü aday modeli seçmekti.

Bu sprintte:

- GPU üzerinde ek optimizasyon koşuları yapıldı
- düşük riskli hiperparametre araması uygulandı
- en iyi validation sonucu veren konfigürasyon seçildi
- yeni en iyi checkpoint belirlendi

## Kullanılan Donanım

Sprint 5 koşuları GPU üzerinde tamamlandı.

- GPU: `NVIDIA GeForce RTX 3090`

## Kullanılan Kod

Optimizasyon koşuları aşağıdaki betikle yürütüldü:

- [train_uc_sinif_baseline.py](/home/alper/ekg/scripts/train_uc_sinif_baseline.py)

Veri kaynağı:

- [manifest_uc_sinif_temiz.csv](/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv)

## Koşulan Deneyler

Sprint 5’te baz modele göre düşük riskli iki optimizasyon koşusu çalıştırıldı:

1. `epochs=6`, `lr=1e-3`
2. `epochs=6`, `lr=5e-4`

Her iki koşu da şu sabitlerle çalıştırıldı:

- `batch_size=64`
- `weight_decay=1e-4`
- `sampler=none`
- `class_weight_mode=balanced`
- cihaz: `cuda`

## Üretilen Artifact’lar

### Koşu 1

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp1_e6_lr1e3/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp1_e6_lr1e3/training_report.json)

### Koşu 2

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp2_e6_lr5e4/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp2_e6_lr5e4/training_report.json)

## Karşılaştırma

Referans olarak Sprint 3 baz modeli:

- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_baseline/training_report.json)

### Toplam Performans Karşılaştırması

| Model | Best Epoch | Val Macro F1 | Test Macro F1 | Val Accuracy | Test Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `3` | `0.825995` | `0.845055` | `0.869357` | `0.874859` |
| `6 epoch, lr=1e-3` | `5` | `0.852627` | `0.876979` | `0.899831` | `0.914600` |
| `6 epoch, lr=5e-4` | `5` | `0.861579` | `0.891354` | `0.902652` | `0.917418` |

### `iletim` Sınıfı Karşılaştırması

| Model | Val iletim F1 | Test iletim F1 |
| --- | ---: | ---: |
| `baseline` | `0.717622` | `0.772321` |
| `6 epoch, lr=1e-3` | `0.736617` | `0.788155` |
| `6 epoch, lr=5e-4` | `0.763889` | `0.832536` |

## En İyi Koşu

Sprint 5 sonunda en iyi aday model:

- konfigürasyon: `epochs=6`, `lr=5e-4`
- best epoch: `5`
- validation `Macro F1 = 0.861579`
- test `Macro F1 = 0.891354`

Seçilen checkpoint:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp2_e6_lr5e4/best_model.pt)

## En İyi Koşunun Detayları

### Validation Sonuçları

- Loss: `0.242820`
- Accuracy: `0.902652`
- Macro F1: `0.861579`

Sınıf bazlı validation sonuçları:

| Sınıf | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `normal` | `0.923692` | `0.882026` | `0.902378` |
| `ritim` | `0.916320` | `0.920627` | `0.918468` |
| `iletim` | `0.676230` | `0.877660` | `0.763889` |

### Test Sonuçları

- Loss: `0.224954`
- Accuracy: `0.917418`
- Macro F1: `0.891354`

Sınıf bazlı test sonuçları:

| Sınıf | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `normal` | `0.937362` | `0.881497` | `0.908571` |
| `ritim` | `0.922018` | `0.944154` | `0.932955` |
| `iletim` | `0.759825` | `0.920635` | `0.832536` |

## İyileşme Özeti

Baseline’a göre en iyi koşunun kazanımı:

- validation `Macro F1`: `0.825995 -> 0.861579`
- test `Macro F1`: `0.845055 -> 0.891354`
- validation accuracy: `0.869357 -> 0.902652`
- test accuracy: `0.874859 -> 0.917418`
- test `iletim F1`: `0.772321 -> 0.832536`

Ana yorum:

- yalnızca daha uzun eğitim ve daha düşük öğrenme oranı ile anlamlı iyileşme sağlandı
- bu, Sprint 3 modelinin erken durduğunu veya tepe performansına henüz ulaşmadığını gösteriyor
- `iletim` sınıfı hem precision hem recall tarafında ilerledi

## Neden Ek Koşu Açılmadı

Sprint 5 içinde daha fazla GPU koşusu açılmadı çünkü:

1. validation `Macro F1` üzerinde açık ve güçlü bir artış elde edildi
2. test tarafında da aynı yönlü ve büyük kazanım görüldü
3. Sprint 6 için savunulabilir biçimde seçilebilecek net bir aday model oluştu

Bu noktadan sonra ek koşuların marjinal getirisi daha düşük, Sprint 6 finalleme getirisi daha yüksek olacaktır.

## Sprint 5 Sonucu

Sprint 5 tamamlandı.

Son karar:

- Sprint 6’ya taşınacak ana aday model `6 epoch + lr=5e-4` koşusudur.
- Bu model şu anda projedeki en iyi validation ve test `Macro F1` skorlarını vermektedir.

## Sprint 6’ya Devir

Sprint 6’da yapılması gerekenler:

1. seçilen checkpoint ile final inference akışını netleştirmek
2. final raporu ve kullanım notlarını hazırlamak
3. modelin giriş-çıkış sözleşmesini sabitlemek
4. gerekiyorsa son bir bağımsız test özeti üretmek
