# 50 Epoch Testler, Confusion Matrix ve Grad-CAM Raporu

## Kapsam

Bu rapor, minimum `50 epoch` şartıyla çalıştırılan yeni deneyleri, bu deneylere ait confusion matrix çıktılarını ve Grad-CAM görselleştirmelerini özetler.

Bu turda iki model koşuldu:

1. `baseline` mimarisi, uzun eğitim
2. `resnet_se` mimarisi, uzun eğitim

Her iki koşu da GPU üzerinde tamamlandı.

## Kullanılan GPU

- `NVIDIA GeForce RTX 3090`

## Koşulan Deneyler

### Deney 1: Baseline 50 Epoch

Konfigürasyon:

- model: `baseline`
- epoch: `50`
- lr: `5e-4`
- scheduler: `cosine`
- min lr: `1e-6`
- batch size: `64`
- class weights: `balanced`

Artifact’lar:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/training_report.json)
- [confusion_matrix_test.png](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/confusion_matrix_test.png)
- [gradcam_summary_test.json](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/gradcam_summary_test.json)

### Deney 2: ResNet-SE 50 Epoch

Konfigürasyon:

- model: `resnet_se`
- epoch: `50`
- lr: `3e-4`
- scheduler: `cosine`
- min lr: `1e-6`
- dropout: `0.3`
- batch size: `64`
- class weights: `balanced`

Artifact’lar:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/training_report.json)
- [confusion_matrix_test.png](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/confusion_matrix_test.png)
- [gradcam_summary_test.json](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/gradcam_summary_test.json)

## Sonuç Karşılaştırması

Karşılaştırma referansı olarak önceki en iyi 12-epoch model de eklenmiştir:

- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/training_report.json)

| Model | Best Epoch | Val Macro F1 | Test Macro F1 | Val Accuracy | Test Accuracy | Val iletim F1 | Test iletim F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `12 epoch baseline + cosine` | `12` | `0.908420` | `0.921371` | `0.933126` | `0.939121` | `0.850000` | `0.881443` |
| `50 epoch baseline + cosine` | `36` | `0.909418` | `0.916426` | `0.931433` | `0.937148` | `0.857895` | `0.869565` |
| `50 epoch resnet_se + cosine` | `50` | `0.921781` | `0.929923` | `0.946953` | `0.954340` | `0.863158` | `0.873995` |

## Ana Gözlem

### 50 Epoch Baseline

- validation tarafında çok küçük bir artış sağladı: `0.908420 -> 0.909418`
- ancak test tarafında mevcut 12-epoch lider modelin gerisine düştü: `0.921371 -> 0.916426`

Yorum:

- daha uzun eğitim baseline mimarisinde ek validation tepe noktası üretti
- fakat genelleme tarafında aynı kazanımı taşımadı

### 50 Epoch ResNet-SE

- validation `Macro F1` en iyi sonuç: `0.921781`
- test `Macro F1` en iyi sonuç: `0.929923`
- genel doğrulukta da açık biçimde en iyi model oldu

Yorum:

- daha güçlü mimari + uzun eğitim kombinasyonu bu turdaki en iyi toplam sonucu verdi
- bu model artık projedeki yeni ana aday olarak kabul edilmelidir

## Confusion Matrix Sonuçları

### Baseline 50 Epoch Test Confusion Matrix

Kaynak görsel:

- [confusion_matrix_test.png](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/confusion_matrix_test.png)

Sayısal matris:

| Gerçek \ Tahmin | normal | ritim | iletim |
| --- | ---: | ---: | ---: |
| `normal` | `1367` | `64` | `12` |
| `ritim` | `111` | `1798` | `7` |
| `iletim` | `12` | `17` | `160` |

### ResNet-SE 50 Epoch Test Confusion Matrix

Kaynak görsel:

- [confusion_matrix_test.png](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/confusion_matrix_test.png)

Sayısal matris:

| Gerçek \ Tahmin | normal | ritim | iletim |
| --- | ---: | ---: | ---: |
| `normal` | `1367` | `65` | `11` |
| `ritim` | `50` | `1856` | `10` |
| `iletim` | `13` | `13` | `163` |

### Confusion Matrix Yorumu

`resnet_se` modelinin temel farkı:

- `ritim -> normal` hatalarını ciddi biçimde azaltması
- `ritim` sınıfını daha temiz toplaması
- toplam sınıf ayrımını daha dengeli hale getirmesi

Ancak not:

- `iletim` test F1, 12-epoch eski lider modelde biraz daha yüksekti
- buna rağmen toplam `Macro F1` ve toplam doğruluk açısından `resnet_se` önde

## Grad-CAM Çıktıları

Her model için test kümesinden 6 örnek seçildi:

- 1 adet doğru `normal`
- 1 adet doğru `ritim`
- 1 adet doğru `iletim`
- 1 adet `ritim -> normal`
- 1 adet `ritim -> iletim`
- 1 adet `normal -> iletim`

### Baseline 50 Epoch Grad-CAM Örnekleri

- [gradcam_A4113_normal_to_normal.png](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/gradcam_A4113_normal_to_normal.png)
- [gradcam_A0064_ritim_to_ritim.png](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/gradcam_A0064_ritim_to_ritim.png)
- [gradcam_A0730_iletim_to_iletim.png](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/gradcam_A0730_iletim_to_iletim.png)
- [gradcam_JS45132_ritim_to_normal.png](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/gradcam_JS45132_ritim_to_normal.png)
- [gradcam_A2366_ritim_to_iletim.png](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/gradcam_A2366_ritim_to_iletim.png)
- [gradcam_HR07758_normal_to_iletim.png](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/gradcam_HR07758_normal_to_iletim.png)

Özet dosyası:

- [gradcam_summary_test.json](/home/alper/ekg/artifacts/uc_sinif_test50_baseline/visuals/gradcam_summary_test.json)

### ResNet-SE 50 Epoch Grad-CAM Örnekleri

- [gradcam_A0772_normal_to_normal.png](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/gradcam_A0772_normal_to_normal.png)
- [gradcam_A0117_ritim_to_ritim.png](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/gradcam_A0117_ritim_to_ritim.png)
- [gradcam_A1152_iletim_to_iletim.png](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/gradcam_A1152_iletim_to_iletim.png)
- [gradcam_JS35470_ritim_to_normal.png](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/gradcam_JS35470_ritim_to_normal.png)
- [gradcam_A2366_ritim_to_iletim.png](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/gradcam_A2366_ritim_to_iletim.png)
- [gradcam_E10163_normal_to_iletim.png](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/gradcam_E10163_normal_to_iletim.png)

Özet dosyası:

- [gradcam_summary_test.json](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/visuals/gradcam_summary_test.json)

## Grad-CAM Yorumu

Bu çıktılar şu işler için kullanılabilir:

- modelin doğru örneklerde hangi lead ve zaman pencerelerine odaklandığını görmek
- yanlış sınıflarda dikkat odağının kayıp kaymadığını incelemek
- `ritim -> normal` ve `ritim -> iletim` gibi zor kararlarda modelin hangi sinyal parçalarını öne çıkardığını karşılaştırmak

Pratik kullanım:

- aynı hata tipindeki baseline ve `resnet_se` Grad-CAM çıktıları yan yana incelenirse mimari farkı yorumlanabilir
- özellikle `A2366` gibi ortak hata örneği iki model arasında doğrudan karşılaştırmaya uygundur

## Nihai Karar

Bu 50-epoch test turunun sonunda yeni en iyi model:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_test50_resnet_se/training_report.json)

Gerekçe:

- en yüksek validation `Macro F1`
- en yüksek test `Macro F1`
- en yüksek genel accuracy
- confusion matrix üzerinde özellikle `ritim` sınıfında daha temiz ayrım

## Sonuç

50 epoch ve farklı parametrelerle yapılan testler tamamlandı. Confusion matrix ve Grad-CAM çıktıları üretildi. Bu turdaki en güçlü model `50 epoch resnet_se + cosine` koşusudur.
