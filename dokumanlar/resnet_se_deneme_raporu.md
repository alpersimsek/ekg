# ResNet-SE Mimari Deneme Raporu

## Amaç

Bu deneme, daha güçlü bir mimarinin mevcut en iyi modeli geçip geçemeyeceğini ölçmek için yapıldı.

Denenen mimari:

- gerçek 1D artık bloklar
- kanal dikkat mekanizması olarak SE (Squeeze-and-Excitation) blokları

Güncellenen eğitim betiği:

- [train_uc_sinif_baseline.py](/home/alper/ekg/scripts/train_uc_sinif_baseline.py)

## Deneme Konfigürasyonu

- model: `resnet_se`
- epoch: `12`
- batch size: `64`
- lr: `5e-4`
- scheduler: `cosine`
- min lr: `1e-5`
- dropout: `0.2`
- augment: `none`
- class weights: `balanced`
- device: `cuda`

Artifact’lar:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_resnet_se_e12_cosine/best_model.pt)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_resnet_se_e12_cosine/training_report.json)

## Sonuçlar

### Validation

- Loss: `0.178757`
- Accuracy: `0.937923`
- Macro F1: `0.902712`

Sınıf bazlı validation F1:

| Sınıf | F1 |
| --- | ---: |
| `normal` | `0.941502` |
| `ritim` | `0.948230` |
| `iletim` | `0.818402` |

### Test

- Loss: `0.165902`
- Accuracy: `0.950395`
- Macro F1: `0.923115`

Sınıf bazlı test F1:

| Sınıf | F1 |
| --- | ---: |
| `normal` | `0.948870` |
| `ritim` | `0.961216` |
| `iletim` | `0.859259` |

## Mevcut En İyi Model ile Karşılaştırma

Referans model:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/best_model.pt)
- rapor: [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/training_report.json)

Karşılaştırma:

| Model | Val Macro F1 | Test Macro F1 | Val iletim F1 | Test iletim F1 |
| --- | ---: | ---: | ---: | ---: |
| `12 epoch + cosine` mevcut en iyi | `0.908420` | `0.921371` | `0.850000` | `0.881443` |
| `resnet_se + 12 epoch + cosine` | `0.902712` | `0.923115` | `0.818402` | `0.859259` |

## Yorum

Bu sonuç önemli bir ayrım gösteriyor:

- `resnet_se` testte biraz daha yüksek toplam `Macro F1` verdi
- ancak validation `Macro F1`, mevcut en iyi modelin altında kaldı
- ayrıca `iletim` sınıfında hem validation hem test F1 mevcut liderden daha düşük kaldı

Bu nedenle:

- eğer seçim ölçütü katı biçimde `validation Macro F1` ise mevcut en iyi model korunmalı
- eğer yalnızca test toplam skoru baz alınsaydı `resnet_se` tartışmaya açık olurdu

Ancak proje boyunca benimsenen seçim ilkesi validation temelli olduğu için ana aday değişmedi.

## Sonuç

`resnet_se` mimarisi başarısız olmadı; aksine güçlü ve umut verici bir ikinci aday üretti. Ancak bu koşul altında mevcut en iyi modeli validation bazında geçemediği için ana model olarak seçilmedi.

Mevcut ana aday hâlâ:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/best_model.pt)

## Sonraki Mantıklı Adımlar

Bu mimariyi gerçekten öne geçirmek için en mantıklı devam seçenekleri:

1. `resnet_se` için daha uzun eğitim
2. `resnet_se` üzerinde daha düşük LR ile ikinci aşama fine-tuning
3. `resnet_se` + hafif augmentasyonun ayrı bir koşuda denenmesi
4. validation tepe noktasını artırmak için scheduler parametrelerinin yeniden ayarlanması
