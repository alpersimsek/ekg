# Fine-Tuning Deneme Raporu

## Amaç

Bu rapor, mevcut en iyi model üzerinde ek fine-tuning denemelerinin sonucunu özetler.

Başlangıç noktası:

- mevcut en iyi model: [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/best_model.pt)
- kaynak rapor: [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/training_report.json)

Başlangıç model performansı:

- validation `Macro F1 = 0.908420`
- test `Macro F1 = 0.921371`

## Denenen Fine-Tuning Yaklaşımı

Amaç, mevcut en iyi checkpoint’ten eğitime devam ederek daha düşük öğrenme oranı ile ek kazanım olup olmadığını test etmekti.

Bu kapsamda eğitim betiği `resume_checkpoint` desteği alacak şekilde güncellendi:

- [train_uc_sinif_baseline.py](/home/alper/ekg/scripts/train_uc_sinif_baseline.py)

## Deneme 1: Smoke Test

Konfigürasyon:

- `resume_checkpoint = uc_sinif_opt_exp3_e12_cosine/best_model.pt`
- `epochs = 1`
- `lr = 1e-4`
- `scheduler = cosine`
- `num_workers = 0`

Artifact:

- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_finetune_smoke/training_report.json)

Sonuç:

- validation `Macro F1 = 0.895028`
- test `Macro F1 = 0.900637`

Yorum:

- Tek epoch devam eğitimi bile mevcut en iyi modelden geri kaldı.
- Bu, modelin zaten tepeye çok yakın olduğunu ve devam eğitiminin hemen kazanç üretmediğini gösterdi.

## Deneme 2: Daha Yumuşak Fine-Tuning

Konfigürasyon:

- `resume_checkpoint = uc_sinif_opt_exp3_e12_cosine/best_model.pt`
- `epochs = 5`
- `lr = 2e-5`
- `scheduler = cosine`
- `num_workers = 0`

Artifact:

- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_finetune_e5_lr2e5/training_report.json)

En iyi epoch:

- `epoch = 4`

Sonuçlar:

- validation `Macro F1 = 0.907458`
- test `Macro F1 = 0.918375`

Sınıf bazlı sonuçlar:

| Bölüm | normal F1 | ritim F1 | iletim F1 |
| --- | ---: | ---: | ---: |
| Validation | `0.934471` | `0.942333` | `0.845570` |
| Test | `0.930618` | `0.948859` | `0.875648` |

Yorum:

- Sonuç güçlü kaldı
- ancak mevcut en iyi modelin validation `Macro F1 = 0.908420` seviyesini geçemedi
- test tarafında da `0.921371` seviyesinin altında kaldı

## Karşılaştırma

| Model | Val Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| `12 epoch + cosine` en iyi model | `0.908420` | `0.921371` |
| `fine-tune 1 epoch, lr=1e-4` | `0.895028` | `0.900637` |
| `fine-tune 5 epoch, lr=2e-5` | `0.907458` | `0.918375` |

## Sonuç

Fine-tuning denendi fakat mevcut en iyi modeli geçemedi.

Bu nedenle şu an için ana aday model değişmedi:

- en iyi model: [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/best_model.pt)
- rapor: [training_report.json](/home/alper/ekg/artifacts/uc_sinif_opt_exp3_e12_cosine/training_report.json)

## Teknik Yorum

Bu sonuç şu anlama geliyor:

- model, `12 epoch + cosine scheduler` koşusunda zaten güçlü bir optimuma ulaşmış
- aynı ağı daha düşük öğrenme oranı ile mevcut checkpoint’ten sürdürmek ek kazanç getirmedi
- bundan sonraki anlamlı sıçrama muhtemelen yalnızca daha fazla epoch ile değil, mimari değişim veya veri/özellik stratejisi değişimi ile gelecektir

## Önerilen Sonraki Adımlar

Eğer daha ileri geliştirme istenirse en yüksek olasılıklı yönler:

1. daha güçlü mimari: gerçek 1D ResNet veya channel attention
2. ritim alt tipleri için hedefli curriculum veya alt grup dengelemesi
3. logit calibration veya margin-based loss
4. hasta-bazlı split mümkünse yeniden üretim

## Nihai Karar

Fine-tuning denemesi tamamlandı ve ölçüldü. Şu anda en iyi model hâlâ `artifacts/uc_sinif_opt_exp3_e12_cosine` altındaki modeldir.
