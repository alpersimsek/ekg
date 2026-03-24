# Sprint 4 Hata Analizi ve İyileştirme Raporu

## Amaç

Sprint 4 hedefi, Sprint 3 baz modelinin hata desenlerini görünür hale getirmek ve ölçülebilir iyileştirme adaylarını deneysel olarak karşılaştırmaktı.

Bu sprint kapsamında:

- baz model için kayıt bazlı hata analizi üretildi
- baskın hata çiftleri çıkarıldı
- `iletim` false positive örnekleri analiz edildi
- `ritim -> normal` kayan örnekler analiz edildi
- iki farklı eğitim varyasyonu baseline ile karşılaştırıldı

## Üretilen Kod ve Artifact’lar

### Kod

- [analyze_uc_sinif_errors.py](/home/alper/ekg/scripts/analyze_uc_sinif_errors.py)
- [train_uc_sinif_baseline.py](/home/alper/ekg/scripts/train_uc_sinif_baseline.py)

Not:

- Eğitim betiği Sprint 4’te `weighted sampler` ve `class-weight-mode` parametrelerini destekleyecek şekilde genişletildi.

### Analiz ve Deney Çıktıları

- [error_analysis.json](/home/alper/ekg/artifacts/uc_sinif_analysis/error_analysis.json)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_baseline/training_report.json)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_exp_weighted_sampler_balanced/training_report.json)
- [training_report.json](/home/alper/ekg/artifacts/uc_sinif_exp_weighted_sampler_unweighted/training_report.json)

## Baseline Hata Analizi

Analiz edilen model:

- [best_model.pt](/home/alper/ekg/artifacts/uc_sinif_baseline/best_model.pt)

### Validation Üzerindeki En Büyük Hata Çiftleri

| Gerçek | Tahmin | Adet |
| --- | --- | ---: |
| `ritim` | `normal` | `288` |
| `ritim` | `iletim` | `83` |
| `normal` | `ritim` | `42` |
| `normal` | `iletim` | `31` |
| `iletim` | `normal` | `10` |
| `iletim` | `ritim` | `9` |

### Test Üzerindeki En Büyük Hata Çiftleri

| Gerçek | Tahmin | Adet |
| --- | --- | ---: |
| `ritim` | `normal` | `301` |
| `ritim` | `iletim` | `49` |
| `normal` | `ritim` | `41` |
| `normal` | `iletim` | `37` |
| `iletim` | `ritim` | `8` |
| `iletim` | `normal` | `8` |

Ana gözlem:

- En baskın hata `ritim -> normal` kaymasıdır.
- `iletim` tarafında ana problem recall değil precision’dır.
- Yani model `iletim` vakalarını büyük ölçüde yakalıyor, fakat bazı `normal` ve `ritim` örneklerini yanlış biçimde `iletim`e itiyor.

## Tanı Kodu Örüntüleri

### Validation: `ritim -> normal` içinde sık görülen kodlar

- `427393009`
- `426177001`
- `284470004,284470004`
- `284470004`
- `164884008`
- `164889003`

Yorum:

- Bu kümede özellikle sinüs düzensizliği, sinüs bradikardisi ve erken vurular öne çıkıyor.
- Model bu örneklerin bir kısmını morfolojik olarak `normal`e fazla yakın görüyor olabilir.

### Validation: `iletim` false positive içinde sık görülen gerçek kodlar

- `426783006`
- `164884008`
- `164889003`
- `284470004`
- `284470004,284470004`
- `426177001`

### Test: `ritim -> normal` içinde sık görülen kodlar

- `426177001`
- `427393009`
- `164889003`
- `284470004`
- `164884008`

### Test: `iletim` false positive içinde sık görülen gerçek kodlar

- `426783006`
- `164884008`
- `164889003`
- `426177001`

Ana yorum:

- `normal` kayıtların bir kısmı `iletim`e kayıyor, fakat daha kritik olan şey bazı ritim kodlarının iletim gibi görünmesi.
- Özellikle `164884008`, `284470004` ve bazı `AFIB/SB/SA` örnekleri baz model için ayrıştırılması zor alt gruplar oluşturuyor.

## Deneyler

Sprint 4’te aşağıdaki iki deney koşuldu:

1. `weighted sampler + balanced class weight`
2. `weighted sampler + unweighted loss`

Karşılaştırma ölçütü:

- ana seçim metriği `validation Macro F1`

## Deney Karşılaştırması

| Deney | Val Macro F1 | Test Macro F1 | Val iletim Precision | Val iletim Recall | Test iletim Precision | Test iletim Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `0.825995` | `0.845055` | `0.597173` | `0.898936` | `0.667954` | `0.915344` |
| `weighted sampler + balanced` | `0.813443` | `0.832161` | `0.525223` | `0.941489` | `0.580328` | `0.936508` |
| `weighted sampler + unweighted` | `0.822227` | `0.851376` | `0.540625` | `0.920213` | `0.630824` | `0.931217` |

## Deney Sonuçlarının Yorumu

### 1. Weighted Sampler + Balanced Class Weight

Sonuç:

- `iletim recall` arttı
- fakat `iletim precision` belirgin düştü
- validation `Macro F1` baseline’ın altına indi

Karar:

- Bu konfigürasyon aşırı telafi etkisi yaratıyor
- mevcut veri ve kayıp kurgusunda tercih edilmemeli

### 2. Weighted Sampler + Unweighted Loss

Sonuç:

- test `Macro F1` en yüksek skor oldu: `0.851376`
- ancak validation `Macro F1`, baseline’ın altında kaldı: `0.822227 < 0.825995`
- `iletim recall` yüksek kalırken precision baseline’a göre düştü

Karar:

- umut verici ama henüz ana aday değil
- model seçimi validation metriğine göre yapılacağı için baseline’ın yerine geçirilmedi

## Sprint 4 Sonucu

Sprint 4 tamamlandı.

Bu sprintten çıkan ana kararlar:

1. En kritik hata paterni `ritim -> normal`.
2. `iletim` sınıfında ana sorun düşük recall değil, precision kaybı.
3. Weighted sampler tek başına problemi çözmüyor.
4. Baseline model validation `Macro F1` açısından hâlâ en güçlü referans model.

## Sprint 5 İçin Önerilen Odak

Bir sonraki sprintte aşağıdaki iyileştirmeler daha yüksek önceliğe sahip:

1. `ritim -> normal` örnekleri için hedefli augmentasyon veya sınıf-altı dengeleme
2. `iletim` false positive azaltmak için logit calibration veya decision-margin analizi
3. ritim alt tiplerine duyarlı daha güçlü mimari veya daha uzun eğitim
4. `164884008`, `284470004`, `426177001`, `427393009` gibi zor kod kümeleri için alt grup analizi

## Sonuç

Sprint 4 sonunda baz modelin hataları somutlaştırıldı ve en etkili görünen kısa vadeli iyileştirme adayları ölçüldü. Bu aşamada en savunulabilir karar, Sprint 5’e baseline modeli referans alarak girmek ve iyileştirmeleri `ritim -> normal` ile `iletim precision` sorunlarını doğrudan hedefleyecek biçimde tasarlamaktır.
