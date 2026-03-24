# BA05 Ikinci Tur Bucket-Aware Plani

Tarih: 24 Mart 2026

Bu not, [ba05_bucket_aware](/home/alper/ekg/secili_model_adaylari/ba05_bucket_aware) kosusu etrafinda tasarlanacak ikinci daraltma turunu tanimlar.

## Neden Ikinci Tur Gerekiyor

`ba05` su iki konuda net kazanc verdi:

- test `Macro F1 = 0.934890`
- kritik `normal -> ritim + ritim -> normal` toplami `= 116`

Ama bedeli:

- test `iletim F1 = 0.887052`

Bu nedenle yeni hedef:

- `normal <-> ritim` kazancini korumak
- ama `iletim F1`i tekrar `0.900+` bandina yaklastirmak

## Hipotez

Ilk bucket-aware turdaki `iletim` kaybi muhtemelen su iki nedenle olustu:

1. `normal` ve `ritim` bucket’lari fazla agirliklandigi icin `iletim` sinifi goreli olarak geri plana itildi
2. `light augment` ile hedefli sampler birlikte calisirken karar siniri `normal/ritim` tarafina fazla kaydi

Bu nedenle ikinci turda:

- hedef agirliklar yumusatilmali
- `ba05`in dogru yonu korunmali
- ama `iletim` tarafina nefes birakilmali

## Daraltilmis Deney Alani

Merkez kosu:

- `focus_normal_weight = 2.0`
- `focus_ritim_weight = 2.0`
- `augment = light`
- `weight_decay = 1e-4`

Ikinci turda asagidaki eksenler denenecek:

- daha yumusak agirliklar:
  - `1.5 / 1.5`
  - `1.5 / 2.0`
  - `2.0 / 1.5`
  - `1.75 / 1.75`
- sinirli regularization varyasyonu
- secili kosularda `augment` ac/kapa

Bu turda:

- yeni mimari yok
- yeni genel hypergrid yok
- sadece `ba05` etrafinda daraltma var

## Basari Kriterleri

Yeni aday model su hedefe olabildigince yaklasmali:

- test `Macro F1 >= 0.934`
- test `iletim F1 >= 0.900`
- kritik toplam `<= 130`

Ideal aday:

- `ba05`e yakin `Macro F1`
- `run007`ye yakin `iletim F1`
- `run007`den anlamli derecede dusuk kritik toplam

## Ayrı Script

Bu ikinci tur icin ayri script:

- [run_uc_sinif_bucket_aware_refined_sweep.py](/home/alper/ekg/scripts/run_uc_sinif_bucket_aware_refined_sweep.py)

Bu script:

- mevcut bucket-aware trainer’i kullanir
- ayri artefact klasorune yazar
- her kosu sonunda confusion matrix ve Grad-CAM uretir

## Beklenen Karar

Bu turun sonunda iki durumdan biri secilecek:

1. `ba05`den daha dengeli bir aday bulunursa yeni gelistirme adayi o olur
2. bulunamazsa `ba05`, bucket-aware yonun dogru oldugunu gosteren ama son hali olmayan ara aday olarak kalir
