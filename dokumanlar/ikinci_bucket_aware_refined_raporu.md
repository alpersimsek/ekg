# Ikinci Bucket-Aware Refined Sweep Raporu

Tarih: 24 Mart 2026

Bu rapor, [ba05_ikinci_tur_bucket_aware_plani.md](/home/alper/ekg/dokumanlar/ba05_ikinci_tur_bucket_aware_plani.md) dogrultusunda kosulan ikinci daraltma turunu ozetler.

Sweep cikti kok klasoru:

- [uc_sinif_bucket_aware_refined](/home/alper/ekg/artifacts/uc_sinif_bucket_aware_refined)

Toplu ozet:

- [bucket_aware_refined_summary.json](/home/alper/ekg/artifacts/uc_sinif_bucket_aware_refined/bucket_aware_refined_summary.json)

## 1. Amaç

Bu turun hedefi:

- `ba05` ile elde edilen `normal <-> ritim` kazancini korumak
- ayni zamanda `iletim F1` kaybini geri almak

Yani aranan profil:

- test `Macro F1` yuksek
- test `iletim F1 >= 0.900`
- kritik toplam (`normal -> ritim + ritim -> normal`) dusuk

## 2. Kosulan Tur

Toplam kosu:

- `8`

Deney eksenleri:

- `focus_normal_weight`: `1.5`, `1.75`, `2.0`
- `focus_ritim_weight`: `1.5`, `1.75`, `2.0`
- `augment`: `light` veya `none`
- secili kosuda daha dusuk `weight_decay`

## 3. En Guclu Sonuclar

### Toplam test lideri: `rba06`

- [rba06_bucket_aware_tepe](/home/alper/ekg/secili_model_adaylari/rba06_bucket_aware_tepe)
- test `Macro F1 = 0.937708`
- test `iletim F1 = 0.898072`
- `normal -> ritim = 62`
- `ritim -> normal = 60`
- kritik toplam `= 122`

Yorum:

- bu turdaki en yuksek toplam test skoru
- kritik toplam da hala belirgin bicimde dusuk
- ama `iletim F1` hedef `0.900` cizgisinin biraz altinda

### En dengeli aday: `rba02`

- [rba02_bucket_aware_dengeli](/home/alper/ekg/secili_model_adaylari/rba02_bucket_aware_dengeli)
- test `Macro F1 = 0.937501`
- test `iletim F1 = 0.905556`
- `normal -> ritim = 58`
- `ritim -> normal = 79`
- kritik toplam `= 137`

Yorum:

- `iletim F1` tekrar `0.900+` bandina cikti
- toplam test skoru da cok guclu
- ama kritik toplam `rba06` kadar iyi degil

## 4. Dikkat Ceken Bulgular

### 4.1 Daha yumusak agirliklar daha iyi calisti

Ilk bucket-aware turunda agirliklar daha sertti.

Ikinci turda:

- `1.5 / 2.0`
- `1.5 / 1.5`
- `1.75 / 1.75`

gibi yumusak agirliklar daha iyi sonuc verdi.

Bu suyu gosteriyor:

- bucket-aware fikir dogru
- ama asiri agresif sampler agirligi `iletim` performansini gereksiz bozuyor

### 4.2 `light augment` yine faydali

En guclu iki kosu:

- `rba06`
- `rba02`

ikisinde de `augment = light`

Bu, bu bolgede `light augment`in korunmasi gerektigini destekliyor.

### 4.3 `weight_decay` ufak ayari katkili olabilir

`rba06` ile `rba02` arasindaki temel fark:

- ayni agirlik profili
- `rba06` icin daha dusuk `weight_decay`

Bu da:

- ayni sampler dengesinde ufak regularization ayarinin test tavanina katkisi olabilecegini
- ama bunun `iletim` dengesini biraz bozabildigini gosturuyor

## 5. Referanslarla Karsilastirma

### `run007`

- test `Macro F1 = 0.934375`
- test `iletim F1 = 0.911602`
- kritik toplam `= 163`

### `ba05`

- test `Macro F1 = 0.934890`
- test `iletim F1 = 0.887052`
- kritik toplam `= 116`

### `rba02`

- test `Macro F1 = 0.937501`
- test `iletim F1 = 0.905556`
- kritik toplam `= 137`

### `rba06`

- test `Macro F1 = 0.937708`
- test `iletim F1 = 0.898072`
- kritik toplam `= 122`

## 6. Sonuc

Bu ikinci turdan cikan ana yargi:

- bucket-aware yon dogru
- `ba05` ara istasyondu
- ikinci turda hem toplam skor hem de `iletim` dengesi ileri tasindi

Su anda:

- saf toplam performans icin `rba06`
- daha dengeli genel aday icin `rba02`

one cikiyor.

## 7. Teknik Karar Onerisi

Eger tek bir gelistirme adayi secilecekse:

- varsayilan olarak `rba02` daha savunulabilir

Neden:

- test `Macro F1` cok guclu
- test `iletim F1` tekrar `0.900+`
- kritik toplam hala `run007`den anlamli sekilde daha iyi

Eger sadece toplam test tavanini optimize etmek istenirse:

- `rba06` dikkate alinabilir

Ama `rba06` icin:

- `iletim F1` biraz daha dusuk kaldigi icin
- dikkatli karar verilmelidir
