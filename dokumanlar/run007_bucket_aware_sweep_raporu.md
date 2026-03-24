# Run007 Bucket-Aware Sweep Raporu

Tarih: 24 Mart 2026

Bu rapor, [run007_oncelikli_hata_analizi_raporu.md](/home/alper/ekg/dokumanlar/run007_oncelikli_hata_analizi_raporu.md) sonrasinda tasarlanan bucket-aware egitim turunun sonucunu ozetler.

Sweep cikti kok klasoru:

- [uc_sinif_bucket_aware_sweep](/home/alper/ekg/artifacts/uc_sinif_bucket_aware_sweep)

Toplu ozet:

- [bucket_aware_summary.json](/home/alper/ekg/artifacts/uc_sinif_bucket_aware_sweep/bucket_aware_summary.json)

## 1. Amaç

Bu turun hedefi, genel Macro F1 performansini koruyarak veya hafif artirarak su iki kritik hata hucresini dusurmekti:

- `normal -> ritim`
- `ritim -> normal`

Temel fikir:

- `run007` modelinde baskin hata ureten kodlari egitim sirasinda daha sik gostermek
- ama mevcut basarili tarifeyi tamamen bozmamak

Hedef kodlar:

- `normal -> ritim`: `426783006`
- `ritim -> normal`: `427393009`, `426177001`, `284470004`

## 2. Kosulan Tur

Kosulan toplam deneme:

- `8`

Her kosuda:

- `50 epoch`
- `baseline` mimarisi
- `lr = 5e-4`
- `scheduler = cosine`
- `sampler = weighted`
- confusion matrix ve Grad-CAM uretimi

Degisen kisim:

- hedef `normal` kodlari icin sampler agirligi
- hedef `ritim` kodlari icin sampler agirligi
- `augment`
- bazi kosularda `weight_decay`

## 3. Referans Model

Referans:

- [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans)

Referans test sonuclari:

- test `Macro F1 = 0.934375`
- test `iletim F1 = 0.911602`
- `normal -> ritim = 81`
- `ritim -> normal = 82`
- kritik toplam `= 163`

## 4. En Iyi Bucket-Aware Kosu

En iyi toplam test sonucu:

- [ba05_bucket_aware](/home/alper/ekg/secili_model_adaylari/ba05_bucket_aware)

Sonuc:

- validation `Macro F1 = 0.930157`
- test `Macro F1 = 0.934890`
- test `iletim F1 = 0.887052`
- `normal -> ritim = 48`
- `ritim -> normal = 68`
- kritik toplam `= 116`

Yani:

- test `Macro F1` referansa gore hafif yukselmis
- kritik hata toplami `163 -> 116` olarak ciddi sekilde dusmus

Ama:

- `iletim F1` referansa gore dusmus

## 5. Diger Dikkat Ceken Kosular

### `ba07_fn20_fr30_light_aug`

- [ba07_fn20_fr30_light_aug](/home/alper/ekg/artifacts/uc_sinif_bucket_aware_sweep/ba07_fn20_fr30_light_aug)
- test `Macro F1 = 0.934822`
- test `iletim F1 = 0.895442`
- `normal -> ritim = 50`
- `ritim -> normal = 81`
- kritik toplam `= 131`

Yorum:

- toplam skor guclu
- `iletim` kaybi `ba05`e gore daha sinirli
- ama `ritim -> normal` hucresinde belirgin iyilesme yok

### `ba03_fn20_fr30_none_aug`

- [ba03_fn20_fr30_none_aug](/home/alper/ekg/artifacts/uc_sinif_bucket_aware_sweep/ba03_fn20_fr30_none_aug)
- test `Macro F1 = 0.930469`
- test `iletim F1 = 0.889503`
- `normal -> ritim = 81`
- `ritim -> normal = 61`
- kritik toplam `= 142`

Yorum:

- `ritim -> normal`i azaltmis
- ama `normal -> ritim`i neredeyse hic azaltmamis

## 6. En Onemli Teknik Bulgular

### 6.1 Bucket-aware sampling ise yariyor

Bu turdan sonra artik net olarak soyleyebiliriz:

- hedefli sampler ile `normal <-> ritim` karisimi gercekten azaltilabiliyor

Ozellikle `ba05`:

- `normal -> ritim`i `81 -> 48`
- `ritim -> normal`i `82 -> 68`

indirerek ana hedefe dokunuyor.

### 6.2 Iletim sinifi bedel oduyor

Bu turdaki iyilesme ucretsiz gelmedi.

Referans `run007`:

- test `iletim F1 = 0.911602`

`ba05`:

- test `iletim F1 = 0.887052`

Bu da su anlama geliyor:

- model `normal` ve `ritim` ayrimini iyilestirirken
- `iletim` tarafindaki onceki dengeyi bir miktar kaybediyor

### 6.3 Light augmentation faydali gozukuyor

En iyi iki kosu:

- `ba05`
- `ba07`

ikisinde de `augment = light`

Bu, hedefli sampler ile hafif augmentasyonun birlikte daha yararli oldugunu dusunduruyor.

## 7. Karar Acisindan Ozet

Eger oncelik:

- toplam hata dengesi
- `normal <-> ritim` karisiminin ciddi azaltimi
- ve test `Macro F1`in korunmasi

ise `ba05` guclu bir adaydir.

Eger oncelik:

- `iletim` sinifini korumak
- ve mevcut dengeli yapidan fazla uzaklasmamak

ise `run007` halen daha guvenli adaydir.

## 8. Sonuc

Bu tur sonunda su iki yargi birlikte dogru:

1. bucket-aware yaklasim teknik olarak calisiyor
2. ama su anki haliyle `iletim` tarafinda maliyet yaratiyor

Yani sonraki en dogru adim:

- bucket-aware fikrini tamamen birakmak degil
- `iletim` kaybini daha kontrollu tutacak ikinci bir daraltma turu tasarlamak

En mantikli yeni odak:

- `ba05` tarifesini referans alip
- `iletim` kaybini azaltacak sekilde daha yumusak sampler / augment varyasyonu denemek
