# Run007, BA05, RBA02, RBA06 Karar Karsilastirmasi

Tarih: 24 Mart 2026

Bu not, su dort aday arasindaki teknik secim tablosunu ozetler:

- [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans)
- [ba05_bucket_aware](/home/alper/ekg/secili_model_adaylari/ba05_bucket_aware)
- [rba02_bucket_aware_dengeli](/home/alper/ekg/secili_model_adaylari/rba02_bucket_aware_dengeli)
- [rba06_bucket_aware_tepe](/home/alper/ekg/secili_model_adaylari/rba06_bucket_aware_tepe)

## Sayisal Ozet

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

## Yorum

`run007`:

- en guvenli klasik referans
- `iletim` sinifini en iyi koruyan aday
- ama `normal <-> ritim` karisimi en zayif aday

`ba05`:

- bucket-aware yonun ise yaradigini kanitladi
- kritik toplamda cok guclu dusus getirdi
- ama `iletim` kaybi fazla oldu

`rba02`:

- bucket-aware avantajini koruyor
- `iletim F1`i tekrar `0.900+` bandina tasiyor
- toplam test skorunda da belirgin guclu
- bu nedenle en dengeli yeni aday

`rba06`:

- toplam test skorunda lider
- kritik toplam da cok iyi
- ama `iletim F1` `rba02`den dusuk

## Hangi Durumda Hangisi Secilir

### `run007`

Secilir eger:

- `iletim` sinifi mutlak oncelikse
- en konservatif karar isteniyorsa

### `rba06`

Secilir eger:

- birinci hedef toplam test `Macro F1` ise
- `iletim`deki kucuk kayip kabul edilebiliyorsa

### `rba02`

Secilir eger:

- hem toplam skor hem de `iletim` dengesi birlikte onemliyse
- bucket-aware stratejiyi yeni ana aday yapmak isteniyorsa

## Teknik Oneri

Su an tek bir yeni ana gelistirme adayi secilecekse:

- `rba02`

en savunulabilir secimdir.

Gerekce:

- `run007`ye gore daha yuksek test `Macro F1`
- `run007`ye gore daha dusuk kritik toplam
- `ba05` ve `rba06`a gore daha iyi `iletim` dengesi

Kisa karar cumlesi:

- final referans olarak `run007` tutulabilir
- ama yeni nesil bucket-aware ana aday olarak `rba02` one cikiyor
