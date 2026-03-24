# Run007 vs BA05 Karar Notu

Tarih: 24 Mart 2026

Bu not, referans model [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans) ile bucket-aware turun en iyi kosusu [ba05_bucket_aware](/home/alper/ekg/secili_model_adaylari/ba05_bucket_aware) arasindaki secim kararini ozetler.

## Sayisal Karsilastirma

### `run007`

- test `Macro F1 = 0.934375`
- test `iletim F1 = 0.911602`
- `normal -> ritim = 81`
- `ritim -> normal = 82`
- kritik toplam `= 163`

### `ba05`

- test `Macro F1 = 0.934890`
- test `iletim F1 = 0.887052`
- `normal -> ritim = 48`
- `ritim -> normal = 68`
- kritik toplam `= 116`

## Yorum

`ba05` su iki konuda acik sekilde daha iyi:

- toplam test `Macro F1`
- `normal <-> ritim` karisiminin azaltimi

Ama `run007` su konuda daha iyi:

- `iletim F1`

Farkin anlami:

- `ba05`, modeli genel olarak daha iyi dengeliyor
- ama `iletim` sinifindaki onceki guclu dengeyi tam koruyamiyor

## Hangi Durumda Hangisi Secilir

### `ba05` secilir eger

- ana operasyonel sorun `normal` ile `ritim` karisimi ise
- confusion matrix dengesine toplam olarak bakiliyorsa
- `iletim` tarafindaki sinirli dusus kabul edilebilirse

### `run007` secilir eger

- `iletim` sinifi daha hassas onemdeyse
- onceki dengeli performansi korumak isteniyorsa
- bucket-aware stratejinin ikinci turu daha yapilmadan karar verilecekse

## Teknik Tavsiye

Su anda dogrudan kalici model degisimi yapmak yerine:

- `run007` referans olarak korunmali
- `ba05` ise gelistirme adayi olarak tutulmali

Neden:

- `ba05` iyi bir yon gosteriyor
- ama `iletim F1` kaybi, modeli hemen final adayina cevirmek icin biraz buyuk

Bu nedenle en savunulabilir karar:

- final referans olarak `run007`
- sonraki iyilestirme tabani olarak `ba05`
