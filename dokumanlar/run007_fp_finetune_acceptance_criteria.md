# Run007 FP Fine-Tuning Acceptance Criteria

Tarih: 24 Mart 2026

Bu not, `run007` modeli uzerinde yapilacak false positive odakli fine-tuning turu icin basari kriterlerini tanimlar.

Temel aday model:

- [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans)

## 1. Baslangic Referansi

Run007 test confusion matrix:

```text
             Tahmin
           normal  ritim  iletim
Gerçek
normal       1357     81       5
ritim          82   1831       3
iletim         11     13     165
```

Buradan gelen mevcut FP degerleri:

- `FP(normal) = 93`
  - `ritim -> normal = 82`
  - `iletim -> normal = 11`
- `FP(ritim) = 94`
  - `normal -> ritim = 81`
  - `iletim -> ritim = 13`
- `FP(iletim) = 8`
  - `normal -> iletim = 5`
  - `ritim -> iletim = 3`

Mevcut temel metrikler:

- test `Macro F1 = 0.934375`
- test `iletim F1 = 0.911602`
- test precision:
  - `normal = 0.935862`
  - `ritim = 0.951169`
  - `iletim = 0.953757`

## 2. Ana Fine-Tuning Hedefi

Ana hedef:

- `normal` ve `ritim` siniflari arasindaki false positive hareketini azaltmak

Birincil odak hucreler:

- `normal -> ritim`
- `ritim -> normal`

Ikincil odak hucreler:

- `normal -> iletim`
- `ritim -> iletim`

## 3. Basari Kriterleri

Bir fine-tune kosusu, asagidaki kosullardan cogu saglandiginda basarili kabul edilir.

### 3.1 Birincil Kabul Kriteri

- `normal -> ritim` mevcut `81` degerinden dusmeli
- `ritim -> normal` mevcut `82` degerinden dusmeli

En azindan:

- bu iki hucrenin toplami `163` degerinden anlamli sekilde daha asagi inmelidir

Pratik hedef:

- toplamin `<= 150` olmasi guclu bir iyilesme kabul edilir

### 3.2 Toplam Performans Koruma Kriteri

- test `Macro F1` mevcut `0.934375` degerini ciddi bicimde kaybetmemeli

Kabul siniri:

- test `Macro F1 >= 0.932`

Bu sinirin altina dusus, FP azalmasi olsa bile dikkatli degerlendirilmelidir.

### 3.3 Iletim Koruma Kriteri

`iletim` sinifi bu modelin guclu tarafi oldugu icin bozulmamali.

Kabul siniri:

- test `iletim F1 >= 0.900`

Tercih edilen hedef:

- test `iletim F1 >= 0.905`

### 3.4 Precision Iyilesme Kriteri

Ozellikle:

- `normal precision` mevcut `0.935862` degerini korumali veya artirmali
- `ritim precision` mevcut `0.951169` degerini korumali veya artirmali

Guclu sonuc:

- iki precision degerinden en az biri anlamli bicimde yukari giderken digeri bozulmuyorsa

## 4. Basarisiz Sayilacak Durumlar

Asagidaki durumlar olursa kosu basarisiz veya en azindan zayif kabul edilir:

- `normal -> ritim` azalirken `ritim -> normal` sert artarsa
- `ritim -> normal` azalirken `normal -> ritim` sert artarsa
- test `Macro F1 < 0.932`
- test `iletim F1 < 0.900`
- `iletim` precision veya recall ciddi sekilde bozulursa

## 5. Karar Sirasi

Fine-tune kosulari su siraya gore degerlendirilmelidir:

1. test `Macro F1` korundu mu
2. `iletim F1` korundu mu
3. `normal -> ritim` ve `ritim -> normal` toplami dustu mu
4. `normal precision` ve `ritim precision` iyilesti mi
5. confusion matrix genel dengesi daha temiz mi

## 6. En Iyi Sonuc Nasil Tanimlanacak

Bir kosu asagidaki tabloya yaklasiyorsa yeni lider aday sayilabilir:

- test `Macro F1 >= 0.934`
- test `iletim F1 >= 0.905`
- `normal -> ritim + ritim -> normal < 163`
- ideal olarak `<= 150`

## 7. Son Not

Bu fine-tuning turunun amaci yeni bir radikal model degil, su sonucu elde etmektir:

- run007’nin `iletim` gucunu koruyup
- `normal` ve `ritim` arasindaki gereksiz false positive hareketini azaltmak

Yani hedef:

- daha temiz confusion matrix
- benzer toplam skor
- daha kullanisli final model
