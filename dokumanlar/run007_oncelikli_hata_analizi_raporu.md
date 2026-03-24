# Run007 Oncelikli Hata Analizi Raporu

Tarih: 24 Mart 2026

Bu rapor, [analyze_run007_priority_errors.py](/home/alper/ekg/scripts/analyze_run007_priority_errors.py) ile [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans) modeli uzerinde uretilen hata ayrisimini ozetler.

Analiz cikti kok klasoru:

- [run007_priority_error_analysis](/home/alper/ekg/artifacts/run007_priority_error_analysis)

Ana JSON raporu:

- [priority_error_report.json](/home/alper/ekg/artifacts/run007_priority_error_analysis/priority_error_report.json)

## 1. Kisa Sonuc

Bu analizden cikan en onemli sonuc su:

- sonraki iyilestirme turu genel sweep olmamali
- `normal <-> ritim` ciftine odakli olmali
- hatta bu cift icinde de veri alt-tipleri hedeflenmeli

Test tarafinda iki baskin bucket:

- `normal -> ritim = 81`
- `ritim -> normal = 82`

Bu iki hucre tek basina ana hata yukunu tasiyor.

## 2. Test Ozet Metrikleri

Referans test metrikleri:

- test `Macro F1 = 0.934375`
- test confusion matrix:

```text
             Tahmin
           normal  ritim  iletim
Gercek
normal       1357     81       5
ritim          82   1831       3
iletim         11     13     165
```

## 3. En Kritik Iki Hata Havuzu

### 3.1 `normal -> ritim`

Test bulgulari:

- adet: `81`
- baskin kod: `426783006` (`81 / 81`)
- ortalama tahmin guveni: `0.7938`

Yorum:

- bu hata bucket’i neredeyse tamamen tek bir normal kod etrafinda toplaniyor
- yani sorun “tum normal sinifi” degil, belirli bir normal alt tipin ritim gibi gorunmesi

Onemli sonuc:

- burada hedefli hard-example stratejisi uygulanabilir
- genel loss degisikliginden daha verimli olma ihtimali yuksek

CSV:

- [test/normal_to_ritim.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/normal_to_ritim.csv)

### 3.2 `ritim -> normal`

Test bulgulari:

- adet: `82`
- baskin kodlar:
  - `427393009`: `55`
  - `426177001`: `9`
  - `284470004`: `9`
  - `164889003`: `4`
  - `427084000`: `4`
- ortalama tahmin guveni: `0.8314`

Yorum:

- bu bucket da rastgele dagilmiyor
- hatalar daha cok belirli ritim alt tiplerinde toplanmis durumda
- ustelik model bu yanlislarin buyuk kismini yuksek guvenle yapiyor

Onemli sonuc:

- burada sadece daha fazla epoch acmak yerine
- bu kodlara hedefli replay veya pair-aware sampler daha mantikli

CSV:

- [test/ritim_to_normal.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/ritim_to_normal.csv)

## 4. Ikincil Bucket’lar

### `ritim -> iletim`

- adet: `3`
- baskin kodlar: `284470004`, `164884008`, `427084000`
- ortalama tahmin guveni: `0.9582`

Not:

- hacim dusuk
- ama modelin burada yaptigi yanlislar cok guvenli
- ana oncelik olmasa da calibration veya vaka-ozel inceleme icin uygun

CSV:

- [test/ritim_to_iletim.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/ritim_to_iletim.csv)

### `normal -> iletim`

- adet: `5`
- baskin kod: `426783006`
- ortalama tahmin guveni: `0.7058`

Not:

- hacim dusuk
- `iletim` precision halen guclu
- bu bucket ilk tur iyilestirme hedefi olmamali

CSV:

- [test/normal_to_iletim.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/normal_to_iletim.csv)

## 5. Validation ve Test Tutarliligi

Validation tarafinda da ayni patern korunuyor:

- `normal -> ritim = 79`
- `ritim -> normal = 83`

Bu onemli; cunku gordugumuz hata deseni testte rastgele olusmus gorunmuyor.

Validation tarafinda da:

- `normal -> ritim` yine sadece `426783006`
- `ritim -> normal` yine agirlikla `427393009`, `284470004`, `426177001`

Bu nedenle bu hata alani sweep secim yan etkisi degil, modelin gercek karar siniri sorunu.

## 6. Onem Sirasi

Bu analizden sonra sonraki adimlar su sirada ele alinmali:

1. `normal -> ritim` bucket’i icin `426783006` odakli hard-example stratejisi
2. `ritim -> normal` bucket’i icin `427393009`, `426177001`, `284470004` odakli replay/sampler
3. Gerekirse bu iki bucket icin pair-aware loss veya ornekleyici
4. Daha sonra calibration ve dusuk hacimli `iletim` yanlislari

## 7. Uretilen Ciktilar

Split bazli CSV’ler:

- [dogrulama/all_misclassified.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/dogrulama/all_misclassified.csv)
- [test/all_misclassified.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/all_misclassified.csv)
- [dogrulama/top_confidence_errors.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/dogrulama/top_confidence_errors.csv)
- [test/top_confidence_errors.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/top_confidence_errors.csv)

Bucket bazli test CSV’leri:

- [test/normal_to_ritim.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/normal_to_ritim.csv)
- [test/ritim_to_normal.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/ritim_to_normal.csv)
- [test/ritim_to_iletim.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/ritim_to_iletim.csv)
- [test/normal_to_iletim.csv](/home/alper/ekg/artifacts/run007_priority_error_analysis/test/normal_to_iletim.csv)

## 8. Karar

Bu rapordan sonra en savunulabilir sonraki teknik adim:

- `run007` icin hedefli, bucket-aware bir yeni sweep tasarlamak
- ama bunu tum veri uzerinde degil
- `426783006` ve `427393009/426177001/284470004` agirlikli bir hata odagi ile yapmak

Daha net soylemek gerekirse:

- problem artik “model daha fazla egitilsin mi” problemi degil
- “hangi hata alt-kumesi nasil hedeflenecek” problemine donusmus durumda
