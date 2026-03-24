# Run007 Oncelikli Iyilestirme Plani

Tarih: 24 Mart 2026

Bu not, [run007_referans](/home/alper/ekg/secili_model_adaylari/run007_referans) modeli uzerinde bundan sonraki iyilestirme islerinin onem sirasina gore hangi sirayla ele alinacagini tanimlar.

## Neden Yeni Bir Plan Gerekiyor

Son fine-tuning turu, `run007` uzerindeki kisa resume egitimlerinin modeli net bicimde gecemedigini gosterdi:

- test `Macro F1` korunurken `normal <-> ritim` hatalarini belirgin azaltan kosu cikmadi
- `normal -> ritim` ve `ritim -> normal` toplami referans modelde `163`
- `iletim` sinifi guclu kaldigi icin genel kayip daha cok `normal` ve `ritim` sinirinda olusuyor

Bu nedenle sonraki adim daha fazla genel fine-tuning degil, hata kaynagini ayrintili ayristirmak olmalidir.

## Onem Sirasi

### 1. Kritik hata havuzunu veri duzeyinde ayristir

Ilk hedef:

- `normal -> ritim`
- `ritim -> normal`

hatalarini kayit, tani kodu, yas, cinsiyet ve model guveni bazinda cikarmak.

Neden birinci oncelik:

- En buyuk FP/FN hareketi bu iki hucrede
- Kisa fine-tuning burada net kazanc getirmedi
- Veri alt-tipleri gorulmeden yeni bir sweep acmak verimsiz olur

Beklenen cikti:

- bucket bazli CSV listeleri
- top tani kodlari
- en yuksek guvenli yanlislar
- validation ve test tarafinda ortak hata deseni

### 2. Hata kodlarina gore veri odakli strateji tasarla

Birinci adimdan sonra su karar verilecek:

- Hangi SNOMED kodlari `normal -> ritim`te yogunlasiyor
- Hangi SNOMED kodlari `ritim -> normal`te yogunlasiyor
- Bu kodlar egitim verisinde az temsil mi ediliyor
- Bu kodlarin bir kismi etiket sinirinda dogal olarak belirsiz mi

Muhtemel aksiyonlar:

- hedefli oversampling
- hata havuzuna ozel mini-train subset
- kod-temelli dengeleme
- belirsiz etiket denetimi

### 3. Loss ve sampling tarafini sadece sorunlu sinif cifti icin ayarla

Ancak ikinci adimdan sonra uygulanmali.

Hedef:

- tum modeli tekrar aramak yerine
- `normal` ve `ritim` ayrimini daha iyi ogrenmesi icin egitim recetesini ayarlamak

Muhtemel yonler:

- `normal/ritim` agirlikli sampler
- hatali kayitlar icin hard example replay
- sinif cifti odakli loss agirligi

### 4. Kalibrasyon ve karar katmani

Eger ana hata yuksek guvenli yanlislar ise:

- temperature scaling
- logit calibration
- sinif bazli karar analizi

ozellikle `normal` ve `ritim` arasinda faydali olabilir.

Bu adim dorduncu oncelik olmali; cunku once hata havuzunun veri ve kod yapisi gorulmeli.

## Uygulanacak Ilk Adim

Bu planin ilk uygulanacak parcasi icin ayri script:

- [analyze_run007_priority_errors.py](/home/alper/ekg/scripts/analyze_run007_priority_errors.py)

Amaci:

- `run007` checkpoint’ini yuklemek
- validation ve test splitlerinde tum tahminleri yeniden hesaplamak
- oncelikli hata bucket’larini ayristirmak
- CSV ve JSON ciktilarini tek klasorde toplamak

Beklenen artefact koku:

- [run007_priority_error_analysis](/home/alper/ekg/artifacts/run007_priority_error_analysis)

## Bu Adimdan Sonra Verilecek Karar

Bu analiz tamamlandiktan sonra bir sonraki teknik karar su olacak:

1. veri odakli mini-sweep mi
2. hard-example retraining mi
3. normal/ritim ciftine ozel loss/sampler mi
4. calibration mi

Bu karar yeni bir genel sweep ile degil, analiz ciktilarindaki baskin hata kodlarina gore verilecektir.
