# Sprint 1 Veri Denetimi ve Etiket Sözleşmesi Raporu

## Kapsam

Bu rapor, `/home/alper/ekg/data/data_3_Sinif` veri kümesi üzerinde Sprint 1 hedeflerinin tamamlanması için yapılan teknik denetimi ve etiket sözleşmesi kararlarını özetler.

Sprint 1 hedefi:

- veri kümesinin fiziksel bütünlüğünü doğrulamak
- 3 sınıflı hedef için etiket sözleşmesini netleştirmek
- eğitim dışında bırakılacak kayıt türlerini belirlemek
- Sprint 2 için kullanılacak veri manifesti şemasını tanımlamak

## Kullanılan Kaynaklar

- Veri kümesi: `/home/alper/ekg/data/data_3_Sinif`
- Nihai etiket eşleme tablosu: `/home/alper/ekg/yedek/20260323_182919/yeni_yaklasım/siniflar_final.csv`
- Önceki strateji notu: `/home/alper/ekg/yedek/20260323_182919/docs/uc_sinif_stratejisi.md`

## Önemli Karar

Sprint 1 sonunda etiket kaynağı olarak eski Python içi sabit kümeler değil, `siniflar_final.csv` dosyası kabul edilmiştir.

Gerekçe:

- veri kümesindeki SNOMED kapsamı bu tabloda daha güncel ve daha tamdır
- `exclude` gibi teknik dışlama kararları bu dosyada açıkça tanımlanmıştır
- eski sabit kümeler ile hesaplanan dağılım ile güncel tablo arasında fark vardır

Bu nedenle Sprint 2 ve sonrası için `siniflar_final.csv` tek gerçek kaynak olarak kullanılmalıdır.

## Teknik Veri Denetimi

### Dosya Bütünlüğü

- Toplam dosya sayısı: `125970`
- Toplam kayıt kimliği: `62985`
- `.hea` sayısı: `62985`
- `.mat` sayısı: `62985`
- Eksik `.hea` sayısı: `0`
- Eksik `.mat` sayısı: `0`
- Sıfır byte dosya sayısı: `0`

Sonuç:

- Veri kümesinde her kayıt için `.hea/.mat` çifti mevcuttur.
- Fiziksel dosya bütünlüğü açısından Sprint 1 kabul kriteri sağlanmıştır.

### Header Formatı

Header denetiminde tüm kayıtların hedef formatı sağladığı görülmüştür:

- Kanal sayısı: `12`
- Örnekleme hızı: `500 Hz`
- Örnek sayısı: `5000`
- Header format dağılımı: yalnızca `12 / 500 / 5000`

Sonuç:

- Model girdisi açısından veri kümesi standardı tutarlıdır.

### Header Anomalisi

Denetimde `1` adet bozuk ama kurtarılabilir header yapısı bulundu:

- Kayıt: `JS23074.hea`

Bu dosyada ilk satır standart WFDB başlığı ile ilk kanal tanımını birleştirmiş durumdadır. Örnek:

```text
S23074 12 500 5000 JS23074.mat 16+24 1000/mV 16 0 54 12431 0 I
```

Etkisi:

- Basit satır tabanlı parser bu dosyada kayabilir.
- Veri yükleyici bu tür dosyaları toleranslı biçimde işlemelidir.

Karar:

- Sprint 2 veri okuyucusu bu bozuk ilk satır varyasyonunu desteklemelidir.

### Sinyal Tensörü Doğrulaması

Rastgele seçilen `256` `.mat` dosyası yüklenerek `val` matrisi kontrol edildi.

- Beklenen boyut: `(12, 5000)`
- Sorunlu örnek sayısı: `0`

Sonuç:

- Örneklenen sinyal payload’larında yapısal bozulma tespit edilmedi.
- Sprint 2 öncesinde tam veri kümesi üzerinde tam payload doğrulaması opsiyonel bir ek kontrol olarak bırakılabilir; ancak Sprint 1 için örnekleme bazlı doğrulama yeterli kabul edilmiştir.

## Etiket Sözleşmesi

Sprint 1 sonunda aşağıdaki etiket sözleşmesi kabul edilmiştir.

### Hedef Sınıflar

- `normal`
- `ritim`
- `iletim`

### Dışlanan Kategoriler

- `morfolojik_iskemik`
- `exclude`

### Normal Tanımı

Bir kayıt yalnızca tanı kodu kümesi tam olarak `426783006` olduğunda `normal` kabul edilir.

Yani:

- saf sinüs ritmi ise `normal`
- `426783006` ile birlikte başka patolojik kod varsa `normal` verilmez

### Eğitim Kohortu Kuralı

Model eğitimi için kayıt seçim kuralı:

1. Kayıt yalnızca bir ana hedefe düşmeli: `normal` veya `ritim` veya `iletim`
2. `morfolojik_iskemik` etiketi taşımamalı
3. `exclude` etiketi taşımamalı

Bu sözleşme ile problem tek etiketli 3 sınıflı sınıflandırma olarak sabitlenmiştir.

## Sınıf Dağılımı

`siniflar_final.csv` kullanılarak veri kümesi aşağıdaki şekilde ayrışmıştır:

| Kategori | Kayıt Sayısı |
| --- | ---: |
| `normal` | 14419 |
| `ritim` | 19155 |
| `iletim` | 1884 |
| `morfolojik_iskemik` nedeniyle dışlanan | 27523 |
| `exclude` nedeniyle dışlanan | 4 |

Temiz 3 sınıflı eğitim kohortu toplamı:

- `35458` kayıt

Temiz kohort içindeki oranlar:

| Sınıf | Kayıt | Oran | Balanced Weight |
| --- | ---: | ---: | ---: |
| `normal` | 14419 | `%40.67` | `0.819705` |
| `ritim` | 19155 | `%54.02` | `0.617036` |
| `iletim` | 1884 | `%5.31` | `6.273531` |

Yorum:

- Ana sınıf dengesizliği `iletim` sınıfındadır.
- Sprint 2 ve Sprint 3 sırasında `Macro F1` zorunlu ana metrik olarak korunmalıdır.
- `accuracy` tek başına kullanılırsa modelin `ritim` lehine kayma riski yüksektir.

## Etiket Kapsamı

`siniflar_final.csv` ile denetim yapıldığında:

- Eşleme satırı sayısı: `116`
- Eşlenmeyen tıbbi kod sayısı: `0`

Tek istisna teknik dışlama kodudur:

- `251139008 = Suspect arm ECG leads reversed`

Bu kod `siniflar_final.csv` içinde `exclude` olarak tanımlanmıştır ve tıbbi sınıf hedeflerinden biri değildir.

## Sprint 1 Çıkışları

Sprint 1 sonunda aşağıdaki çıktılar tamamlanmış kabul edilmiştir:

1. Veri kümesinin dosya bütünlüğü doğrulandı.
2. Header standardı doğrulandı.
3. Tolerans gerektiren özel header anomalisi tespit edildi.
4. Etiket kaynağı olarak `siniflar_final.csv` seçildi.
5. Nihai 3 sınıflı eğitim sözleşmesi sabitlendi.
6. Temiz eğitim kohortunun sınıf dağılımı çıkarıldı.

## Sprint 2 İçin Teknik Notlar

Sprint 2 başlamadan önce aşağıdaki kararlar doğrudan uygulanmalıdır:

1. Veri yükleyici `JS23074.hea` benzeri birleşik ilk satırları tolere etmelidir.
2. Etiket üretimi `siniflar_final.csv` üzerinden yapılmalıdır.
3. `morfolojik_iskemik` ve `exclude` etiketli kayıtlar eğitim dışında bırakılmalıdır.
4. Çıktı formatı tek etiketli 3 sınıflı olmalıdır.
5. Eğitim ve model seçimi `validation Macro F1` ile yapılmalıdır.

## Sprint 2 İçin Manifest Şeması

Sprint 2’de üretilecek manifest için önerilen alanlar:

- `kayit_id`
- `hea_yolu`
- `mat_yolu`
- `hedef_sinif`
- `normal`
- `ritim`
- `iletim`
- `morfolojik_iskemik`
- `exclude`
- `bolum`
- `tani_kodlari`

## Sprint 1 Durum Sonucu

Sprint 1 tamamlanmıştır.

Durum özeti:

- Veri kümesi eğitim için fiziksel olarak kullanılabilir durumda.
- Etiket sözleşmesi netleştirilmiştir.
- Temiz 3 sınıflı kohort tanımlanmıştır.
- Sprint 2 için veri hazırlama ve split üretimi aşamasına geçilebilir.
