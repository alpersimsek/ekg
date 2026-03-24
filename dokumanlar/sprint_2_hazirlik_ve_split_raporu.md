# Sprint 2 Veri Hazırlama, Manifest ve Split Raporu

## Amaç

Sprint 2 hedefi, `/home/alper/ekg/data/data_3_Sinif` veri kümesi için model eğitimine doğrudan bağlanabilecek bir hazırlık hattı oluşturmaktı.

Tamamlanan işler:

- veri kümesi için tekrar çalıştırılabilir hazırlık betiği yazıldı
- tüm kayıt manifesti üretildi
- temiz 3 sınıflı eğitim manifesti üretildi
- stratified train/validation/test split üretildi
- ön işleme sözleşmesi sabitlendi
- örneklem bazlı tensor doğrulaması yapıldı

## Üretilen Kod

Sprint 2 kapsamında aşağıdaki betik eklendi:

- `/home/alper/ekg/scripts/prepare_uc_sinif_dataset.py`

Bu betik şu işleri yapar:

1. `.hea` header’larını toleranslı biçimde parse eder
2. `siniflar_final.csv` üzerinden etiket üretir
3. dışlanacak kayıtları ayırır
4. temiz 3 sınıflı kohortu çıkarır
5. sınıf bazlı stratified split üretir
6. özet JSON ve iki ayrı manifest CSV dosyası yazar
7. rastgele seçilen `.mat` dosyalarında tensor şekli doğrulaması yapar

## Üretilen Artifact’lar

Sprint 2 sonunda aşağıdaki dosyalar üretildi:

- [manifest_tum_kayitlar.csv](/home/alper/ekg/artifacts/uc_sinif/manifest_tum_kayitlar.csv)
- [manifest_uc_sinif_temiz.csv](/home/alper/ekg/artifacts/uc_sinif/manifest_uc_sinif_temiz.csv)
- [hazirlik_ozeti.json](/home/alper/ekg/artifacts/uc_sinif/hazirlik_ozeti.json)

## Teknik Denetim Sonuçları

Hazırlık betiği çalıştırıldığında aşağıdaki sonuçlar elde edildi:

- Toplam kayıt: `62985`
- `.hea` sayısı: `62985`
- `.mat` sayısı: `62985`
- Eksik `.mat`: `0`
- Sıfır byte dosya: `0`
- Header formatı: tüm kayıtlarda `12 / 500 / 5000`
- Malformed header sayısı: `1`
- Malformed örnek: `JS23074`

Sonuç:

- Veri kümesi Sprint 3 model eğitimi için kullanılabilir durumda.
- `JS23074` için toleranslı parser ihtiyacı hazırlanmış betik içinde ele alındı.

## Temiz 3 Sınıflı Kohort

Etiket kaynağı olarak `siniflar_final.csv` kullanıldı.

Eğitime alınan temiz kohort:

- Toplam: `35458`
- `normal`: `14419`
- `ritim`: `19155`
- `iletim`: `1884`

Dışlanan kayıtlar:

- `morfolojik_iskemik`: `27523`
- `technical_exclude`: `4`

### Sınıf Dengesizliği

| Sınıf | Kayıt | Oran | Balanced Weight |
| --- | ---: | ---: | ---: |
| `normal` | 14419 | `%40.67` | `0.819705` |
| `ritim` | 19155 | `%54.02` | `0.617036` |
| `iletim` | 1884 | `%5.31` | `6.273531` |

Yorum:

- `iletim` sınıfı belirgin şekilde azınlık sınıftır.
- Sprint 3’te weighted loss veya benzeri dengeleme mekanizması kullanılmalıdır.
- Ana model seçim metriği yine `Macro F1` olmalıdır.

## Split Sonuçları

Split üretimi `seed=42` ile deterministik olarak yapıldı.

### Eğitim

- `normal`: `11535`
- `ritim`: `15324`
- `iletim`: `1507`

### Doğrulama

- `normal`: `1441`
- `ritim`: `1915`
- `iletim`: `188`

### Test

- `normal`: `1443`
- `ritim`: `1916`
- `iletim`: `189`

Değerlendirme:

- Split dağılımı sınıf oranlarını iyi koruyor.
- Train/validation/test kümeleri Sprint 3 eğitim akışına doğrudan bağlanabilir.

## Ön İşleme Sözleşmesi

Sprint 2 sonunda aşağıdaki veri sözleşmesi sabitlendi:

- Girdi tensor şekli: `[12, 5000]`
- Kanal sırası: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`
- Varsayılan normalizasyon: `lead_bazli_zscore`
- Filtreleme: varsayılan olarak kapalı, yalnızca deneysel olarak fayda gösterirse eklenecek

Bu kararın nedeni:

- Sprint 2’nin amacı veri akışını sabitlemekti
- aşırı ön işleme ile morfoloji bozulması riski erken safhada istenmiyor
- Sprint 3’te model taban çizgisi önce yalın veri hattı üzerinde kurulmalı

## Tensor Doğrulaması

Rastgele seçilen `256` temiz kohort kaydında `.mat` yükleme denetimi yapıldı:

- Beklenen şekil: `(12, 5000)`
- Sorunlu örnek: `0`

Sonuç:

- Veri yükleme hattı örneklem bazında doğrulandı.

## Teknik Sınırlama

Bu veri kümesinde açık bir `patient_id` alanı bulunmadığı için split şu anda kayıt-bazlı stratified olarak üretilmiştir.

Bu ne anlama gelir:

- Split deterministik ve sınıf dengesini koruyan yapıdadır
- ancak gerçek hasta bazlı ayrım garantisi vermez

Karar:

- Sprint 3 mevcut split ile başlatılabilir
- eğer veri kaynağında güvenilir hasta kimliği elde edilirse split mantığı daha sonra hasta-bazlı biçimde güncellenmelidir

## Sprint 2 Durum Sonucu

Sprint 2 tamamlanmıştır.

Tamamlanan Sprint 2 çıktıları:

1. Hazırlık betiği yazıldı.
2. Tüm kayıt manifesti üretildi.
3. Temiz 3 sınıflı manifest üretildi.
4. Deterministik stratified split üretildi.
5. Ön işleme sözleşmesi sabitlendi.
6. Tensor doğrulaması tamamlandı.

Sprint 3’e geçiş için mevcut başlangıç noktası:

- girdi: `manifest_uc_sinif_temiz.csv`
- hedef: `normal / ritim / iletim`
- metrik: `Macro F1`
- veri sözleşmesi: `12 x 5000`, lead-bazlı z-score
