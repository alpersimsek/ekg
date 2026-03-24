# 12 Kanallı EKG ile 3 Sınıflı Tahminleme Sprint Planı

## Proje Hedefi

Bu projenin amacı, `/home/alper/ekg/data/data_3_Sinif` veri kümesindeki 10 saniyelik ham 12 kanallı EKG sinyallerini kullanarak her kayıt için aşağıdaki 3 sınıftan tam olarak birini tahmin eden bir model geliştirmektir:

- `normal`
- `ritim`
- `iletim`

Model başarısı sınıf dengesizliğinden dolayı `accuracy` yerine öncelikli olarak `Macro F1` metriği ile değerlendirilecektir.

## Problem Tanımı

- Girdi: `12 kanal x 5000 örnek` ham EKG sinyali
- Çıktı: `normal`, `ritim` veya `iletim`
- Görev tipi: tek etiketli 3 sınıflı sınıflandırma
- Ana veri kaynağı: `/home/alper/ekg/data/data_3_Sinif`
- Ana performans metriği: `Macro F1`

## Temel Prensipler

- Veri bölme işlemi sızıntı riski açısından dikkatle doğrulanmalıdır.
- Az temsil edilen sınıf olan `iletim` için recall ve F1 düzenli olarak izlenmelidir.
- Model seçimi `accuracy` değil `validation Macro F1` üzerinden yapılmalıdır.
- Nihai sistem tek bir EKG kaydı için yalnızca tek bir sınıf üretmelidir.

## Sprint 1: Veri Denetimi ve Etiket Sözleşmesi

### Amaç
`data_3_Sinif` veri kümesinin eğitim için güvenilir, tutarlı ve 3 sınıflı hedefe uygun hale getirilmesi.

### Görevler

1. `/home/alper/ekg/data/data_3_Sinif` altındaki tüm `.hea` ve `.mat` dosya çiftlerini doğrula.
2. Her kaydın beklenen 12 kanal ve 5000 örnek yapısını sağlayıp sağlamadığını kontrol et.
3. `normal`, `ritim`, `iletim` etiket kurallarını sabitle.
4. Tek etiketli eğitim hedefi ile çelişen kayıtları tespit et.
5. Eğitimde kullanılacak nihai manifest yapısını tanımla.

### Çıktılar

- Temizlenmiş kayıt listesi
- Etiketleme kuralları dokümanı
- Nihai manifest şeması: `record_id`, `dosya_yolu`, `hedef_sinif`, `bolum`

### Kabul Kriterleri

- Her eğitim kaydının geçerli `.hea/.mat` çifti olmalı.
- Her kaydın hedef etiketi 3 sınıftan yalnızca biri olmalı.
- Bozuk veya eksik kayıtlar raporlanmış olmalı.

## Sprint 2: Veri Yükleme, Ön İşleme ve Bölme Altyapısı

### Amaç
Model eğitimine uygun, tekrarlanabilir veri boru hattını kurmak.

### Görevler

1. WFDB tabanlı veri yükleyiciyi oluştur.
2. 12 kanallı sinyalleri tensör formatında oku.
3. Gerekli ön işlemleri uygula:
   - temel normalizasyon
   - isteğe bağlı filtreleme
   - deneysel olarak fayda sağlıyorsa baz çizgisi düzeltme
4. Eğitim, doğrulama ve test bölümlerini sabitle.
5. Tekrarlanabilirlik için seed ve veri hazırlama adımlarını standartlaştır.

### Çıktılar

- Eğitim veri yükleyicisi
- Ön işleme modülü
- Dondurulmuş train/validation/test manifestleri

### Kabul Kriterleri

- Tüm kayıtlar aynı tensör biçiminde yüklenebilmeli.
- Aynı konfigürasyon tekrar çalıştırıldığında aynı split elde edilmeli.
- Veri yükleme hattı hatasız şekilde eğitim döngüsüne bağlanabilmeli.

## Sprint 3: Bazal 3 Sınıflı Modelin Kurulması

### Amaç
Ham 12 kanallı EKG sinyali üzerinden çalışan ilk güvenilir baz modelin eğitilmesi.

### Görevler

1. 1D CNN veya 1D ResNet tabanlı baz mimariyi kur.
2. Çıkış katmanını 3 sınıflı tek etiketli sınıflandırmaya göre tanımla.
3. Sınıf dengesizliği için weighted cross-entropy veya focal loss dene.
4. İlk eğitim koşusunu tamamla.
5. Validation set üzerinde `Macro F1`, sınıf bazlı `precision`, `recall`, `F1` ve confusion matrix üret.

### Çıktılar

- İlk eğitimli checkpoint
- Bazal validation performans raporu
- Confusion matrix

### Kabul Kriterleri

- Eğitim süreci kararlı biçimde tamamlanmalı.
- Çoğunluk sınıfı tahmini yapan naif yaklaşımdan belirgin şekilde daha iyi `Macro F1` elde edilmeli.
- `iletim` sınıfı için performans ayrı raporlanmalı.

## Sprint 4: Hata Analizi ve İyileştirme Döngüsü

### Amaç
Modelin özellikle azınlık sınıflardaki hatalarını görünür hale getirmek ve en etkili iyileştirmeleri seçmek.

### Görevler

1. Yanlış sınıflanan `iletim` örneklerini incele.
2. `ritim` lehine oluşan yanlış pozitif örüntülerini analiz et.
3. Aşağıdaki iyileştirmeleri kontrollü deneylerle karşılaştır:
   - class weight ayarı
   - balanced sampler
   - veri artırma
   - farklı ön işleme varyantları
4. En yüksek etkili 2 veya 3 iyileştirmeyi seç.

### Çıktılar

- Hata analizi raporu
- Deney karşılaştırma tablosu
- Bir sonraki sprintte kullanılacak iyileştirme listesi

### Kabul Kriterleri

- Hangi sınıfta neden hata yapıldığı açıkça raporlanmalı.
- Seçilen iyileştirmeler ölçülebilir validation kazanımı göstermeli.

## Sprint 5: Macro F1 Optimizasyonu

### Amaç
Nihai aday modeli `validation Macro F1` açısından en iyi noktaya taşımak.

### Görevler

1. Öğrenme oranı, batch size ve loss ağırlıklarını ayarla.
2. Model derinliği ve düzenlileştirme seçeneklerini karşılaştır.
3. En iyi ön işleme ve örnekleme stratejisini sabitle.
4. Model seçimini yalnızca validation `Macro F1` ile yap.

### Çıktılar

- En iyi validation skoruna sahip aday model
- Sabit eğitim reçetesi
- Deney özeti

### Kabul Kriterleri

- Validation `Macro F1`, baz modele göre anlamlı artış göstermeli.
- `normal`, `ritim`, `iletim` için sınıf bazlı F1 raporlanmalı.
- Model seçimi açık ve tekrarlanabilir olmalı.

## Sprint 6: Final Test ve Üretime Hazırlık

### Amaç
Kilidi açılmış en iyi modelin bağımsız test kümesinde değerlendirilmesi ve tahmin hattının tamamlanması.

### Görevler

1. En iyi validation modelini tek seferlik test değerlendirmesine al.
2. Test kümesi için `Macro F1`, sınıf bazlı `precision`, `recall`, `F1` ve confusion matrix üret.
3. Tek kayıt üzerinden çalışan inference akışını hazırla.
4. Girdi EKG sinyalinden tek sınıf çıktısı üreten final betiğini oluştur.
5. Bilinen zayıf noktaları ve sınıf bazlı riskleri dokümante et.

### Çıktılar

- Nihai test raporu
- Final checkpoint
- Inference betiği
- Kullanım notları

### Kabul Kriterleri

- Test set sonuçları dokümante edilmiş olmalı.
- Model tek bir EKG kaydı için 3 sınıftan yalnızca birini döndürmeli.
- Nihai başarı metriği olarak `Macro F1` raporun başında yer almalı.

## İzlenecek Ana Metrikler

- `Macro F1`
- Sınıf bazlı `F1`
- Sınıf bazlı `Recall`
- Sınıf bazlı `Precision`
- Confusion matrix

## Riskler

- `iletim` sınıfı düşük örnek sayısı nedeniyle ezilebilir.
- Veri sızıntısı olursa validation ve test skorları yanıltıcı yükselir.
- Accuracy odaklı seçim yapılırsa model `ritim` sınıfına aşırı kayabilir.
- Aşırı ön işleme bazı tanısal morfolojileri bozabilir.

## Tamamlanma Tanımı

Proje aşağıdaki koşullar sağlandığında tamamlanmış kabul edilir:

- `/home/alper/ekg/data/data_3_Sinif` için tekrarlanabilir eğitim manifesti üretilmiş olmalı.
- 12 kanallı EKG sinyalini girdi alan 3 sınıflı model eğitilmiş olmalı.
- Nihai model `normal`, `ritim`, `iletim` sınıflarından tam olarak birini tahmin etmeli.
- Validation ve test sonuçlarında ana karar metriği `Macro F1` olmalı.
- Final raporda sınıf bazlı performans ve hata dağılımı açıkça sunulmalı.
