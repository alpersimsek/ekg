# EKG Projesi – Şu Ana Kadar Yapılanların Özet Raporu

**Tarih:** 23 Mart 2026  
**Hazırlayan:** Otomatik rapor (dokümanlardan derleme)

---

## 1. Bu Proje Ne Yapıyor?

Bu projede, hastanelerde kullanılan **EKG** (Elektrokardiyografi) cihazlarından alınan kalp sinyalleri bir **yapay zekâ modeli** tarafından analiz ediliyor. Amaç, her bir EKG kaydını otomatik olarak şu üç gruptan birine sınıflandırmak:

| Sınıf | Ne Anlama Geliyor? |
|---|---|
| **Normal** | Kalp ritmi düzgün çalışıyor, herhangi bir sorun yok. |
| **Ritim** | Kalbin atış düzeninde bir bozukluk var (örneğin çok hızlı, çok yavaş veya düzensiz atım). |
| **İletim** | Kalpteki elektrik sinyalinin yolunun tıkanması veya gecikmesi var (örneğin "dal bloğu" denen durumlar). |

### Neden Önemli?

Doktorlar her gün yüzlerce EKG kaydına bakıyor. Bir yapay zekâ modeli bu kayıtları ön taramadan geçirirse, doktorun işi hızlanır ve gözden kaçan sorunlar yakalanabilir.

---

## 2. Kullanılan Veriler

### EKG Kaydı Nedir?

Vücuda yapıştırılan 12 farklı elektrottan (sensörden) aynı anda 10 saniye boyunca ölçüm alınır. Her bir elektrot farklı bir açıdan kalbin elektrik aktivitesini görür. Bunlara **kanal** denir. Projede her kayıt şu formattadır:

- **12 kanal** (I, II, III, aVR, aVL, aVF, V1–V6)
- Her kanalda **5.000 ölçüm noktası** (saniyede 500 ölçüm × 10 saniye)

### Veri Havuzu Büyüklüğü

| Bilgi | Değer |
|---|---|
| Toplam EKG kaydı | 62.985 |
| Kullanılan farklı tanı kodu sayısı | 64 |
| Yaş aralığı | 0 – 92 |
| Ortalama yaş | 57,6 |
| Kadın / Erkek oranı | 29.256 / 33.712 |

### Eğitimde Kullanılan Temiz Alt Küme

Tüm kayıtların hepsi eğitimde kullanılmadı. Bazı kayıtlar **morfolojik/iskemik** (kalp kası hasarı gibi) sorunlar içerdiği veya teknik olarak sorunlu olduğu için çıkarıldı. Geriye kalan temiz küme:

| Sınıf | Kayıt Sayısı | Yüzde |
|---|---:|---:|
| Normal | 14.419 | %40,7 |
| Ritim | 19.155 | %54,0 |
| İletim | 1.884 | %5,3 |
| **Toplam** | **35.458** | **%100** |

> **Dikkat:** İletim sınıfı çok az sayıda kayıt içeriyor. Bu, yapay zekâ için zorlu bir durum çünkü az gördüğü şeyi öğrenmekte güçlük çeker. Bu soruna **sınıf dengesizliği** denir.

---

## 3. Kullanılan Terimler Sözlüğü

Aşağıda raporda ve sprint dokümanlarında geçen teknik terimlerin lise seviyesinde açıklamaları verilmiştir:

| Terim | Açıklama |
|---|---|
| **EKG (Elektrokardiyografi)** | Kalbin elektrik sinyallerini ölçen tıbbi test. Göğse ve kollara yapıştırılan sensörlerle yapılır. |
| **Yapay Zekâ Modeli** | Bilgisayara örneklerden öğrenme yeteneği kazandıran bir matematik formüller bütünü. Yeterli örnekle eğitilirse yeni verilerde tahmin yapabilir. |
| **1D CNN (Tek Boyutlu Evrişimli Sinir Ağı)** | Zaman serisi verilerde (ses, EKG gibi) örüntüleri yakalamak için tasarlanmış bir yapay zekâ mimarisi. Sinyalin üzerinde küçük bir pencere kaydırarak anlamlı kalıpları öğrenir. |
| **Epoch (Dönem)** | Modelin tüm eğitim verisini baştan sona bir kere görmesi. 12 epoch demek, modelin veriyi 12 kere tekrar tekrar incelediği anlamına gelir. |
| **Batch Size (Parti Büyüklüğü)** | Modelin bir adımda kaç kayda birden baktığı. Örneğin 64 demek, her adımda 64 EKG kaydı birlikte işleniyor demek. |
| **Learning Rate (Öğrenme Oranı)** | Modelin her adımda ne kadar büyük bir düzeltme yapacağını belirleyen ayar. Çok büyükse model kararsızlaşır, çok küçükse çok yavaş öğrenir. |
| **Loss (Kayıp)** | Modelin tahminlerinin gerçek cevaplardan ne kadar uzak olduğunu gösteren bir sayı. Düşük loss = daha iyi tahmin. |
| **Weighted Cross-Entropy (Ağırlıklı Çapraz Entropi)** | Az sayıdaki sınıfa (örn. iletim) daha fazla önem vererek modelin bu sınıfı görmezden gelmesini engelleyen bir kayıp fonksiyonu. |
| **Cosine Scheduler (Kosinüs Planlayıcısı)** | Öğrenme oranını eğitim boyunca yavaşça düşüren bir strateji. Başta hızlı öğrenir, sonra ince ayar yapar. |
| **Augmentasyon (Veri Artırma)** | Mevcut verilere küçük değişiklikler (gürültü ekleme, kaydırma vb.) uygulayarak yapay olarak daha fazla eğitim verisi oluşturma tekniği. |
| **Label Smoothing (Etiket Yumuşatma)** | Modele "%100 eminim bu normal" demek yerine "%95 eminim" gibi yumuşatılmış cevaplar vererek aşırı özgüvenin önüne geçen bir teknik. |
| **Dropout** | Eğitim sırasında rastgele bazı bağlantıları kapatarak modelin ezberleme yapmasını engelleyen bir yöntem. |
| **Accuracy (Doğruluk)** | Tüm tahminlerin yüzde kaçının doğru olduğu. Ancak dengesiz veride yanıltıcı olabilir (çoğunluk sınıfını hep tahmin edersen de yüksek çıkar). |
| **Macro F1** | Her sınıf için ayrı ayrı hesaplanan F1 skorlarının ortalaması. Küçük sınıfları da eşit ağırlıkla değerlendirdiği için dengesiz veride en güvenilir metriktir. |
| **Precision (Kesinlik)** | "Model X dedi" dediğinde, gerçekten X olanların oranı. Yüksek precision = az yanlış alarm. |
| **Recall (Duyarlılık)** | Gerçekte X olanların ne kadarını modelin yakalayabildiği. Yüksek recall = az gözden kaçırma. |
| **F1 Skoru** | Precision ve recall'un dengeli birleşimi. İkisi de yüksekse F1 yüksek olur. |
| **Confusion Matrix (Karışıklık Matrisi)** | Modelin her sınıf için ne tahmin ettiğini gösteren tablo. Hangi sınıfı neyle karıştırdığı buradan anlaşılır. |
| **Validation Set (Doğrulama Kümesi)** | Eğitim sırasında modelin hiç görmediği ama performansını ölçmek için kullanılan veri parçası. |
| **Test Set (Test Kümesi)** | Tüm geliştirme bittikten sonra modelin son kez sınandığı, hiç dokunulmamış veri parçası. |
| **Checkpoint** | Eğitim sırasında modelin o anki halinin kaydedilmiş dosyası. En iyi performanstaki checkpoint seçilir. |
| **Stratified Split (Katmanlı Bölme)** | Veriyi eğitim/doğrulama/test olarak bölerken her parçada sınıf oranlarının korunmasını sağlayan yöntem. |
| **SNOMED Kodu** | Dünya genelinde kullanılan standart tıbbi tanı kodlama sistemi. Her hastalığa / bulguya bir numara verilir. |
| **WFDB** | EKG gibi fizyolojik sinyalleri standart formatta (.hea ve .mat dosyaları) saklayan bir dosya biçimi ve kütüphane. |
| **Z-Score Normalizasyon** | Her kanalın ortalamasını 0'a, standart sapmasını 1'e getirerek farklı ölçeklerdeki kanalları eşitleme işlemi. |
| **Manifest** | Her kayıdın dosya yolunu, sınıfını ve hangi bölümde (eğitim/doğrulama/test) olduğunu listeleyen rehber dosya. |
| **GPU (Grafik İşlem Birimi)** | Yapay zekâ hesaplamalarını çok hızlı yapan özel donanım. Bu projede NVIDIA RTX 3090 kullanıldı. |
| **Balanced Weight (Dengeli Ağırlık)** | Az görülen sınıflara daha yüksek ağırlık vererek modelin onları da öğrenmesini teşvik eden bir ayarlama. |
| **Weighted Sampler (Ağırlıklı Örnekleyici)** | Eğitim sırasında az sayıdaki sınıftan daha sık örnek çekerek dengesizliği gidermeye çalışan bir veri besleme yöntemi. |

---

## 4. Adım Adım Yapılanlar (Sprint'ler)

Proje 6 aşamaya (sprint) bölündü. Şu ana kadar ilk 5 sprint tamamlandı.

### Sprint 1 – Veri Denetimi ✅

**Amaç:** Elimizdeki EKG verisinin sağlam ve kullanılabilir olduğundan emin olmak.

**Yapılanlar:**
- 62.985 kayıdın tamamında `.hea` (başlık) ve `.mat` (sinyal) dosya çiftlerinin eksiksiz olduğu doğrulandı
- Tüm kayıtların 12 kanal ve 5.000 ölçüm noktası formatında olduğu kontrol edildi
- 1 adet bozuk başlık dosyası (`JS23074.hea`) tespit edildi — sorun küçüktü ve yazılımda tolere edildi
- Hangi tanı kodlarının hangi sınıfa (normal / ritim / iletim) ait olduğu bir eşleme tablosuyla sabitlendi
- Morfolojik/iskemik ve teknik sorunlu kayıtlar (27.527 adet) eğitim dışı bırakıldı

**Sonuç:** Veri güvenilir ve kullanıma hazır.

---

### Sprint 2 – Veri Hazırlama ve Bölme ✅

**Amaç:** Veriyi modelin anlayacağı formata dönüştürmek ve eğitim/doğrulama/test parçalarına ayırmak.

**Yapılanlar:**
- Otomatik veri hazırlama betiği (scripti) yazıldı
- 35.458 temiz kayıt manifeste kaydedildi
- Veri üç parçaya bölündü (sınıf oranları korunarak):

| Bölüm | Normal | Ritim | İletim | Toplam |
|---|---:|---:|---:|---:|
| Eğitim | 11.535 | 15.324 | 1.507 | 28.366 |
| Doğrulama | 1.441 | 1.915 | 188 | 3.544 |
| Test | 1.443 | 1.916 | 189 | 3.548 |

- Ön işleme kuralı belirlendi: her kanal kendi içinde z-score normalizasyonuna tabi tutulacak; gereksiz filtreleme uygulanmayacak

**Sonuç:** Model eğitimine hazır, tekrarlanabilir bir veri hattı kuruldu.

---

### Sprint 3 – İlk Model Eğitimi (Baz Model) ✅

**Amaç:** İlk test edilebilir yapay zekâ modelini eğitmek ve referans performans almak.

**Yapılanlar:**
- 1D CNN (tek boyutlu evrişimli sinir ağı) mimarisi kuruldu
- Sınıf dengesizliği için ağırlıklı kayıp fonksiyonu kullanıldı
- 3 epoch eğitim yapıldı (toplam ~18 saniye, GPU üzerinde)
- İlk sonuçlar alındı

**İlk Sonuçlar:**

| Metrik | Doğrulama | Test |
|---|---:|---:|
| **Macro F1** | 0,826 | 0,845 |
| Normal F1 | 0,881 | 0,876 |
| Ritim F1 | 0,880 | 0,887 |
| İletim F1 | 0,718 | 0,772 |

**Yorum:** Model çalışıyor ve rastgele tahminden çok daha iyi. Ancak iletim sınıfında performans diğerlerine göre düşük — çünkü bu sınıftan çok az örnek var.

---

### Sprint 4 – Hata Analizi ve Deneyler ✅

**Amaç:** Modelin nerelerde hata yaptığını anlamak ve düzeltme denemek.

**Ana Bulgular:**
- En büyük hata: **ritim** olan kayıtların **normal** olarak tahmin edilmesi (288 adet doğrulama, 301 adet test)
- İletim sınıfında asıl sorun "yakalayamamak" değil, "yanlış alarm" vermek (normal veya ritim kayıtları iletim diye etiketlemek)
- Özellikle sinüs bradikardisi, sinüs aritmisi ve erken atımlar gibi hafif ritim bozuklukları modeli şaşırtıyor

**Denenen İyileştirmeler:**

| Deney | Doğrulama Macro F1 | Test Macro F1 |
|---|---:|---:|
| Baz model (referans) | 0,826 | 0,845 |
| Ağırlıklı örnekleme + dengeli kayıp | 0,813 | 0,832 |
| Ağırlıklı örnekleme + ağırlıksız kayıp | 0,822 | 0,851 |

**Sonuç:** Ağırlıklı örnekleme tek başına sorunu çözmedi. Baz model hâlâ en güçlü referans olarak kaldı.

---

### Sprint 5 – Optimizasyon (İnce Ayar) ✅

**Amaç:** Modeli daha uzun eğitim ve daha akıllı ayarlarla iyileştirmek.

**Yapılanlar (4 ayrı deney):**

| Deney | Epoch | Öğrenme Oranı | Ek Özellik | Doğr. Macro F1 | Test Macro F1 |
|---|---:|---|---|---:|---:|
| Baz model | 3 | 1e-3 | — | 0,826 | 0,845 |
| Deney 1 | 6 | 1e-3 | — | 0,853 | 0,877 |
| Deney 2 | 6 | 5e-4 | — | 0,862 | 0,891 |
| **Deney 3** | **12** | **5e-4** | **Kosinüs planlayıcı** | **0,908** | **0,921** |
| Deney 4 | 12 | 5e-4 | Kosinüs + augmentasyon + etiket yumuşatma | 0,907 | 0,910 |

**En İyi Modelin Sınıf Bazlı Test Sonuçları:**

| Sınıf | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0,926 | 0,940 | 0,933 |
| Ritim | 0,958 | 0,942 | 0,950 |
| İletim | 0,859 | 0,905 | 0,881 |

**Sonuç:** 
- En büyük kazanım **daha uzun eğitim** ve **kosinüs öğrenme oranı planlayıcısı** ile geldi
- Augmentasyon ve etiket yumuşatma bu veri setinde ek fayda sağlamadı
- İletim sınıfı F1 skoru 0,772'den 0,881'e yükseldi — çok anlamlı bir artış

---

## 5. Genel İlerleme Özeti

Aşağıdaki grafik baz modelden en iyi modele kadar olan ilerlemeyi gösterir:

| Metrik | Sprint 3 (Baz) | Sprint 5 (En İyi) | İyileşme |
|---|---:|---:|---|
| Doğrulama Macro F1 | 0,826 | 0,908 | **+0,082** |
| Test Macro F1 | 0,845 | 0,921 | **+0,076** |
| Test Normal F1 | 0,876 | 0,933 | +0,057 |
| Test Ritim F1 | 0,887 | 0,950 | +0,063 |
| Test İletim F1 | 0,772 | 0,881 | **+0,109** |
| Test Doğruluk | %87,5 | %93,9 | +%6,4 |

---

## 6. En İyi Modelin Reçetesi

Şu an projedeki en başarılı model şu ayarlarla eğitildi:

- **Mimari:** 1D CNN (tek boyutlu evrişimli sinir ağı)
- **Girdi:** 12 kanal × 5.000 ölçüm noktası
- **Ön işleme:** Kanal bazlı z-score normalizasyon
- **Epoch sayısı:** 12
- **Öğrenme oranı:** 0,0005 (5e-4)
- **Parti büyüklüğü:** 64
- **Kayıp fonksiyonu:** Ağırlıklı çapraz entropi (dengeli ağırlıklar)
- **Öğrenme oranı planlayıcısı:** Kosinüs
- **Augmentasyon:** Yok
- **Dropout:** 0,2
- **Donanım:** NVIDIA RTX 3090

**Model dosyası:** `artifacts/uc_sinif_opt_exp3_e12_cosine/best_model.pt`

---

## 7. Bilinen Sınırlılıklar

1. **Sınıf dengesizliği:** İletim sınıfı verideki en küçük grup. İyileşme sağlandı ama hâlâ en zayıf sınıf.
2. **Kayıt bazlı bölme:** Aynı hastanın birden fazla kaydı varsa, bunlar farklı bölümlere düşebilir. Hasta bazlı kesin ayırım garanti edilemedi çünkü veri setinde hasta kimliği bulunmuyor.
3. **Sınırlı model mimarisi:** Şu an basit bir CNN kullanılıyor. Daha gelişmiş mimariler (ResNet, Transformer vb.) denenmedi.
4. **Karıştırılan alt tipler:** Özellikle hafif ritim bozuklukları (sinüs bradikardisi, sinüs aritmisi) bazen normal olarak sınıflandırılıyor.

---

## 8. Sonraki Adım (Sprint 6)

Sprint 6'da yapılması planlananlar:

1. En iyi modelin bağımsız test kümesinde nihai değerlendirmesi
2. Tek bir EKG kaydı verince sonuç döndüren inference (tahmin) betiğinin yazılması
3. Nihai raporun ve kullanım kılavuzunun hazırlanması
4. Modelin girdi-çıktı kurallarının son haline getirilmesi

---

## 9. Sonuç

Bu proje, 12 kanallı EKG sinyallerinden kalp ritim ve iletim bozukluklarını otomatik tespit eden bir yapay zekâ sistemi geliştirmeyi hedeflemiştir. 5 sprint boyunca:

- **62.985** kayıtlık bir veri havuzu denetlenip temizlendi
- **35.458** temiz kayıtla eğitim yapıldı
- İlk baz modelden başlayarak sistematik deneylerle model iyileştirildi
- **Test Macro F1 skoru 0,845'ten 0,921'e** çıkarıldı
- En zor sınıf olan iletimde bile **%88,1** F1 skoruna ulaşıldı

Model, her üç sınıfta da güvenilir sonuçlar üretebilecek düzeye gelmiştir ve Sprint 6'da finalize edilmeye hazırdır.
