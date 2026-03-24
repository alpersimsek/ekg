# Hypergrid Kullanim Kilavuzu

Tarih: 24 Mart 2026

Bu dokuman, hiperparametre tarama sisteminin kullanimini aciklar.

## 1. Genel Yapi

Projede artik iki ayri egitim yolu vardir:

- mevcut calisan akış:
  - [train_uc_sinif_baseline.py](/home/alper/ekg/scripts/train_uc_sinif_baseline.py)
  - bu script eski ve stabil egitim akisini korur
- yeni hiperparametre laboratuvari:
  - [train_uc_sinif_hparam.py](/home/alper/ekg/scripts/train_uc_sinif_hparam.py)
  - [run_uc_sinif_hypergrid.py](/home/alper/ekg/scripts/run_uc_sinif_hypergrid.py)

Karar:

- mevcut modelleme akisini bozmak istemiyorsak eski script kullanilir
- genis scheduler ve hiperparametre taramasi yapmak istiyorsak yeni sistem kullanilir

## 2. Yeni Sistemin Amaci

Yeni sistemin amaci, makul hiperparametrelerin alabilecegi degerlerden genis bir deney uzayi kurmaktir.

Bu sistem:

- teorik kombinasyon sayisini hesaplar
- secilen sayida kosu uretir
- farkli scheduler turlerini destekler
- her kosu sonunda confusion matrix ve Grad-CAM uretir
- kaldigi yerden devam edebilir
- shard mantigi ile bolunebilir

## 3. Dosyalar

Ana dosyalar:

- trainer: [train_uc_sinif_hparam.py](/home/alper/ekg/scripts/train_uc_sinif_hparam.py)
- orchestration: [run_uc_sinif_hypergrid.py](/home/alper/ekg/scripts/run_uc_sinif_hypergrid.py)
- gorsellestirme: [render_confusion_and_gradcam.py](/home/alper/ekg/scripts/render_confusion_and_gradcam.py)

## 4. Desteklenen Schedulerlar

Yeni trainer tarafinda desteklenen scheduler secenekleri:

- `none`
- `cosine`
- `plateau`
- `onecycle`
- `cosine_warm_restarts`
- `step`
- `multistep`

Ek scheduler parametreleri:

- `min_lr`
- `scheduler_gamma`
- `scheduler_step_size`
- `scheduler_milestones`
- `plateau_patience`
- `plateau_factor`
- `onecycle_pct_start`
- `warm_restart_t0`
- `warm_restart_tmult`

## 5. Tanımlı Hiperparametre Uzayi

Hypergrid scriptinde su alanlar tanimlidir:

- `model`
- `epochs`
- `batch_size`
- `lr`
- `weight_decay`
- `dropout`
- `label_smoothing`
- `augment`
- `sampler`
- `class_weight_mode`
- `scheduler`
- scheduler’a ozel alt parametreler

Mevcut tanimli uzayda teorik toplam kombinasyon sayisi:

- `17406`

Not:

- bu sayi tum olasi kosularin ayni anda kosulacagi anlamina gelmez
- `--max-runs` ile kac kosunun secilecegi kontrol edilir

## 6. Temel Komutlar

### 6.1 Dry-run

Sadece planlanan kosulari gormek icin:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py --dry-run --max-runs 20
```

Bu komut:

- egitimi baslatmaz
- secilen kosulari JSON olarak yazar
- teorik toplam kombinasyon sayisini gosterir

Dry-run ciktisini dosyaya almak icin:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --dry-run \
  --max-runs 20 \
  > /home/alper/ekg/artifacts/hypergrid_dry_run_20.json
```

Bu kullanim ne zaman mantikli:

- yeni bir sweep baslatmadan once hangi kosularin secilecegini gormek istediginde
- secim modunun bekledigin gibi dagilim verip vermedigini kontrol etmek istediginde
- shard kullanacaksan her shard’in nasil bir kosu listesi aldigini once gormek istediginde

### 6.2 64 Kosuluk Sweep

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 64 \
  --selection-mode even \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid64
```

Bu komutun anlami:

- teorik uzaydan `64` kosu secer
- secimi tum uzaya yayilmis sekilde yapar
- butun ciktlari `/home/alper/ekg/artifacts/uc_sinif_hypergrid64` altina yazar
- her kosu sonunda otomatik confusion matrix ve Grad-CAM uretir

### 6.3 Rastgele Secim ile Sweep

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 64 \
  --selection-mode random \
  --seed 42 \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid64_random
```

Bu komutun anlami:

- teorik uzaydan `64` kosu secer
- secim rastgele yapilir
- ayni `seed` ile ayni alt kume tekrar uretilebilir
- ikinci veya ucuncu tur cesitlilik ararken kullanislidir

### 6.4 Kucuk Pilot Kosu

Oncelikle kisa bir pilot tarama baslatmak icin:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 8 \
  --selection-mode even \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid_pilot8
```

Bu tur ne ise yarar:

- yeni sistemin dogru calistigini gorursun
- artifact yapisinin beklendigi gibi olustugunu dogrularsin
- tek seferde cok buyuk maliyete girmeden ilk kalite sinyalini alirsin

## 7. Secim Modlari

Desteklenen secim modlari:

- `first`
  - listedeki ilk `N` kosuyu alir
- `even`
  - tum uzaya yayilmis sekilde kosu secer
- `random`
  - seed kontrollu rastgele secim yapar

Pratik oneriler:

- hizli ve dengeli tarama icin `even`
- ikinci tur cesitlilik icin `random`
- debug amacli kucuk kosular icin `first`

## 8. Shard Destegi

Buyuk taramayi bolmek icin:

- `--shard-count`
- `--shard-index`

Ornek:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 128 \
  --selection-mode even \
  --shard-count 4 \
  --shard-index 0 \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid_shard0
```

Bu yapi:

- toplam adaylari 4 parcaya boler
- sadece `0` indeksli parcayi kosar

Tum shard’lari ayri terminallerde baslatma ornegi:

Terminal 1:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 128 \
  --selection-mode even \
  --shard-count 4 \
  --shard-index 0 \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid_shard0
```

Terminal 2:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 128 \
  --selection-mode even \
  --shard-count 4 \
  --shard-index 1 \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid_shard1
```

Terminal 3:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 128 \
  --selection-mode even \
  --shard-count 4 \
  --shard-index 2 \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid_shard2
```

Terminal 4:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 128 \
  --selection-mode even \
  --shard-count 4 \
  --shard-index 3 \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid_shard3
```

Bu yontem ne zaman mantikli:

- birden fazla GPU veya calisma ortami varsa
- tek terminalde aylar surecek taramayi bolmek istiyorsan
- buyuk sweep’i parca parca raporlamak istiyorsan

## 9. Kaldigi Yerden Devam Etme

Bitmis kosulari atlayarak devam etmek icin:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 64 \
  --skip-completed \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid64
```

Bir kosunun tamamlanmis sayilmasi icin beklenen ana dosyalar:

- `training_report.json`
- `visuals/gradcam_summary_test.json`

Pratik senaryo:

1. sweep basladi
2. makine yeniden basladi veya surec yarida kesildi
3. ayni komutu `--skip-completed` ile tekrar calistirirsin
4. bitmis kosular atlanir, eksik olanlar devam eder

Ornek:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 64 \
  --selection-mode even \
  --skip-completed \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid64
```

## 10. Uretilen Ciktilar

Her kosu klasoru altinda:

- `best_model.pt`
- `training_report.json`
- `run_summary.json`
- `visuals/confusion_matrix_test.png`
- `visuals/gradcam_summary_test.json`
- ornek Grad-CAM PNG dosyalari

Sweep kok klasorunde:

- `hypergrid_plan.json`
- `hypergrid_summary.json`

Bu iki dosyanin gorevi:

- `hypergrid_plan.json`
  - secilen kosularin planini tutar
  - daha egitim bitmeden hangi kosularin planlandigini gormeni saglar
- `hypergrid_summary.json`
  - tamamlanan kosularin ozet tablosudur
  - en iyi validation/test skorlarini listeler

Bir kosunun detayina bakmak icin:

```bash
sed -n '1,220p' /home/alper/ekg/artifacts/uc_sinif_hypergrid64/run0001_.../training_report.json
```

Toplu sonucu incelemek icin:

```bash
sed -n '1,260p' /home/alper/ekg/artifacts/uc_sinif_hypergrid64/hypergrid_summary.json
```

Bir klasorde kac kosunun tamamen bittigini saymak icin:

```bash
find /home/alper/ekg/artifacts/uc_sinif_hypergrid64 -maxdepth 3 -name run_summary.json | wc -l
```

Confusion matrix sayisini kontrol etmek icin:

```bash
find /home/alper/ekg/artifacts/uc_sinif_hypergrid64 -maxdepth 3 -path '*/visuals/confusion_matrix_test.png' | wc -l
```

Grad-CAM summary sayisini kontrol etmek icin:

```bash
find /home/alper/ekg/artifacts/uc_sinif_hypergrid64 -maxdepth 3 -path '*/visuals/gradcam_summary_test.json' | wc -l
```

## 11. Dikkat Edilecek Noktalar

- bu sistem cok buyuk kombinasyon uzayi uretebilir
- tum teorik kombinasyonlari tek seferde kosmak pahali olabilir
- once `dry-run` ile plan kontrol edilmelidir
- sonra `max-runs` ile kontrollu bir tur secilmelidir
- uzun kosularda `skip-completed` kullanmak mantiklidir

## 12. Onerilen Is Akisi

Onerilen sira:

1. `dry-run` ile secilecek kosulari gormek
2. `max-runs 16` veya `32` ile pilot sweep yapmak
3. sonuc iyiyse `64` veya `128` kosuya cikmak
4. en iyi kosulari `hypergrid_summary.json` uzerinden secmek
5. gerekirse sadece iyi bolgelerde ikinci tur sweep acmak

Bu akisin nasil uygulanacagi:

### Adim 1: Dry-run ile plan kontrolu

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --dry-run \
  --max-runs 16 \
  --selection-mode even \
  > /home/alper/ekg/artifacts/hypergrid_plan_16.json
```

Bu adimda bakilacak sey:

- secilen kosu sayisi dogru mu
- scheduler cesitliligi yeterli mi
- run isimleri bekledigin gibi mi

### Adim 2: Kucuk pilot tarama

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 16 \
  --selection-mode even \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid16
```

Bu adimda beklenen:

- script basarili sekilde tum 16 kosuyu tamamladi mi
- her kosuda confusion matrix ve Grad-CAM olustu mu
- hangi scheduler ve LR bolgeleri umut veriyor

### Adim 3: Sonuclari hizli kontrol etme

```bash
sed -n '1,260p' /home/alper/ekg/artifacts/uc_sinif_hypergrid16/hypergrid_summary.json
```

Bu adimda odak:

- en iyi `validation Macro F1`
- en iyi `test Macro F1`
- en iyi `test iletim F1`

### Adim 4: Sweep’i 64 kosuya buyutme

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 64 \
  --selection-mode even \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid64
```

Bu adimin amaci:

- daha genis uzayi taramak
- pilotta iyi gorunen bolgelerin daha fazla varyasyonunu yakalamak

### Adim 5: Gerekirse kaldigi yerden devam etme

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 64 \
  --selection-mode even \
  --skip-completed \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid64
```

Bu adim:

- yarida kalan sweep’i tamamlar
- bitmis kosulari tekrar kosmaz

### Adim 6: Ikinci tur, daha cesitli secim

Pilot veya ilk 64 kosu sonunda farkli alt uzaylar denemek istersen:

```bash
.venv/bin/python scripts/run_uc_sinif_hypergrid.py \
  --max-runs 64 \
  --selection-mode random \
  --seed 123 \
  --output-root /home/alper/ekg/artifacts/uc_sinif_hypergrid64_random123
```

Bu adim:

- ilk turda hic denenmemis bolgelere temas etmeni saglar
- ayni uzayin farkli kesitlerini denemene yardim eder

### Adim 7: En iyi kosulari secme

Toplu ozeti ac:

```bash
sed -n '1,320p' /home/alper/ekg/artifacts/uc_sinif_hypergrid64/hypergrid_summary.json
```

Sonra secilen lider kosunun confusion matrix ve Grad-CAM dosyalarina bak:

```bash
ls /home/alper/ekg/artifacts/uc_sinif_hypergrid64/run0001_*/visuals
```

Bu adimda karar seklinin tipik formu:

- toplam kalite oncelikliyse `test Macro F1`
- azinlik sinif oncelikliyse `test iletim F1`
- deployment icin hem toplam denge hem `iletim` hassasiyeti birlikte degerlendirilir

### Adim 8: Daraltılmış ikinci tur

Ilk taramada iyi gorunen ayarlara gore yeni bir daraltılmış uzay tanimlanabilir.

Pratik ornek:

- `resnet_se` daha iyi gorunduyse sadece o mimariye odaklanmak
- `onecycle` zayif gorunduyse onu gecici olarak cikarmak
- `lr` araligini `1e-3`, `5e-4`, `3e-4` civarina daraltmak
- `dropout` icin sadece `0.2` ve `0.3` ile devam etmek

Bu noktada script icindeki uzayi revize ederek ikinci bir hypergrid turu acilabilir.

## 13. Ornek Uretim Benzeri Senaryo

Asagidaki sira pratikte en dengeli yaklasimdir:

1. `dry-run 16`
2. `pilot 16`
3. `summary` kontrolu
4. `64 even`
5. gerekirse `64 random`
6. en iyi 5 kosuyu sec
7. deployment icin lider modeli ayri raporla

## 13. Son Not

Bu yeni sistem:

- mevcut calisan modeli korur
- yeni deney altyapisini ayrik tutar
- zamanla daha fazla mimari veya loss secenegi eklemeye uygundur

Bu nedenle ana uretim akisi ile deneysel hiperparametre arastirmasi birbirine karismadan ilerletilebilir.
