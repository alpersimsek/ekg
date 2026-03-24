# Run004 ve Run007 Confusion Matrix + Grad-CAM Karsilastirma Raporu

Tarih: 24 Mart 2026

Bu rapor, refined sweep16 sonunda one cikan iki adayin confusion matrix ve Grad-CAM ozetlerini yan yana incelemek icin hazirlanmistir.

Incelenen kosular:

- [run004](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced)
- [run007](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run007_baseline_ep50_bs64_lr00005_wd00001_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone)

## 1. Genel Sonuc

Toplam kalite acisindan:

- `run004` biraz daha iyi

`iletim` sinifi acisindan:

- `run007` belirgin sekilde daha iyi

Bu nedenle:

- genel amacli aday olarak `run004`
- `iletim` odakli aday olarak `run007`

## 2. Skor Karsilastirmasi

`run004`:

- test `Macro F1 = 0.934861`
- test `iletim F1 = 0.893151`

`run007`:

- test `Macro F1 = 0.934375`
- test `iletim F1 = 0.911602`

Yorum:

- toplam `Macro F1` farki cok kucuk
- `iletim F1` farki daha belirgin

## 3. Confusion Matrix Karsilastirmasi

Gorseller:

- [run004 confusion matrix](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced/visuals/confusion_matrix_test.png)
- [run007 confusion matrix](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run007_baseline_ep50_bs64_lr00005_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone/visuals/confusion_matrix_test.png)

Sayisal ozet:

`run004` confusion matrix:

- `normal -> normal = 1382`
- `normal -> ritim = 53`
- `normal -> iletim = 8`
- `ritim -> normal = 74`
- `ritim -> ritim = 1837`
- `ritim -> iletim = 5`
- `iletim -> normal = 10`
- `iletim -> ritim = 16`
- `iletim -> iletim = 163`

`run007` confusion matrix:

- `normal -> normal = 1357`
- `normal -> ritim = 81`
- `normal -> iletim = 5`
- `ritim -> normal = 82`
- `ritim -> ritim = 1831`
- `ritim -> iletim = 3`
- `iletim -> normal = 11`
- `iletim -> ritim = 13`
- `iletim -> iletim = 165`

Ana farklar:

- `run004`, `normal` ve `ritim` ayrimini daha iyi yapiyor
- `run007`, `iletim` sinifini biraz daha iyi tutuyor
- `run007`, `normal` ve `ritim` arasinda daha fazla kayma uretirken `iletim` dogru sayisini artiriyor

Kritik gozlem:

- `run004` daha dengeli bir genel confusion yapisina sahip
- `run007` ise `iletim` icin daha agresif ve daha faydali

## 4. Grad-CAM Ozet Karsilastirmasi

Grad-CAM summary dosyalari:

- [run004 Grad-CAM summary](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run004_baseline_ep50_bs64_lr0001_wd00005_do02_ls0_cosine_min1em06_auglight_samplernone_weightbalanced/visuals/gradcam_summary_test.json)
- [run007 Grad-CAM summary](/home/alper/ekg/artifacts/uc_sinif_refined_sweep16/run007_baseline_ep50_bs64_lr00005_do02_ls005_cosine_min1em05_augnone_samplerweighted_weightnone/visuals/gradcam_summary_test.json)

`run004` ozet ornekleri:

- dogru `normal`: `E05958`
- dogru `ritim`: `A0064`
- dogru `iletim`: `A0131`
- hata `ritim -> normal`: `JS45132`
- hata `ritim -> iletim`: `A2366`
- hata `normal -> iletim`: `JS36182`

`run007` ozet ornekleri:

- dogru `normal`: `HR06470`
- dogru `ritim`: `JS45436`
- dogru `iletim`: `A5830`
- hata `ritim -> normal`: `JS35846`
- hata `ritim -> iletim`: `A2366`
- hata `normal -> iletim`: `E10163`

Grad-CAM tarafinda dikkat ceken sey:

- iki model de benzer zorlayici hata tiplerine sahip
- `A2366` her iki modelde de ortak zor ornek olarak kalmis
- bu, veri veya etiket tarafinda ayrimi intrinsik olarak zor bir alt bolge olabilecegini gosteriyor

## 5. Teknik Yorum

`run004`:

- daha iyi genel denge
- daha iyi toplam `Macro F1`
- klinik olmayan ortamda genel-purpose aday gibi duruyor

`run007`:

- `iletim` sinifinda daha iyi
- azinlik sinifi biraz daha iyi koruyor
- toplam skor kaybi cok kucuk kaldigi icin daha stratejik bir aday olabilir

## 6. Karar Onerisi

Eger tek KPI:

- toplam `Macro F1`

ise:

- `run004`

Eger oncelik:

- `iletim` sinifinda daha iyi yakalama
- azinlik sinif davranisinda daha iyi denge

ise:

- `run007`

Bu raporun teknik tavsiyesi:

- varsayilan final aday olarak `run007`

Gerekce:

- toplam skor kaybi ihmal edilebilir kadar kucuk
- `iletim F1` kazanci anlamli
- klinik siniflandirma senaryosunda azinlik sinifin iyilesmesi genellikle daha degerlidir
