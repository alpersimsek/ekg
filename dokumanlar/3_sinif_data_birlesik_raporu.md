# 3 Sinif Data Birlesik Raporu

## Genel Durum

- Veri klasoru: `/home/alper/ekg/data/3_sinif_data_birlesik`
- `.hea` sayisi: `62985`
- `.mat` sayisi: `62985`
- `12 500 5000` formatini saglayan `.hea` sayisi: `62985`
- Benzersiz SNOMED sayisi: `64`

## Sinif Dagilimi

| Sinif | Kayit | Oran | Balanced Weight | Normalized Inv Weight |
| --- | ---: | ---: | ---: | ---: |
| `normal` | 22107 | %35.10 | 0.949699 | 0.112732 |
| `ritim` | 37844 | %60.08 | 0.554778 | 0.065854 |
| `iletim` | 3034 | %4.82 | 6.919908 | 0.821414 |

## Agirlik Yorumu

- `balanced weight = N / (sinif_sayisi * sinif_adedi)` formulu ile hesaplandi.
- `normalized inverse weight` degerleri toplami 1 olacak sekilde normalize edildi.
- En az gorulen sinif `iletim`; egitimde en yuksek agirligi bunun almasi gerekir.

## Demografi

- Cinsiyet dagilimi: `{'Female': 29256, 'Male': 33712, 'Unknown': 17}`
- Yas grup dagilimi: `{'0-17': 2090, '18-39': 7896, '40-64': 27783, '65-74': 13637, '75+': 11357, 'bilinmiyor': 222}`
- Yas min/max/ortalama: `0` / `92` / `57.57`

## SNOMED Listesi - normal

| SNOMED | EN | TR | Kayit Sayisi |
| --- | --- | --- | ---: |
| `426783006` | Sinus rhythm | Sinüs ritmi | 22107 |

## SNOMED Listesi - ritim

| SNOMED | EN | TR | Kayit Sayisi |
| --- | --- | --- | ---: |
| `426177001` | Sinus bradycardia | Sinüs bradikardisi | 16366 |
| `427084000` | Sinus tachycardia | Sinüs taşikardisi | 8066 |
| `164890007` | Atrial flutter | Atriyal flutter | 5722 |
| `164889003` | Atrial fibrillation | Atriyal fibrilasyon | 3360 |
| `427393009` | Sinus arrhythmia | Sinüs aritmisi | 2779 |
| `284470004` | Premature atrial contraction | Prematür atriyal kontraksiyon | 1967 |
| `427172004` | Premature ventricular contractions | Prematür ventriküler kontraksiyonlar | 785 |
| `426761007` | Supraventricular tachycardia | Supraventriküler taşikardi | 674 |
| `164884008` | Ventricular ectopics | Ventriküler ektopi | 531 |
| `17338001` | Ventricular premature beats | Ventriküler prematür atımlar | 482 |
| `713422000` | Atrial tachycardia | Atriyal taşikardi | 196 |
| `106068003` | Atrial rhythm | Atriyal ritim | 185 |
| `63593006` | Supraventricular premature beats | Supraventriküler prematür atımlar | 85 |
| `426627000` | Bradycardia | Bradikardi | 83 |
| `425856008` | Paroxysmal ventricular tachycardia | Paroksismal ventriküler taşikardi | 82 |
| `29320008` | Atrioventricular junctional rhythm | Atriyoventriküler junctional ritim | 71 |
| `81898007` | Ventricular escape rhythm | Ventriküler kaçış ritmi | 70 |
| `164896001` | Ventricular fibrillation | Ventriküler fibrilasyon | 65 |
| `426995002` | Junctional escape beat | Kavşak kaçış vurusu | 42 |
| `75532003` | Ventricular escape beat | Ventriküler kaçış vurusu | 39 |
| `251170000` | Blocked premature atrial contraction | Bloke prematür atriyal kontraksiyon | 32 |
| `11157007` | Ventricular bigeminy | Ventriküler bigemini | 31 |
| `50799005` | Atrioventricular dissociation | Atriyoventriküler dissosiyasyon | 25 |
| `426648003` | Junctional tachycardia | Kavşak taşikardisi | 23 |
| `67198005` | Paroxysmal supraventricular tachycardia | Paroksismal supraventriküler taşikardi | 21 |
| `5609005` | Sinus arrest | Sinüs aresti | 20 |
| `426664006` | Accelerated junctional rhythm | Hızlanmış junctional ritim | 19 |
| `13640000` | Fusion beats | Füzyon atımları | 19 |
| `233897008` | Re-entrant atrioventricular tachycardia | Re-entran atriyoventriküler taşikardi | 18 |
| `251166008` | Atrioventricular nodal re-entry tachycardia | AV nodal re-entran taşikardi | 16 |
| `251187003` | Atrial escape complex | Atriyal kaçış kompleksi | 15 |
| `251180001` | Ventricular trigeminy | Ventriküler trigemini | 14 |
| `233892002` | Ectopic atrial tachycardia | Ektopik atriyal taşikardi | 14 |
| `195080001` | Atrial fibrillation and flutter | Atriyal fibrilasyon ve flutter | 12 |
| `61277005` | Accelerated idioventricular rhythm | Hızlanmış idioventriküler ritim | 12 |
| `251164006` | Junctional premature complex | Kavşak erken kompleksi | 9 |
| `17366009` | Atrial arrhythmia | Atriyal aritmi | 7 |
| `111288001` | Ventricular flutter | Ventriküler flutter | 7 |
| `195101003` | Wandering atrial pacemaker | Gezici atriyal pacemaker | 6 |
| `251173003` | Atrial bigeminy | Atriyal bigemini | 3 |
| `426749004` | Chronic atrial fibrillation | Kronik atriyal fibrilasyon | 1 |

## SNOMED Listesi - iletim

| SNOMED | EN | TR | Kayit Sayisi |
| --- | --- | --- | ---: |
| `59118001` | Right bundle branch block | Sağ dal bloğu | 873 |
| `270492004` | 1st degree atrioventricular block | 1. derece AV blok | 799 |
| `10370003` | Pacing rhythm | Pacing ritmi | 623 |
| `713426002` | Incomplete right bundle branch block | İnkomplet sağ dal bloğu | 259 |
| `164909002` | Left bundle branch block | Sol dal bloğu | 244 |
| `445118002` | Left anterior fascicular block | Sol anterior fasiküler blok | 149 |
| `698252002` | Nonspecific intraventricular conduction disorder | Nonspesifik intraventriküler ileti bozukluğu | 123 |
| `6374002` | Bundle branch block | Dal bloğu | 75 |
| `713427006` | Complete right bundle branch block | Komplet sağ dal bloğu | 73 |
| `251120003` | Incomplete left bundle branch block | İnkomplet sol dal bloğu | 51 |
| `251268003` | Atrial pacing pattern | Atriyal pacing paterni | 42 |
| `195042002` | 2nd degree atrioventricular block | 2. derece AV blok | 27 |
| `251266004` | Ventricular pacing pattern | Ventriküler pacing paterni | 22 |
| `27885002` | Complete heart block | Tam AV blok | 19 |
| `445211001` | Left posterior fascicular block | Sol posterior fasiküler blok | 16 |
| `74390002` | Wolff-Parkinson-White pattern | WPW paterni | 3 |
| `233917008` | Atrioventricular block | AV blok | 2 |
| `164947007` | Prolonged PR interval | Uzamış PR aralığı | 2 |
| `65778007` | Sinoatrial block | Sinoatriyal blok | 2 |
| `195060002` | Ventricular pre-excitation | Ventriküler pre-eksitasyon | 1 |
| `49578007` | Shortened PR interval | Kısalmış PR aralığı | 1 |
| `733534002` | Complete left bundle branch block | Komplet sol dal bloğu | 1 |
