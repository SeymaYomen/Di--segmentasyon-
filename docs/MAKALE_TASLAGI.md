# Makale Taslağı

## Geçici başlık

**Panoramik Diş Röntgeni Segmentasyonunda Dış-Merkez Genellenebilirlik,
CLAHE Ön İşleme ve Conformal Belirsizlik Analizi**

## Özet

Amaç, panoramik diş röntgenlerinde U-Net++ tabanlı ikili diş
segmentasyonunun iç ve bağımsız dış veri dağılımlarındaki performansını
incelemek; CLAHE ön işlemesinin genellenebilirliğe etkisini ve piksel düzeyi
conformal belirsizlik davranışını değerlendirmektir. Model, hasta bazında
ayrılmış CDPR verisi üzerinde eğitilmiş ve iki bağımsız kaynakta test
edilmiştir. Ana sonuç, CLAHE'nin iç testte küçük bir performans düşüşüne rağmen
Mendeley dış testinde Dice ve IoU'yu artırmasıdır. İkinci dış test küçük ve
yoğun sızıntı temizliği sonrası kaldığı için keşifsel olarak raporlanmıştır.

## 1. Giriş

Panoramik dental radyografiler, tüm dentisyonu tek görüntüde göstermeleri
nedeniyle tanı, tedavi planlaması ve izlemde yaygın olarak kullanılmaktadır.
Dişlerin otomatik olarak bölütlenmesi; sonraki numaralandırma, hastalık
tespiti ve nicel görüntü analizi adımları için temel bir ön işlemdir. Ancak tek
bir veri kaynağında elde edilen yüksek başarı, cihaz, merkez, hasta dağılımı ve
görüntüleme protokolü değiştiğinde aynı düzeyde performansı garanti etmez.

Bu çalışmada kullanılan CDPR kaynağı, çocuk ve yetişkin panoramik
radyografilerini ve diş maskelerini açık olarak sunan bir veri tanımlama
çalışmasına dayanmaktadır [1]. Segmentasyon modeli olarak, iç içe ve yoğun
atlama bağlantılarıyla kodlayıcı-kod çözücü arasındaki anlamsal farkı azaltmayı
amaçlayan U-Net++ kullanılmıştır [2]. Görüntüler arası yerel kontrast
farklılıklarının etkisini incelemek için baseline koluna ek olarak
Contrast-Limited Adaptive Histogram Equalization (CLAHE) kolu kurulmuştur [3].

Çalışmanın katkısı yalnızca yüksek bir Dice skoru raporlamak değildir. Üç
araştırma sorusu birlikte ele alınmıştır:

1. İç testten bağımsız dış teste geçildiğinde performans ne kadar değişmektedir?
2. CLAHE, bu genelleme farkını tutarlı biçimde azaltmakta mıdır?
3. Piksel düzeyi conformal kalibrasyon, hedeflenen kapsama düzeyini sağlarken
   ne ölçüde belirsiz/boş tahmin üretmektedir?

## 2. Materyal ve Yöntem

### 2.1 Veri kümeleri

CDPR kaynağında dosya bütünlüğü, boş maske ve aynı görüntünün tekrarı denetlenmiş,
2.398 temiz görüntü/maske çifti tutulmuştur. Eğitim, kalibrasyon, doğrulama ve
iç test ayrımı hasta düzeyinde gerçekleştirilmiştir. İç test 360 görüntüden
(330 yetişkin, 30 çocuk) oluşmaktadır.

Birinci bağımsız dış test, Mendeley üzerinden edinilen 329 panoramik
görüntü/maske çiftinden oluşmaktadır. İkinci aday dış kaynak olan STS-2D-Tooth
ile CDPR ve Mendeley kaynakları arasında exact hash ve perceptual hash taraması
yapılmıştır. Etiketli 900 örneğin 848'i olası örtüşme nedeniyle dışlanmış,
kalan 52 görüntü yalnızca keşifsel dış test olarak kullanılmıştır. Her iki dış
test kaynağı da eğitim, eşik seçimi, erken durdurma ve conformal kalibrasyona
dahil edilmemiştir.

### 2.2 Ön işleme

Görüntüler RGB biçimine dönüştürülmüş ve 512×512 piksele yeniden
boyutlandırılmıştır. Maskeler en yakın komşu enterpolasyonla aynı boyuta
getirilmiş ve diş/arka plan olacak şekilde ikili hale dönüştürülmüştür.
Baseline ve CLAHE deneyleri aynı veri ayrımları, model ve eğitim ayarlarıyla
yürütülmüştür; iki kol arasındaki kontrollü fark yalnızca CLAHE ön işlemesidir.

### 2.3 Model ve eğitim

Model, ImageNet başlangıç ağırlıklarıyla ResNet34 kodlayıcılı U-Net++ olarak
kurulmuştur. Çıkış tek kanallı logit haritasıdır. Optimizasyonda Dice ve
Binary Cross-Entropy kayıplarının eşit ağırlıklı birleşimi kullanılmıştır.
En iyi checkpoint doğrulama kaybına göre saklanmış; iyileşme durduğunda erken
durdurma uygulanmıştır. Baseline ve CLAHE deneyleri sabit seed ve aynı hasta
ayrımlarıyla karşılaştırılmıştır. CLAHE kolunda en iyi doğrulama kaybı 16.
epochta elde edilmiş ve eğitim 26. epochta erken durdurulmuştur.

### 2.4 Değerlendirme

Birincil metrik Dice, ikincil metrikler Intersection-over-Union (IoU) ve piksel
doğruluğudur. Metrikler görüntü bazında hesaplanmış ve veri kaynağı düzeyinde
özetlenmiştir. Yaş alt grubu sonuçları bootstrap %95 güven aralıklarıyla
verilmiştir. Çocuk alt grubunun örneklem sayısı düşük olduğundan bu analiz
keşifsel kabul edilmiştir. Önceden belirlenmiş yöntem karşılaştırmalarında
eşleştirilmiş bootstrap ve gerektiğinde Holm düzeltmesi kullanılacak şekilde
analiz akışı hazırlanmıştır.

### 2.5 Conformal analiz

Conformal eşik yalnızca ayrı kalibrasyon bölümünde, `alpha=0.10` hedef hata
düzeyiyle belirlenmiştir. Gerçek sınıfa atanan olasılığın tamamlayıcısı
nonconformity skoru olarak kullanılmıştır. Raporlanan güvence piksel düzeyinde
marjinal kapsamadır; aynı görüntü içindeki piksellerin uzamsal bağımlılığı
nedeniyle görüntü veya anatomik bölge düzeyinde garanti olarak
yorumlanmamalıdır. Bu sınırlama, görüntü segmentasyonunda conformal yöntemlerin
özel tasarım gerektirdiğini gösteren güncel çalışmalarla uyumludur [4,5].

## 3. Bulgular

### 3.1 Ana performans

| Test kaynağı | Yöntem | n | Dice | IoU | Piksel doğruluğu |
|---|---:|---:|---:|---:|---:|
| İç CDPR | Baseline | 360 | 0.9440 | 0.8952 | 0.9804 |
| İç CDPR | CLAHE | 360 | 0.9392 | 0.8866 | 0.9788 |
| Dış OPG | Baseline | 329 | 0.8924 | 0.8077 | 0.9730 |
| Dış OPG | CLAHE | 329 | 0.8993 | 0.8184 | 0.9744 |
| Temiz STS (keşifsel) | Baseline | 52 | 0.9000 | 0.8193 | 0.9696 |
| Temiz STS (keşifsel) | CLAHE | 52 | 0.8995 | 0.8186 | 0.9695 |

Bu karşılaştırma `results/published/model_performance.svg` dosyasında
görselleştirilmiştir.

### 3.2 CLAHE etkisi

- İç CDPR'da Dice: 0.9440 → 0.9392
- Dış OPG'de Dice: 0.8924 → 0.8993
- Temiz STS'de Dice: 0.9000 → 0.8995

### 3.3 Conformal sonuçlar

| Yöntem | α | Eşik | Ampirik piksel kapsaması | Belirsiz/boş oranı |
|---|---:|---:|---:|---:|
| Baseline | 0.10 | 0.002444 | 0.900798 | 0.098015 |
| CLAHE | 0.10 | 0.007710 | 0.901208 | 0.097311 |

Her iki kol da hedeflenen yaklaşık %90 marjinal piksel kapsamasına ulaşmıştır.
Sonuçlar `results/published/conformal_comparison.svg` dosyasında sunulmuştur.

### 3.4 Alt grup sonucu

| Yöntem | Alt grup | n | Dice | %95 GA |
|---|---|---:|---:|---:|
| Baseline | Yetişkin | 330 | 0.9466 | 0.9435–0.9495 |
| Baseline | Çocuk | 30 | 0.9150 | 0.9101–0.9202 |
| CLAHE | Yetişkin | 330 | 0.9412 | 0.9381–0.9444 |
| CLAHE | Çocuk | 30 | 0.9163 | 0.9124–0.9204 |

Çocuk alt grubu n=30 olduğu için bulgular keşifsel olarak yorumlanmalıdır.
Cinsiyet bilgisi bulunmadığından cinsiyet analizi yapılmamıştır. Güven
aralıkları `results/published/age_subgroup_dice.svg` dosyasında
görselleştirilmiştir.

## 4. Tartışma

Baseline model iç testte daha yüksek Dice üretirken, CLAHE dış OPG testinde
Dice'ı yaklaşık 0.0069 ve IoU'yu yaklaşık 0.0107 artırmıştır. Buna karşılık
temiz STS alt kümesinde yöntemler neredeyse aynı sonucu vermiştir. Dolayısıyla
CLAHE evrensel bir iyileştirme olarak değil, görüntü dağılımına bağlı bir
ön-işleme seçeneği olarak değerlendirilmelidir.

İç CDPR ile dış OPG arasındaki belirgin performans farkı, yalnızca iç test
sonucuna dayalı model seçiminin genellenebilirliği abartabileceğini
göstermektedir. Conformal katman her iki yöntemde de hedeflenen marjinal
kapsamaya ulaşmış olsa da yaklaşık %9.8 belirsiz/boş oranı üretmiştir. Bu çıktı
klinik karar yerine, uzman incelemesine yönlendirme sinyali olarak
yorumlanmalıdır.

Çalışmanın başlıca sınırlılıkları tek ana eğitim kaynağı, tek mimari, çocuk alt
grubundaki düşük örneklem sayısı ve ikinci dış testte sızıntı temizliği sonrası
yalnızca 52 örneğin kalmasıdır. Ayrıca piksel düzeyi conformal kapsam, anatomik
bölge veya hasta düzeyinde güvence sağlamaz. Gelecek çalışmalarda birden fazla
bağımsız merkez, farklı mimariler ve bölge düzeyi belirsizlik yöntemleri
değerlendirilmelidir.

## 5. Sonuç

Çalışma, yalnızca en yüksek iç test skorunu seçmek yerine dış-merkez
genellenebilirlik ve belirsizliğin birlikte raporlanmasının önemini
göstermektedir. CLAHE evrensel bir iyileştirme değildir; dış veri dağılımına
bağlı olarak yarar sağlayabilir.

## Veri ve kod erişilebilirliği

Kod, sabit yapılandırmalar ve kimliksiz toplu sonuçlar GitHub deposunda
yayımlanmaktadır. Lisans ve mahremiyet nedenleriyle ham radyografiler, maskeler,
hasta düzeyi kayıtlar ve model checkpoint dosyaları repoya eklenmemiştir.
CDPR verisi ilgili Scientific Data/Figshare kaynağından, dış OPG verisi ilgili
Mendeley Data kaynağından edinilmelidir.

## Etik beyan

Çalışmada yalnızca kaynak veri yayımlayıcıları tarafından kimliksizleştirilmiş
araştırma verileri kullanılmıştır. Veri kaynaklarının lisans ve atıf koşulları
izlenmelidir. Geliştirilen sistem araştırma amaçlıdır; klinik tanı aracı olarak
doğrulanmamıştır.

## Kaynaklar

1. Zhang Y, Ye F, Chen L, et al. Children’s dental panoramic radiographs
   dataset for caries segmentation and dental disease detection. *Scientific
   Data*. 2023;10:380. https://doi.org/10.1038/s41597-023-02237-5
2. Zhou Z, Siddiquee MMR, Tajbakhsh N, Liang J. UNet++: Redesigning skip
   connections to exploit multiscale features in image segmentation. *IEEE
   Transactions on Medical Imaging*. 2020;39(6):1856–1867.
   https://doi.org/10.1109/TMI.2019.2959609
3. Zuiderveld K. Contrast Limited Adaptive Histogram Equalization. In:
   *Graphics Gems IV*. Academic Press; 1994:474–485.
4. Brunekreef J, Marcus R, Oomen T, van der Heijden F. Kandinsky conformal
   prediction: efficient calibration of image segmentation algorithms.
   *CVPR*. 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Brunekreef_Kandinsky_Conformal_Prediction_Efficient_Calibration_of_Image_Segmentation_Algorithms_CVPR_2024_paper.html
5. Belhasin O, Romano Y, Freedman D, Rivlin E, Elad M. Conformal prediction
   for image segmentation using morphological prediction sets. 2025.
   https://arxiv.org/abs/2503.05618

## Yazım tamamlanmadan önce

- [ ] Danışmanın onayladığı kesin başlık
- [ ] Yazar sırası ve kurum bilgileri
- [ ] Hedef dergi formatı
- [ ] Kaynakça yönetimi ve DOI kontrolü
- [ ] Etik/lisans beyanının son kontrolü
- [ ] Nitel tahmin görsellerinin seçimi
