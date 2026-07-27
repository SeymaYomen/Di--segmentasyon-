# Panoramik Diş Röntgeni Segmentasyonunda Dış-Merkez Genellenebilirlik, CLAHE ve Conformal Belirsizlik Analizi

> Bilimsel içerik kesin sonuçlarla tamamlanmıştır. Gönderim öncesinde yalnızca
> yazar sırası, kurum bilgileri, hedef dergi biçimi ve danışman onayı eklenmelidir.

## Özet

**Amaç:** Panoramik dental radyografilerde U-Net++ tabanlı ikili diş
segmentasyonunun bağımsız veri dağılımlarındaki genellenebilirliğini incelemek,
Contrast-Limited Adaptive Histogram Equalization (CLAHE) ön işlemesinin etkisini
ölçmek ve piksel düzeyi conformal belirsizlik davranışını değerlendirmek.

**Yöntem:** Kimliksiz 2.398 görüntü/maske çifti kalite ve tekrar kontrollerinden
sonra hasta düzeyinde eğitim, kalibrasyon, doğrulama ve iç test kümelerine
ayrıldı. ResNet34 kodlayıcılı U-Net++ modeli, aynı ayrım ve eğitim ayarlarıyla
baseline ve CLAHE kollarında eğitildi. İç test 360 görüntüden oluştu. Model ayrıca
eğitime hiç katılmayan 329 görüntülük bağımsız OPG kümesinde ve sızıntı taraması
sonrası kalan 52 görüntülük keşifsel STS alt kümesinde değerlendirildi. Dice,
Intersection-over-Union (IoU) ve piksel doğruluğu görüntü bazında hesaplandı.
Yöntem farkları 5.000 tekrarlı eşleştirilmiş bootstrap ve dokuz karşılaştırma
için Holm düzeltmesiyle analiz edildi. Piksel düzeyi conformal eşik ayrı
kalibrasyon kümesinde, hedef hata düzeyi α=0,10 ile belirlendi.

**Bulgular:** İç CDPR testinde baseline daha yüksek Dice verdi
(0,9440'a karşı 0,9392; fark −0,0048; %95 GA −0,0058–−0,0039;
Holm-düzeltilmiş p=0,0036). Bağımsız OPG testinde CLAHE daha başarılıydı
(0,8993'e karşı 0,8924; fark +0,0069; %95 GA 0,0046–0,0094;
Holm-düzeltilmiş p=0,0036). Temiz STS alt kümesinde Dice farkı anlamlı değildi
(−0,0004; %95 GA −0,0024–0,0014; düzeltilmiş p=1,0000). Baseline ve CLAHE
sırasıyla 0,9008 ve 0,9012 ampirik piksel kapsamına ulaşırken belirsiz/boş
tahmin oranları 0,0980 ve 0,0973 oldu. İç testte yetişkin–çocuk Dice değerleri
baseline için 0,9466–0,9150, CLAHE için 0,9412–0,9163 idi.

**Sonuç:** CLAHE evrensel bir iyileştirme değildir; etkisi hedef veri
dağılımına bağlıdır. Yalnızca iç test başarısına dayalı seçim, gerçek
genellenebilirliği olduğundan yüksek gösterebilir. Bağımsız dış test,
eşleştirilmiş belirsizlik analizi ve conformal yönlendirme birlikte
raporlandığında sistemin güçlü ve zayıf yönleri daha şeffaf biçimde ortaya çıkar.

**Anahtar kelimeler:** panoramik radyografi; diş segmentasyonu; U-Net++;
CLAHE; dış doğrulama; conformal prediction; belirsizlik

## 1. Giriş

Panoramik radyografiler, maksilla ve mandibuladaki dentisyonu tek görüntüde
sunmaları nedeniyle dental tanı, tedavi planlama ve izlemde yaygın biçimde
kullanılır. Diş sınırlarının otomatik segmentasyonu; numaralandırma, çürük ve
restorasyon analizi ile nicel morfometrik değerlendirme gibi sonraki görevlerin
temelidir. Bu işlem elle yapıldığında zaman alıcıdır ve gözlemciler arası
değişkenliğe açıktır.

Derin öğrenme, panoramik görüntülerde diş segmentasyonu ve tanımlamasında güçlü
sonuçlar üretmiştir. Mask R-CNN tabanlı erken bir çalışma sınırlı sayıda
panoramik görüntüde ortalama IoU 0,877 bildirmiştir [1]. Daha geniş ve farklı
dentisyon dönemlerini içeren 6.046 radyografili bir çalışmada segmentasyon ve
numaralandırma için yaklaşık %92 IoU raporlanmıştır [2]. Güncel sistematik
derleme ve meta-analizler dental görüntülerde yapay zekânın genel olarak yüksek
duyarlılık ve özgüllüğe ulaşabildiğini, ancak çalışma heterojenliği ve dış
doğrulama eksikliği nedeniyle klinik önerinin hâlâ temkinli olması gerektiğini
göstermektedir [3].

Yüksek iç test başarısı tek başına klinik aktarılabilirlik kanıtı değildir.
Cihaz üreticisi, pozlama, çözünürlük, hasta yaşı, anatomik çeşitlilik ve maske
üretim protokolü değiştiğinde veri dağılımı da değişir. Bu nedenle eğitim
kaynağından bütünüyle ayrı merkezlerde test yapılması, modelin yalnızca
ezberlediği görüntü özelliklerini değil taşınabilir örüntüleri öğrenip
öğrenmediğini anlamak için gereklidir.

Bu çalışmanın ana eğitim kaynağı, çocuk ve yetişkin panoramik radyografilerini
ve diş maskelerini açık biçimde sunan CDPR veri çalışmasına dayanmaktadır [4].
Segmentasyon mimarisi olarak, kodlayıcı ve kod çözücü arasındaki anlamsal farkı
yoğun iç içe atlama bağlantılarıyla azaltan U-Net++ seçilmiştir [5]. Görüntüler
arasındaki yerel kontrast farklılıklarını azaltma olasılığı nedeniyle bir deney
kolunda CLAHE uygulanmıştır [6].

Segmentasyon skorlarının yanında modelin nerede güvenilmez olabileceğinin
gösterilmesi de önemlidir. Conformal prediction, dağılımsal varsayımları
sınırlı tutarak hedeflenen marjinal kapsama için tahmin kümeleri veya
belirsizlik bölgeleri üretir. Bununla birlikte görüntü pikselleri arasındaki
uzamsal bağımlılık, klasik piksel-bazlı garantilerin anatomik bölge veya hasta
düzeyi garanti gibi yorumlanmasını engeller. Son çalışmalar bu nedenle
segmentasyona özel kalibrasyon ve morfolojik tahmin kümeleri önermektedir [7,8].

Çalışmanın özgün katkısı yalnızca yüksek Dice raporlamak değil; kontrollü CLAHE
ablasyonunu, iki bağımsız dış değerlendirmeyi, eşleştirilmiş istatistiksel
karşılaştırmayı ve conformal belirsizlik analizini tek bir izlenebilir akışta
birleştirmektir. Araştırma soruları şunlardır:

1. İç testten bağımsız dış teste geçildiğinde performans ne kadar değişmektedir?
2. CLAHE genelleme farkını tutarlı biçimde azaltmakta mıdır?
3. Piksel düzeyi conformal kalibrasyon hedeflenen kapsamaya hangi
   belirsiz/boş tahmin oranıyla ulaşmaktadır?

## 2. Materyal ve Yöntem

### 2.1 Veri kaynakları ve veri sızıntısının önlenmesi

CDPR kaynağında dosya bütünlüğü, bozuk görüntü, boş maske ve decoded gri-seviye
SHA-256 tekrarı denetlendi. Toplam 2.398 temiz görüntü/maske çifti hasta
düzeyinde eğitim (n=1.451), kalibrasyon (n=242), doğrulama (n=345) ve iç test
(n=360) olarak ayrıldı. İç testte 330 yetişkin ve 30 çocuk görüntüsü bulunuyordu.

Birinci bağımsız dış test, Mendeley Data üzerinden sağlanan 329 panoramik
görüntü/maske çiftinden oluştu [9]. İkinci aday kaynak STS-2D-Tooth için CDPR ve
Mendeley kaynaklarına karşı exact hash ve perceptual hash taraması yapıldı.
Etiketli 900 örneğin 848'i olası örtüşme nedeniyle dışlandı; kalan 52 görüntü
yalnızca keşifsel dış testte kullanıldı [10]. Dış kaynaklar eğitim, erken
durdurma, maske eşiği seçimi ve conformal kalibrasyona hiçbir aşamada girmedi.

### 2.2 Ön işleme

Görüntüler RGB biçimine çevrilip 512×512 piksele yeniden boyutlandırıldı.
Maskeler en yakın komşu enterpolasyonla aynı boyuta getirildi ve diş/arka plan
olarak ikili hale dönüştürüldü. Baseline ve CLAHE deneyleri aynı hasta
ayrımları, seed, model, kayıp ve optimizasyon ayarlarıyla yürütüldü. Kontrollü
tek fark, CLAHE kolunda LAB renk uzayının parlaklık kanalına clip limit 2,0 ve
8×8 tile grid ile CLAHE uygulanmasıydı.

### 2.3 Model ve eğitim

Model, ImageNet başlangıç ağırlıklı ResNet34 kodlayıcılı U-Net++ olarak kuruldu.
Çıkış tek kanallı logit haritasıydı. Kayıp fonksiyonu eşit ağırlıklı Dice kaybı
ve Binary Cross-Entropy birleşimiydi. Adam optimizasyonu ve başlangıç öğrenme
oranı 1×10⁻⁴ kullanıldı. En iyi checkpoint doğrulama kaybına göre saklandı;
iyileşme durduğunda erken durdurma uygulandı. CLAHE kolunda en iyi doğrulama
kaybı 16. epochta elde edildi ve eğitim 26. epochta sonlandı.

### 2.4 Performans ve istatistiksel analiz

Birincil sonlanım Dice; ikincil sonlanımlar IoU ve piksel doğruluğuydu. Her
metrik önce görüntü bazında hesaplandı, sonra aritmetik ortalamayla veri kaynağı
düzeyinde özetlendi. Baseline ve CLAHE aynı görüntülerde değerlendirildiğinden,
yöntem farkları 5.000 tekrarlı eşleştirilmiş bootstrap ile analiz edildi.
Dokuz önceden belirlenmiş karşılaştırmanın p değerlerine Holm düzeltmesi
uygulandı. İki taraflı düzeltilmiş p<0,05 anlamlı kabul edildi. Yaş alt grubu
Dice ortalamaları 2.000 tekrarlı bootstrap %95 güven aralıklarıyla sunuldu.
Çocuk grubunun küçük olması nedeniyle bu analiz keşifsel kabul edildi.

### 2.5 Conformal belirsizlik

Conformal eşik yalnızca ayrılmış kalibrasyon kümesinde, α=0,10 hedef hata
düzeyiyle hesaplandı. Gerçek sınıfa atanan olasılığın tamamlayıcısı
nonconformity skoru olarak kullanıldı. Ampirik piksel kapsamı ve belirsiz/boş
tahmin oranı raporlandı. Bu garanti, piksellerin uzamsal bağımlılığı nedeniyle
görüntü, diş veya hasta düzeyi garanti olarak yorumlanmadı.

### 2.6 İzlenebilirlik ve nitel analiz

Her değerlendirme için yapılandırma, manifest, split ve checkpoint yolları ile
SHA-256 kimlikleri kaydedildi. Altı deneyde görüntü-bazlı CSV ortalamalarının
toplu JSON metrikleriyle 10⁻¹² toleransta eşleştiği doğrulandı. Nitel analizde
bağımsız OPG testindeki her görüntü için iki yöntemin ortalama Dice değeri
hesaplandı; en yüksek iki ve en düşük iki örnek, hasta/dosya kimlikleri
yayımlanmadan ortak panelde gösterildi. Yeşil doğru pozitifleri, kırmızı yanlış
pozitifleri ve turuncu yanlış negatifleri temsil etti.

## 3. Bulgular

### 3.1 Ana performans

| Test kaynağı | Yöntem | n | Dice | IoU | Piksel doğruluğu |
|---|---|---:|---:|---:|---:|
| İç CDPR | Baseline | 360 | 0,9440 | 0,8952 | 0,9804 |
| İç CDPR | CLAHE | 360 | 0,9392 | 0,8866 | 0,9788 |
| Dış OPG | Baseline | 329 | 0,8924 | 0,8077 | 0,9730 |
| Dış OPG | CLAHE | 329 | 0,8993 | 0,8184 | 0,9744 |
| Temiz STS (keşifsel) | Baseline | 52 | 0,9000 | 0,8193 | 0,9696 |
| Temiz STS (keşifsel) | CLAHE | 52 | 0,8995 | 0,8186 | 0,9695 |

İç testten dış OPG testine Dice düşüşü baseline için 0,0516, CLAHE için 0,0398
oldu. Böylece CLAHE dış OPG genelleme farkını yaklaşık 0,0118 Dice puanı
azalttı; ancak bu örüntü temiz STS alt kümesinde tekrarlanmadı.

### 3.2 Eşleştirilmiş yöntem karşılaştırması

| Veri kümesi | Metrik | CLAHE−Baseline | %95 GA | Holm p |
|---|---|---:|---:|---:|
| İç CDPR | Dice | −0,004814 | −0,005790–−0,003864 | 0,003599 |
| İç CDPR | IoU | −0,008616 | −0,010223–−0,006995 | 0,003599 |
| İç CDPR | Piksel doğruluğu | −0,001656 | −0,001985–−0,001336 | 0,003599 |
| Dış OPG | Dice | +0,006927 | 0,004556–0,009415 | 0,003599 |
| Dış OPG | IoU | +0,010711 | 0,007164–0,014421 | 0,003599 |
| Dış OPG | Piksel doğruluğu | +0,001445 | 0,000897–0,001994 | 0,003599 |
| Temiz STS | Dice | −0,000429 | −0,002443–0,001405 | 1,000000 |
| Temiz STS | IoU | −0,000654 | −0,003726–0,002144 | 1,000000 |
| Temiz STS | Piksel doğruluğu | −0,000039 | −0,000716–0,000591 | 1,000000 |

İç CDPR'da baseline üstünlüğü ve dış OPG'de CLAHE üstünlüğü tüm üç metrikte
istatistiksel olarak desteklenirken, temiz STS alt kümesinde yöntemler arasında
kanıtlanmış fark bulunmadı.

### 3.3 Conformal sonuçlar

| Yöntem | α | Eşik | Ampirik piksel kapsamı | Belirsiz/boş oranı |
|---|---:|---:|---:|---:|
| Baseline | 0,10 | 0,002444 | 0,900798 | 0,098015 |
| CLAHE | 0,10 | 0,007710 | 0,901208 | 0,097311 |

Her iki yöntem hedeflenen yaklaşık %90 marjinal piksel kapsamına ulaştı. CLAHE
biraz daha yüksek eşik gerektirmesine karşın belirsiz/boş oranı baseline'a çok
yakındı.

### 3.4 Yaş alt grubu

| Yöntem | Grup | n | Dice | %95 GA |
|---|---|---:|---:|---:|
| Baseline | Yetişkin | 330 | 0,9466 | 0,9435–0,9495 |
| Baseline | Çocuk | 30 | 0,9150 | 0,9101–0,9202 |
| CLAHE | Yetişkin | 330 | 0,9412 | 0,9381–0,9444 |
| CLAHE | Çocuk | 30 | 0,9163 | 0,9124–0,9204 |

Her iki yöntemde çocuk grubunun Dice değeri yetişkinlerden düşüktü. Örneklem
sayısı yalnızca 30 olduğundan sonuç hipotez üretici olarak değerlendirilmelidir.
Cinsiyet bilgisi bulunmadığı için cinsiyet analizi yapılmadı.

### 3.5 Nitel bulgular

İyi örneklerde her iki yöntem diş kronları ve köklerin büyük bölümünü doğru
segmentlerken hata bölgeleri ağırlıkla komşu diş temasları ve düşük kontrastlı
apikal sınırlarla sınırlı kaldı. Zor örneklerde yanlış negatifler kök uçları ve
örtüşen anatomik yapılarda, yanlış pozitifler ise yoğun restorasyonlar ve
mandibular kortikal sınıra yakın bölgelerde yoğunlaştı. Bu örnekler yalnızca
görsel açıklama amacı taşır; nicel sonuçların yerine geçmez.

## 4. Tartışma

Çalışmanın temel bulgusu, ön işlemenin etkisinin veri dağılımına bağlı olmasıdır.
Baseline iç testte anlamlı biçimde daha başarılıyken CLAHE bağımsız OPG
kaynağında anlamlı iyileşme sağladı. Temiz STS alt kümesinde ise fark yoktu.
Dolayısıyla “CLAHE modeli daha iyidir” biçiminde evrensel bir sonuç
çıkarılamaz. Daha doğru yorum, CLAHE'nin belirli kontrast ve cihaz
dağılımlarındaki alan kaymasını kısmen azaltabildiğidir.

İç testteki 0,9440 Dice, literatürde panoramik segmentasyon için bildirilen
güçlü sonuçlarla uyumludur [1,2]; ancak görev, veri ve metrik toplama biçimleri
farklı olduğundan doğrudan üstünlük karşılaştırması yapılamaz. Daha önemli
bulgu, aynı modelin dış OPG'de baseline Dice değerinin 0,8924'e düşmesidir. Bu
yaklaşık 5,2 yüzde puanlık fark, yalnızca tek-kaynak iç test raporlamasının
genellenebilirliği abartabileceğini somutlaştırır.

Eşleştirilmiş bootstrap, küçük görünen farkların yalnızca yuvarlama veya
rastlantı olmadığını gösterdi. İç CDPR ve dış OPG sonuçlarının ters yönde
olması, model seçiminin hedef kullanım bağlamına göre yapılması gerektiğini
vurgular. Bu nedenle araştırma arayüzünde CLAHE dış-genellenebilirlik odaklı
geçici varsayılan olarak sunulabilir; fakat kullanıcı aktif modeli açıkça
görmeli ve baseline seçeneği korunmalıdır. STS sonucu, CLAHE için evrensel
üstünlük iddiasını özellikle sınırlar.

Conformal katman her iki yöntemde hedeflenen marjinal piksel kapsamına ulaştı.
Yaklaşık %9,8 belirsiz/boş oranı, sistemin zor bölgeleri otomatik onaylamak
yerine uzman incelemesine yönlendirebileceği bir güvenlik sinyali sağlar.
Bununla birlikte bu bulgu klinik güvenlik garantisi değildir. Piksel
bağımlılığı, alan kayması ve kalibrasyon kaynağının değişmesi kapsam
garantisinin hasta düzeyine taşınmasını engeller [7,8].

Çocuk alt grubundaki daha düşük skor, farklı dentisyon evreleri, sürmekte olan
dişler ve anatomik örtüşmelerle ilişkili olabilir. Bununla birlikte çocuk
örneklem sayısı düşük olduğundan bu açıklama doğrulayıcı değil keşifseldir.
Daha dengeli, merkezler arası pediatrik veriyle önceden tanımlanmış alt grup
çalışmaları gereklidir.

Başlıca sınırlılıklar tek ana eğitim kaynağı, tek mimari ailesi, sınırlı
pediatrik örneklem ve STS taramasından sonra yalnızca 52 temiz örneğin
kalmasıdır. Perceptual hash olası örtüşmeyi azaltır ancak hasta kimliği olmayan
kaynaklarda tüm hasta-düzeyi ilişkileri kesin olarak dışlayamaz. Nitel örnek
seçimi metrik uçlarına dayalıdır ve prevalansı temsil etmez. Son olarak sistem
klinik tanı amacıyla prospektif olarak doğrulanmamıştır.

## 5. Sonuç

U-Net++ panoramik diş segmentasyonunda yüksek iç test başarısı sağlamış, ancak
bağımsız dış test performansı daha düşük kalmıştır. CLAHE iç testte küçük fakat
anlamlı bir kayba, bağımsız OPG testinde ise küçük fakat anlamlı bir kazanca yol
açmış; ikinci keşifsel dış testte fark yaratmamıştır. Bu sonuçlar tek bir
“evrensel en iyi” ön işleme seçeneği yerine çok-merkezli doğrulama ve hedef
dağılıma duyarlı model seçimini desteklemektedir. Conformal katman, model
çıktılarına uzman incelemesi için kullanılabilecek bir belirsizlik işareti
eklemiş; ancak klinik veya hasta düzeyi garanti sunmamıştır.

## Veri ve kod erişilebilirliği

Kod, sabit yapılandırmalar, kimliksiz toplu sonuçlar, istatistiksel analizler,
sağlık kontrolü ve figür üretim komutları GitHub deposunda yayımlanacaktır.
Lisans, mahremiyet ve dosya boyutu nedenleriyle ham radyografiler, maskeler,
hasta düzeyi kayıtlar ve checkpoint dosyaları repoya eklenmeyecektir. CDPR
Scientific Data/Figshare, birinci dış OPG Mendeley Data ve STS ilgili açık
depolardan kendi kullanım koşulları altında edinilmelidir.

## Etik beyan

Yalnızca kaynak yayıncılar tarafından kimliksizleştirilmiş araştırma verileri
kullanılmıştır. Veri kaynaklarının lisans, atıf ve ikincil kullanım koşulları
izlenmelidir. Geliştirilen sistem araştırma demosudur; klinik tanı veya tedavi
kararı amacıyla doğrulanmamıştır.

## Kaynaklar

1. Jader G, Fontineli J, Ruiz M, Abdalla K, Pithon M, Oliveira L. Deep
   instance segmentation of teeth in panoramic X-ray images. *SIBGRAPI*.
   2018:400–407. https://doi.org/10.1109/SIBGRAPI.2018.00058
2. Tuzoff DV, Tuzova LN, Bornstein MM, et al. Robust automated tooth detection
   and numbering in panoramic radiographs using artificial intelligence.
   *Journal of Dentistry*. 2023;137:104671.
   https://doi.org/10.1016/j.jdent.2023.104671
3. Albalawi F, et al. Diagnostic performance of artificial intelligence in
   panoramic radiography: a systematic review and meta-analysis. 2025.
   https://pubmed.ncbi.nlm.nih.gov/40739210/
4. Zhang Y, Ye F, Chen L, et al. Children's dental panoramic radiographs
   dataset for caries segmentation and dental disease detection.
   *Scientific Data*. 2023;10:380.
   https://doi.org/10.1038/s41597-023-02237-5
5. Zhou Z, Siddiquee MMR, Tajbakhsh N, Liang J. UNet++: Redesigning skip
   connections to exploit multiscale features in image segmentation.
   *IEEE Transactions on Medical Imaging*. 2020;39(6):1856–1867.
   https://doi.org/10.1109/TMI.2019.2959609
6. Zuiderveld K. Contrast Limited Adaptive Histogram Equalization. In:
   *Graphics Gems IV*. Academic Press; 1994:474–485.
7. Brunekreef J, Marcus R, Oomen T, van der Heijden F. Kandinsky conformal
   prediction: efficient calibration of image segmentation algorithms.
   *CVPR*. 2024.
   https://openaccess.thecvf.com/content/CVPR2024/html/Brunekreef_Kandinsky_Conformal_Prediction_Efficient_Calibration_of_Image_Segmentation_Algorithms_CVPR_2024_paper.html
8. Belhasin O, Romano Y, Freedman D, Rivlin E, Elad M. Conformal prediction
   for image segmentation using morphological prediction sets. 2025.
   https://arxiv.org/abs/2503.05618
9. Panoramic Dental X-ray Segmentation Dataset. Mendeley Data.
   https://data.mendeley.com/datasets/jrz4nj82zv/1
10. Wang et al. STS-Tooth: a large-scale 2D and 3D dental dataset.
    *Scientific Data*. 2025. https://doi.org/10.5281/zenodo.10597292

## Gönderim öncesi idari kontrol

- [ ] Kesin yazar sırası ve ORCID bilgileri
- [ ] Kurum ve sorumlu yazar bilgileri
- [ ] Hedef derginin yazım/kelime/şekil biçimi
- [ ] Danışman onaylı başlık
- [ ] Veri lisansı ve etik beyanının kurum tarafından son kontrolü
