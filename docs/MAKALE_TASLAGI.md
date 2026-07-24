# Makale Taslağı

> Çalışma başlığı ve metin danışman görüşüyle sonlandırılacaktır. Köşeli parantezli alanlar deneyler tamamlanınca doldurulacaktır.

## Önerilen başlık

**Panoramik Diş Röntgeni Segmentasyonunda Dış-Merkez Genellenebilirlik ve Conformal Belirsizlik: CLAHE Ön İşlemenin Etkisi**

## Özgün katkı

Bu çalışmanın katkısı yalnızca yüksek Dice skoru elde etmek değildir. Ana katkı; aynı segmentasyon modelinin iç test ve bağımsız dış test davranışını birlikte incelemek, CLAHE ön işlemenin genelleme farkına etkisini ölçmek ve conformal prediction kapsamasının dağılım değişiminde nasıl davrandığını raporlamaktır.

## Araştırma soruları

1. İç CDPR testinden bağımsız External OPG testine geçildiğinde segmentasyon performansı ne kadar değişmektedir?
2. CLAHE ön işleme, iç ve dış test arasındaki genelleme farkını azaltmakta mıdır?
3. Kalibrasyon kümesinde belirlenen conformal eşik, iç ve dış testte hedef kapsama düzeyini ne ölçüde korumaktadır?

## Özet taslağı

**Amaç:** Panoramik diş röntgeni segmentasyonunda merkezler arası genellenebilirliği, CLAHE ön işlemenin etkisini ve conformal belirsizlik davranışını incelemek.

**Yöntem:** U-Net++ modeli, denetlenmiş ve hasta-bazlı ayrılmış Children's Dental Panoramic Radiographs verisi üzerinde eğitildi. Model seçimi doğrulama kümesiyle, conformal eşik ayrı kalibrasyon kümesiyle yapıldı. Bağımsız dış test için eğitim ve hiperparametre seçiminde kullanılmayan 329 görüntülük Mendeley External OPG verisi kullanıldı. Baseline ve CLAHE koşulları aynı bölmeler ve değerlendirme kurallarıyla karşılaştırıldı.

**Bulgular:** Baseline model iç testte Dice 0.94397, IoU 0.89521 ve piksel doğruluğu 0.98044; dış testte sırasıyla 0.89242, 0.80766 ve 0.97300 elde etti. İç testten dış teste Dice 5.16, IoU 8.76 ve piksel doğruluğu 0.74 yüzde puan düştü. CLAHE ve conformal sonuçları: [tamamlanacak].

**Sonuç:** İlk bulgular yüksek iç test başarımının dış merkezde aynı ölçüde korunmadığını göstermektedir. CLAHE ve conformal analizler tamamlandığında, ön işlemenin genellenebilirliğe ve belirsizlik kapsamasına etkisi birlikte değerlendirilecektir.

## 1. Giriş

- Panoramik radyografilerde otomatik diş segmentasyonunun kullanım alanı ve önemi.
- Derin öğrenme modellerinin kurum/cihaz/protokol değişimlerinde performans kaybı sorunu.
- Kontrast farklılıklarına karşı CLAHE kullanımının olası yararı ve aşırı iyileştirme riski.
- Yüksek ortalama performansın tek başına klinik güveni göstermemesi; belirsizlik ve kapsama ihtiyacı.
- Literatür boşluğu: iç/dış test, ön işleme ve conformal kapsamanın tek deney düzeninde birlikte incelenmesi.

## 2. Materyal ve yöntem

### 2.1 Veri kaynakları

**CDPR/Figshare geliştirme verisi:** Denetim sonrası 2.398 temiz örnek. 485 kesin kopya ve 2 boş maske analizden çıkarıldı. Bölmeler: eğitim 1.451, kalibrasyon 242, doğrulama 345 ve iç test 360. İç test 330 yetişkin ve 30 çocuk görüntüsünden oluştu.

**Mendeley External OPG:** 329 görüntü-maske çifti yalnızca bağımsız dış test için kullanıldı; eğitim, model seçimi, eşik veya hiperparametre ayarında kullanılmadı.

**InReDD PAN924:** PhysioNet erişim başvurusu incelemededir. Erişim ve format doğrulaması tamamlanırsa önceden tanımlı ikinci dış test olarak değerlendirilecektir; aksi halde makalenin zorunlu bileşeni değildir.

STS-Tooth, kaynak örtüşmesi ve veri sızıntısı riski nedeniyle çalışmaya dahil edilmedi.

### 2.2 Veri kalitesi ve sızıntı önleme

- Kesin kopyalar, bölme işleminden önce görüntü hash'iyle çıkarıldı.
- Bölmeler hasta bazında oluşturuldu.
- Dış test kaynakları eğitim ve model seçimi dışında tutuldu.
- Ham görüntüler, maskeler ve hasta bilgileri açık kod deposunda yayımlanmadı.

### 2.3 Ön işleme

İki karşılaştırılabilir deney koşulu tanımlandı:

- Baseline: standart yeniden boyutlandırma/normalizasyon, CLAHE yok.
- CLAHE: aynı işlem hattı ve veri bölmeleri, yalnızca ön işleme modu CLAHE.

CLAHE parametreleri ve uygulama sırası final kod/config dosyasından aynen raporlanacaktır.

### 2.4 Model ve eğitim

- Mimari: U-Net++.
- Encoder: ResNet-34.
- Kayıp: Dice + BCE.
- Optimizasyon: Adam, başlangıç öğrenme oranı 1e-4.
- Maksimum epoch: 100; erken durdurma sabrı: 10.
- Ana deney tohumu: 42.

Tek seed sonucu nihai sağlamlık iddiası için yeterli değildir. Kaynak izin verirse seçili deneyler ek seed'lerle tekrarlanacak veya bootstrap güven aralıklarıyla belirsizlik raporlanacaktır.

### 2.5 Değerlendirme

Birincil metrik Dice; ikincil metrikler IoU ve piksel doğruluğudur. Sonuçlar görüntü düzeyindeki metriklerden özetlenecek ve bootstrap %95 güven aralıkları raporlanacaktır. İç-dış fark mutlak yüzde puan olarak verilecektir.

### 2.6 Conformal prediction

Kalibrasyon yalnızca ayrılmış kalibrasyon kümesinde yapılacaktır. Hedef hata düzeyi alpha=0.10'dur. Mevcut uygulama piksel örneklemeli marjinal kapsama verir; piksellerin uzamsal bağımlılığı nedeniyle garanti görüntü veya bölge düzeyinde yorumlanmayacaktır. İç ve dış test kapsaması ayrı raporlanacaktır.

### 2.7 Alt grup ve istatistik planı

- Yetişkin/çocuk iç test karşılaştırması keşifsel olacaktır; çocuk grubu n=30'dur.
- Küçük alt gruplarda bootstrap güven aralığı ve etki büyüklüğü önceliklidir.
- Cinsiyet bilgisi yoksa cinsiyet analizi yapılmayacaktır.
- Birden çok önceden tanımlı hipotez testi uygulanırsa Holm düzeltmesi kullanılacaktır.
- Dış test verisi model veya eşik seçmek için kullanılmayacaktır.

## 3. Bulgular

### 3.1 Baseline sonuçları

| Koşul | Test kümesi | n | Dice | IoU | Piksel doğruluğu |
|---|---|---:|---:|---:|---:|
| Baseline | İç CDPR | 360 | 0.94397 | 0.89521 | 0.98044 |
| Baseline | Dış OPG | 329 | 0.89242 | 0.80766 | 0.97300 |

İç testten dış teste düşüş Dice için 5.16, IoU için 8.76 ve piksel doğruluğu için 0.74 yüzde puandır.

### 3.2 CLAHE karşılaştırması

| Koşul | İç Dice | Dış Dice | Dice genelleme farkı | İç IoU | Dış IoU |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.94397 | 0.89242 | 5.16 yp | 0.89521 | 0.80766 |
| CLAHE | [bekleniyor] | [bekleniyor] | [bekleniyor] | [bekleniyor] | [bekleniyor] |

### 3.3 Conformal sonuçları

| Model | Test kümesi | Hedef kapsama | Ampirik kapsama | Belirsiz/boş küme oranı |
|---|---|---:|---:|---:|
| Baseline | İç CDPR | 0.90 | [bekleniyor] | [bekleniyor] |
| Baseline | Dış OPG | 0.90 | [bekleniyor] | [bekleniyor] |
| CLAHE | İç CDPR | 0.90 | [bekleniyor] | [bekleniyor] |
| CLAHE | Dış OPG | 0.90 | [bekleniyor] | [bekleniyor] |

### 3.4 Alt grup sonuçları

Yetişkin/çocuk sonuçları örneklem büyüklüğü ve bootstrap %95 güven aralıklarıyla sunulacaktır. Çocuk sonuçları doğrulayıcı değil keşifsel olarak yorumlanacaktır.

## 4. Tartışma

- Baseline modelin dış testteki Dice ve özellikle IoU düşüşünün anlamı.
- Piksel doğruluğunun sınıf dengesizliği nedeniyle tek başına yanıltıcı olabilmesi.
- CLAHE'nin iç performansı artırıp dış performansı düşürmesi veya tersi olasılığının değerlendirilmesi.
- Conformal kapsamanın dağılım değişiminde bozulmasının güven katmanı açısından önemi.
- Tek dış merkezin genellenebilirlik iddiasını sınırlaması; InReDD'nin yalnızca erişim sağlanırsa ek doğrulama olarak kullanılması.

## 5. Sınırlılıklar

- Şu aşamada tek bağımsız dış test kaynağı.
- Çocuk iç test alt grubunun küçük olması.
- Mevcut conformal yaklaşımın piksel düzeyinde marjinal kapsama sağlaması.
- Tek ana seed ile eğitim yapılmış olması.
- Panoramik görüntüler ve ikili diş/arka plan göreviyle sınırlı kapsam.

## 6. Sonuç

Çalışma, iç testte yüksek segmentasyon başarımının dış merkez genellenebilirliğini garanti etmediğini nicel olarak göstermektedir. Nihai sonuç, CLAHE'nin bu farkı azaltıp azaltmadığı ve conformal kapsamanın dış dağılımda ne ölçüde korunduğu birlikte değerlendirildikten sonra yazılacaktır.

## Raporlama kontrolü

- [ ] Veri setlerinin tam atıfları ve lisansları kaynakçada doğrulandı.
- [ ] CLAHE parametreleri config ile birebir raporlandı.
- [ ] Checkpoint/resume yöntemi açıklandı.
- [ ] Görüntü düzeyi bootstrap %95 güven aralıkları eklendi.
- [ ] İç/dış conformal kapsama ayrı verildi.
- [ ] Alt grup örneklem sayıları tabloya eklendi.
- [ ] Ham/hasta düzeyi veriler yayımlanmadı.
- [ ] Kod ve anonim toplu sonuçların commit kimlikleri arşivlendi.
