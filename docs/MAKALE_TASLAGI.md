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

- Panoramik röntgen segmentasyonunun klinik iş akışlarındaki yeri
- Tek-merkez performansının dış-merkez genellenebilirliği garanti etmemesi
- Görüntü kontrast farklılıkları ve CLAHE gerekçesi
- Tıbbi yapay zekâda hata kadar belirsizliğin de raporlanması gereği
- Çalışmanın üç araştırma sorusu ve katkıları

## 2. Materyal ve Yöntem

### 2.1 Veri kümeleri

- CDPR: kaynak, lisans, temizleme, 2.398 çift ve hasta bazlı bölme
- Mendeley OPG: 329 bağımsız dış test örneği
- STS-2D-Tooth: 900 etiketli örnekte sızıntı taraması, 848 dışlama, 52 temiz örnek
- Dış testlerin eğitim ve model seçimine dahil edilmediğinin açık beyanı

### 2.2 Ön işleme

- RGB dönüşümü ve 512×512 yeniden boyutlandırma
- Baseline ile CLAHE kollarının tek farkı
- Maske değerlerinin ikili hale getirilmesi

### 2.3 Model ve eğitim

- U-Net++ ve ResNet34 encoder
- Dice + BCE kaybı
- Doğrulama kaybına göre checkpoint
- Erken durdurma ve yeniden üretilebilir seed

### 2.4 Değerlendirme

- Dice, IoU ve piksel doğruluğu
- Görüntü başına metrikler
- Eşleştirilmiş bootstrap %95 güven aralıkları
- Sınırlı sayıdaki karşılaştırmalar için Holm düzeltmesi

### 2.5 Conformal analiz

- Ayrı calibration bölümü
- Nonconformity skoru ve alpha=0.10
- Marjinal piksel kapsamasının kapsamı ve uzamsal bağımlılık sınırlaması

## 3. Bulgular

### 3.1 Ana performans

`results/published/final_metrics.csv` tablosu kullanılacak.

### 3.2 CLAHE etkisi

- İç CDPR'da Dice: 0.9440 → 0.9392
- Dış OPG'de Dice: 0.8924 → 0.8993
- Temiz STS'de Dice: 0.9000 → 0.8995

### 3.3 Conformal sonuçlar

- Baseline kapsama: 0.9008
- CLAHE kapsama: 0.9012
- Belirsiz/boş oranları yaklaşık %9.8

### 3.4 Alt grup sonucu

Çocuk alt grubu n=30 olduğu için yalnızca keşifsel olarak ve bootstrap güven
aralığıyla sunulacak. Cinsiyet verisi olmadığından cinsiyet analizi yapılmayacak.

## 4. Tartışma

- CLAHE'nin veri dağılımına bağlı etkisi
- Yüksek iç test skorunun dış test performansını tek başına garanti etmemesi
- Conformal katmanın “inceleme gerekli” işaretine olası katkısı
- STS temiz alt kümesinin küçük olması
- Tek mimari, tek ana eğitim kaynağı ve piksel düzeyi conformal garanti sınırlamaları

## 5. Sonuç

Çalışma, yalnızca en yüksek iç test skorunu seçmek yerine dış-merkez
genellenebilirlik ve belirsizliğin birlikte raporlanmasının önemini
göstermektedir. CLAHE evrensel bir iyileştirme değildir; dış veri dağılımına
bağlı olarak yarar sağlayabilir.

## Yazım tamamlanmadan önce

- [ ] Danışmanın onayladığı kesin başlık
- [ ] Yazar sırası ve kurum bilgileri
- [ ] Hedef dergi formatı
- [ ] Kaynakça yönetimi ve DOI kontrolü
- [ ] Etik/lisans beyanının son kontrolü
- [ ] Nitel tahmin görsellerinin seçimi

