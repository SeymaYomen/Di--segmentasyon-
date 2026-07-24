# Proje Durumu

Son güncelleme: 24 Temmuz 2026

## Araştırma odağı

Bu çalışma, panoramik diş röntgenlerinde ikili diş segmentasyonunun iç merkez ve bağımsız dış merkez verilerindeki başarımını; CLAHE ön işlemenin genellenebilirliğe etkisini ve conformal prediction ile belirsizlik/kapsama davranışını inceler.

## Veri setleri ve roller

| Veri seti | Kullanılan örnek | Rol | Lisans/durum |
|---|---:|---|---|
| Children's Dental Panoramic Radiographs (Figshare 21621705) | 2.398 temiz örnek | Eğitim, kalibrasyon, doğrulama ve iç test | CC0 |
| Mendeley Panoramic Dental X-ray Segmentation Dataset | 329 | Birinci bağımsız dış test | CC BY 4.0 |
| AKUDENTAL v1.0 | 333 yetişkin panoramik görüntü | Dönüşüm ve denetim sonrası ikinci bağımsız dış test | CC BY-NC-SA 4.0; yalnızca ticari olmayan akademik kullanım |

InReDD/PAN924 erişim bekleme süresi nedeniyle güncel deney planından çıkarılmıştır. OdontoAI erişim platformu sonlandırıldığı için zorunlu veri kaynağı değildir. STS-Tooth, kaynak örtüşmesi ve veri sızıntısı riski nedeniyle çalışmaya dahil edilmeyecektir. Ham görüntüler, maskeler ve hasta bilgileri GitHub'a yüklenmez.

### AKUDENTAL kabul koşulları

AKUDENTAL, Akdeniz Üniversitesi Diş Hastanesinde iki cihazla toplanan bağımsız bir yetişkin kohortudur. Kaynak dosyalar COCO biçiminde instance poligonlarıdır; hazır ikili maske değildir. İkinci dış testte kullanılmadan önce:

1. 32 doğal diş sınıfının poligonları tek bir ikili diş maskesinde birleştirilecek.
2. Köprü ve dolgu/kron restorasyon sınıfları birincil diş maskesine eklenmeyecek.
3. İmplant sınıfı, ana etiket uyumu belirlendikten sonra önceden tanımlı birincil veya duyarlılık analizi kuralıyla ele alınacak.
4. Görüntü-maske eşleşmesi, boş maske, boyut ve poligon sınırı denetimleri çalıştırılacak.
5. CDPR ve Mendeley dış test ile kesin/perceptual hash karşılaştırması yapılacak.
6. Dönüştürülmüş örneklerden rastgele bir alt küme görsel olarak doğrulanacak.

Bu kontroller tamamlanmadan AKUDENTAL sonuçları makalede raporlanmayacaktır.

## Veri denetimi ve bölme

- 485 kesin kopya ve 2 boş maske çıkarıldı.
- Temiz CDPR örnek sayısı: 2.398.
- Bölme: eğitim 1.451, kalibrasyon 242, doğrulama 345, iç test 360.
- İç test: 330 yetişkin, 30 çocuk.
- Çocuk alt grup sonucu küçük örneklem nedeniyle keşifsel olarak, bootstrap güven aralığıyla raporlanacaktır.

## Tamamlanan baseline

İç test (n=360):

- Dice: 0.94397
- IoU: 0.89521
- Piksel doğruluğu: 0.98044

Mendeley dış test (n=329):

- Dice: 0.89242
- IoU: 0.80766
- Piksel doğruluğu: 0.97300

İç testten dış teste düşüş:

- Dice: 5.16 yüzde puan
- IoU: 8.76 yüzde puan
- Piksel doğruluğu: 0.74 yüzde puan

## CLAHE deneyi

CLAHE eğitimi 11 epoch tamamladı ve 12. epoch sırasında kesildi. Drive'da 10. epoch civarındaki en iyi model ağırlıkları bulunmaktadır. Mevcut eğitim kodu optimizer, scheduler, epoch ve erken-durdurma durumunu checkpoint'e kaydetmediği için bu dosya henüz gerçek bir eğitim devam checkpoint'i değildir. GPU tekrar kullanılmadan önce resume desteği eklenip test edilecektir. Aksi halde checkpoint yalnızca ağırlık başlangıcı (warm start) olarak kullanılabilir.

## Conformal prediction

Mevcut yaklaşım piksel örneklemeli, marjinal piksel kapsamasını ölçen bir başlangıçtır. Uzamsal bağımlılık nedeniyle garanti görüntü/bölge düzeyinde yorumlanmayacaktır. Kalibrasyon, iç test ve her dış test sonucu ayrı raporlanacaktır. Conformal çalıştırma model çıkarımı içerdiğinden, hazır olasılık çıktıları yoksa salt CPU istatistiği olarak değerlendirilmemelidir.

## GPU kullanım kuralı

GPU yalnızca şu işler için açılır:

1. Doğrulanmış checkpoint/resume mekanizmasıyla CLAHE eğitimi.
2. Gerekli model çıkarımları (CLAHE değerlendirme ve gerekirse conformal olasılık üretimi).

Dosya denetimi, COCO-poligon dönüşümü, hash kontrolü, bootstrap, güven aralıkları, alt grup analizi, tablo/grafik üretimi, GitHub düzenleme, arayüz ve makale yazımı CPU'da yapılır.

## Kalan işler

- [ ] Eğitim koduna gerçek resume checkpoint desteği ekle ve küçük bir testle doğrula.
- [ ] CLAHE eğitimini kontrollü biçimde tamamla.
- [ ] CLAHE iç ve Mendeley dış test metriklerini üret.
- [ ] AKUDENTAL COCO poligonlarını ikili maskeye dönüştür ve veri denetimini tamamla.
- [ ] AKUDENTAL'ı ikinci dış test olarak baseline ve CLAHE modelleriyle değerlendir.
- [ ] Baseline/CLAHE için conformal kalibrasyon ve kapsama analizini tamamla.
- [ ] Alt grup bootstrap güven aralıkları ve gerektiğinde Holm düzeltmesi uygula.
- [ ] Anonim metrikleri ve yayınlanabilir grafikleri results/published altında yayımla.
- [ ] Streamlit arayüzünü hazırla.
- [ ] Makale taslağını üç ana araştırma sorusu etrafında tamamla.

## Tekrarlanabilirlik notu

Yayınlanan kod/config dosyaları sabit seed, veri bölmesi kuralları ve ön işleme seçeneklerini belgelemelidir. Veri dosyaları ve .pth model ağırlıkları depoya eklenmez. AKUDENTAL kullanılırsa veri seti ve makale uygun biçimde atıflanacak, türetilmiş maskeler ticari olmayan ve aynı lisans koşullarıyla ele alınacaktır.
# Proje Durumu

Son güncelleme: 24 Temmuz 2026

## Araştırma odağı

Bu çalışma, panoramik diş röntgenlerinde ikili diş segmentasyonunun iç merkez ve bağımsız dış merkez verilerindeki başarımını; CLAHE ön işlemenin genellenebilirliğe etkisini ve conformal prediction ile belirsizlik/kapsama davranışını inceler.

## Veri setleri ve roller

| Veri seti | Kullanılan örnek | Rol | Lisans/durum |
|---|---:|---|---|
| Children's Dental Panoramic Radiographs (Figshare 21621705) | 2.398 temiz örnek | Eğitim, kalibrasyon, doğrulama ve iç test | CC0 |
| Mendeley Panoramic Dental X-ray Segmentation Dataset | 329 | Yalnızca bağımsız dış test | CC BY 4.0 |
| InReDD PAN924 | Bekleniyor | Onay sonrası ikinci dış test adayı | PhysioNet erişim başvurusu incelemede |

OdontoAI veri erişimi sonlandırıldığı için zorunlu veri kaynağı değildir. STS-Tooth, kaynak örtüşmesi ve veri sızıntısı riski nedeniyle bu çalışmanın eğitim/test havuzuna dahil edilmeyecektir. Ham görüntüler, maskeler ve hasta bilgileri GitHub'a yüklenmez.

## Veri denetimi ve bölme

- 485 kesin kopya ve 2 boş maske çıkarıldı.
- Temiz örnek sayısı: 2.398.
- Bölme: eğitim 1.451, kalibrasyon 242, doğrulama 345, iç test 360.
- İç test: 330 yetişkin, 30 çocuk.
- Çocuk alt grup sonucu küçük örneklem nedeniyle keşifsel olarak, bootstrap güven aralığıyla raporlanacaktır.

## Tamamlanan baseline

İç test (n=360):

- Dice: 0.94397
- IoU: 0.89521
- Piksel doğruluğu: 0.98044

Mendeley dış test (n=329):

- Dice: 0.89242
- IoU: 0.80766
- Piksel doğruluğu: 0.97300

İç testten dış teste düşüş:

- Dice: 5.16 yüzde puan
- IoU: 8.76 yüzde puan
- Piksel doğruluğu: 0.74 yüzde puan

## CLAHE deneyi

CLAHE eğitimi 11 epoch tamamladı ve 12. epoch sırasında kesildi. Drive'da 10. epoch civarındaki en iyi model ağırlıkları bulunmaktadır. Mevcut eğitim kodu optimizer, scheduler, epoch ve erken-durdurma durumunu checkpoint'e kaydetmediği için bu dosya henüz gerçek bir eğitim devam checkpoint'i değildir. GPU tekrar kullanılmadan önce resume desteği eklenip test edilecektir. Aksi halde checkpoint yalnızca ağırlık başlangıcı (warm start) olarak kullanılabilir.

## Conformal prediction

Mevcut yaklaşım piksel örneklemeli, marjinal piksel kapsamasını ölçen bir başlangıçtır. Uzamsal bağımlılık nedeniyle garanti görüntü/bölge düzeyinde yorumlanmayacaktır. Kalibrasyon, iç test ve dış test sonuçları ayrı raporlanacaktır. Conformal çalıştırma model çıkarımı içerdiğinden, hazır olasılık çıktıları yoksa salt CPU istatistiği olarak değerlendirilmemelidir.

## GPU kullanım kuralı

GPU yalnızca şu işler için açılır:

1. Doğrulanmış checkpoint/resume mekanizmasıyla CLAHE eğitimi.
2. Gerekli model çıkarımları (CLAHE değerlendirme ve gerekirse conformal olasılık üretimi).

Dosya denetimi, bootstrap, güven aralıkları, alt grup analizi, tablo/grafik üretimi, GitHub düzenleme, arayüz ve makale yazımı CPU'da yapılır.

## Kalan işler

- [ ] Eğitim koduna gerçek resume checkpoint desteği ekle ve küçük bir testle doğrula.
- [ ] CLAHE eğitimini kontrollü biçimde tamamla.
- [ ] CLAHE iç ve dış test metriklerini üret.
- [ ] Baseline/CLAHE için conformal kalibrasyon ve kapsama analizini tamamla.
- [ ] Alt grup bootstrap güven aralıkları ve gerektiğinde Holm düzeltmesi uygula.
- [ ] Anonim metrikleri ve yayınlanabilir grafikleri results/published altında yayımla.
- [ ] Streamlit arayüzünü hazırla.
- [ ] Makale taslağını üç ana araştırma sorusu etrafında tamamla.
- [ ] PhysioNet InReDD başvurusunun sonucunu takip et.

## Tekrarlanabilirlik notu

Yayınlanan kod/config dosyaları sabit seed, veri bölmesi kuralları ve ön işleme seçeneklerini belgelemelidir. Veri dosyaları ve .pth model ağırlıkları depoya eklenmez.
