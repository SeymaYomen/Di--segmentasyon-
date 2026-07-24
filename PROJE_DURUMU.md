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
