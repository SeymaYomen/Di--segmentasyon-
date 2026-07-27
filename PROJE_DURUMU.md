# Proje Durumu

Son güncelleme: 27 Temmuz 2026

## Tamamlanan işler

- CDPR veri denetimi, 2.398 temiz görüntü/maske çifti ve hasta bazlı ayırma
- U-Net++/ResNet34 baseline eğitimi
- CLAHE karşılaştırma eğitimi; en iyi doğrulama kaybı 16. epoch
- İç CDPR testi (n=360)
- Mendeley OPG bağımsız dış testi (n=329)
- STS sızıntı taraması ve temiz keşifsel dış test (n=52)
- Dice, IoU ve piksel doğruluğu hesapları
- Görüntü bazlı değerlendirme, eşleştirilmiş bootstrap, %95 güven aralıkları
  ve Holm düzeltmesi
- Baseline ve CLAHE conformal kalibrasyonu
- Yaş alt grup analizi
- Sonuç tablolarının kimliksiz biçimde hazırlanması
- Sonuç grafiklerinin yeniden üretilebilir SVG biçiminde hazırlanması
- Streamlit araştırma demosunun hazırlanması
- Kod, yapılandırma, makale taslağı ve kimliksiz sonuçların GitHub `main` dalına birleştirilmesi

## Bilimsel sonuç

Baseline iç testte daha iyi, CLAHE ise bağımsız Mendeley OPG dış testinde daha
iyi performans verdi. Bu bulgu CLAHE'nin her veri dağılımında genel bir üstünlük
sağlamadığını, fakat belirli dış-merkez görüntü farklılıklarında genelleme
performansını artırabildiğini gösteriyor. Temiz STS alt kümesinde yöntemler
arasında anlamlı fark gözlenmedi.

## Kalan işler

- [x] Drive'daki altı deneyin `metrics.json` ve görüntü bazlı metrikleriyle
  yayımlanan toplu metrikleri doğrulamak
- [x] Baseline–CLAHE eşleştirilmiş bootstrap testini ve Holm düzeltmesini
  tamamlamak
- [x] Mendeley dış testinden her yöntem için sekizer nitel tahmin görseli
  üretmek
- [x] Streamlit araştırma demosunu hazırlamak
- [x] Streamlit servisini gerçek checkpoint çıkarımlarıyla doğrulamak
- [x] Makale taslağı iskeletini oluşturmak
- [ ] Danışmanla başlık, yazar sırası ve hedef dergiyi kesinleştirmek
- [ ] Makale metnini kaynaklarla genişletmek
- [x] Son repo değişikliklerini GitHub'a push etmek

## Güvenli saklama

Checkpoint'ler ve ayrıntılı sonuçlar Google Drive'daki
`dis_segmentasyon_sonuclar/` klasöründedir. Repo yalnızca kodu, yapılandırmaları
ve kimliksiz toplu sonuçları içerir. Ham röntgenler, maskeler ve `.pth` dosyaları
GitHub'a eklenmez.
