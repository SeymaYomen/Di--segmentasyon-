# Proje Durumu

Son güncelleme: 24 Temmuz 2026

## Tamamlanan işler

- CDPR veri denetimi, 2.398 temiz görüntü/maske çifti ve hasta bazlı ayırma
- U-Net++/ResNet34 baseline eğitimi
- CLAHE karşılaştırma eğitimi; en iyi doğrulama kaybı 16. epoch
- İç CDPR testi (n=360)
- Mendeley OPG bağımsız dış testi (n=329)
- STS sızıntı taraması ve temiz keşifsel dış test (n=52)
- Dice, IoU ve piksel doğruluğu hesapları
- Eşleştirilmiş bootstrap, %95 güven aralıkları ve Holm düzeltmesi
- Baseline ve CLAHE conformal kalibrasyonu
- Yaş alt grup analizi
- Sonuç tablolarının kimliksiz biçimde hazırlanması

## Bilimsel sonuç

Baseline iç testte daha iyi, CLAHE ise bağımsız Mendeley OPG dış testinde daha
iyi performans verdi. Bu bulgu CLAHE'nin her veri dağılımında genel bir üstünlük
sağlamadığını, fakat belirli dış-merkez görüntü farklılıklarında genelleme
performansını artırabildiğini gösteriyor. Temiz STS alt kümesinde yöntemler
arasında anlamlı fark gözlenmedi.

## Kalan işler

- [ ] Dış testlerden sekizer nitel tahmin görselini tamamlamak
- [x] Streamlit araştırma demosunu hazırlamak
- [x] Makale taslağı iskeletini oluşturmak
- [ ] Danışmanla başlık, yazar sırası ve hedef dergiyi kesinleştirmek
- [ ] Makale metnini kaynaklarla genişletmek
- [ ] Son repo değişikliklerini GitHub'a push etmek

## Güvenli saklama

Checkpoint'ler ve ayrıntılı sonuçlar Google Drive'daki
`dis_segmentasyon_sonuclar/` klasöründedir. Repo yalnızca kodu, yapılandırmaları
ve kimliksiz toplu sonuçları içerir. Ham röntgenler, maskeler ve `.pth` dosyaları
GitHub'a eklenmez.

