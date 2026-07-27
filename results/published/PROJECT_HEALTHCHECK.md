# Proje Sağlık Kontrolü

**Durum: BAŞARILI**

27 Temmuz 2026 tarihinde Google Colab ve Google Drive üzerindeki asıl deney
artefaktlarıyla doğrulandı.

- 6/6 deney için görüntü bazlı metrikler ile toplu `metrics.json` ortalamaları
  eşleşti.
- Baseline ve CLAHE checkpoint dosyaları bulundu ve gerçek çıkarımda
  kullanılabildi.
- 9/9 eşleştirilmiş bootstrap/Holm karşılaştırması doğrulandı.
- Conformal ve yaş alt grup sonuçları bulundu.
- Mendeley OPG dış testinden Baseline için 8 ve CLAHE için 8 nitel tahmin
  görseli üretildi.
- Streamlit servisi başlatıldı ve sağlık uç noktası `ok` döndürdü.
- Artefaktların SHA-256 kimlikleri Drive'daki
  `published/project_healthcheck.json` dosyasına kaydedildi.

## Doğrulanmış deneyler

- Internal CDPR | Baseline: n=360, Dice=0.943972709
- Internal CDPR | CLAHE: n=360, Dice=0.939159048
- External OPG | Baseline: n=329, Dice=0.892417697
- External OPG | CLAHE: n=329, Dice=0.899344635
- Clean STS exploratory | Baseline: n=52, Dice=0.899960802
- Clean STS exploratory | CLAHE: n=52, Dice=0.899531486

Ham görüntüler, hasta kimlikleri, görüntü bazlı tablolar ve checkpointler
GitHub'a yüklenmez. Kimliksiz toplu sonuçlar ve doğrulama özeti yayımlanır.
