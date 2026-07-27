# Sonuçların İzlenebilirliği

## Üretim zinciri

Her deney aşağıdaki komut biçimiyle değerlendirilir:

```bash
python -m src.evaluate --config <config.yaml> --checkpoint <model.pth>
```

Bu işlem deney klasöründe üç dosya üretir:

- `per_image_metrics.csv`: görüntü bazında Dice, IoU ve piksel doğruluğu
- `metrics.json`: ortalamalar, %95 bootstrap güven aralıkları ve örnek sayısı
- `evaluation_provenance.json`: config, checkpoint, manifest ve split SHA-256
  değerleri; ön işleme, eşik, seed, çalışma ortamı ve kullanılan komut

Toplu yayın tablosu elle yazılmaz. Altı deneyin ham `metrics.json` dosyaları
`src.collect_published_results` ile birleştirilir; `final_metrics.csv` bu
dosyalardan tam hassasiyetle üretilmiştir.

## Doğrulanan deneyler

- İç CDPR — Baseline, n=360
- İç CDPR — CLAHE, n=360
- Bağımsız OPG — Baseline, n=329
- Bağımsız OPG — CLAHE, n=329
- Temiz STS keşifsel — Baseline, n=52
- Temiz STS keşifsel — CLAHE, n=52

## Tek hücrelik sağlık kontrolü

Colab not defterindeki sağlık kontrolü 27 Temmuz 2026 tarihinde başarıyla
çalıştırılmıştır. Altı deneyde `per_image_metrics.csv` ortalamaları ile
`metrics.json` toplu değerleri `1e-12` toleransta eşleşmiştir. Baseline/CLAHE
checkpointleri, dokuz eşleştirilmiş bootstrap/Holm testi, conformal ve yaş alt
grubu çıktıları bulunmuştur. Dosya SHA-256 değerleri Drive'daki
`published/project_healthcheck.json` dosyasına kaydedilmiştir.

## Eşleştirilmiş istatistik

Baseline ve CLAHE aynı test görüntülerinde değerlendirildiği için görüntü
kimliğine göre birebir birleştirilmiştir. Her metrik için 5.000 tekrarlı
eşleştirilmiş bootstrap uygulanmış, dokuz p değeri Holm yöntemiyle
düzeltilmiştir. Sonuçlar `paired_bootstrap_holm.csv` dosyasındadır.

## Nitel iyi/zor örnek paneli

Panel, dış OPG testindeki iki yöntemin görüntü-bazlı Dice ortalamasına göre en
yüksek iki ve en düşük iki örneği seçer. Kimlikler panelde ve yayımlanan
yardımcı CSV/JSON dosyalarında tutulmaz:

```bash
python -m src.create_qualitative_panel \
  --baseline-config configs/opg_external_baseline.yaml \
  --baseline-checkpoint checkpoints/cdpr_baseline_best.pth \
  --clahe-config configs/opg_external_clahe.yaml \
  --clahe-checkpoint checkpoints/cdpr_clahe_best.pth \
  --baseline-metrics results/opg_external_baseline/per_image_metrics.csv \
  --clahe-metrics results/opg_external_clahe/per_image_metrics.csv \
  --output results/published/qualitative_good_hard_panel.png
```

Renkler: yeşil doğru pozitif, kırmızı yanlış pozitif, turuncu yanlış negatiftir.

## Paylaşım sınırı

Ham görüntüler, maskeler, hasta/dosya kimlikleri, görüntü-bazlı metrikler ve
checkpoint dosyaları kamuya açık klasöre eklenmez. Yalnızca kimliksiz toplu
sonuçlar, figürler, seçim yöntemi ve dosya kimlikleri yayımlanır.
