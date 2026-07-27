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

Toplu yayın tablosu elle yazılmaz. Altı deneyin `metrics.json` dosyaları hazır
olduğunda şu komutla üretilir:

```bash
python -m src.collect_published_results
python -m src.generate_publication_figures
```

`collect_published_results` herhangi bir ham değerlendirme dosyası eksikse hata
verir ve mevcut `final_metrics.csv` dosyasının üzerine yazmaz.

## Deney–yapılandırma eşleşmesi

| Veri | Yöntem | Config | Beklenen ham sonuç |
|---|---|---|---|
| İç CDPR | Baseline | `configs/cdpr_baseline.yaml` | `results/cdpr_baseline/metrics.json` |
| İç CDPR | CLAHE | `configs/cdpr_clahe.yaml` | `results/cdpr_clahe/metrics.json` |
| Dış OPG | Baseline | `configs/opg_external_baseline.yaml` | `results/opg_external_baseline/metrics.json` |
| Dış OPG | CLAHE | `configs/opg_external_clahe.yaml` | `results/opg_external_clahe/metrics.json` |
| Temiz STS | Baseline | `configs/sts_external_baseline.yaml` | `results/sts_external_baseline/metrics.json` |
| Temiz STS | CLAHE | `configs/sts_external_clahe.yaml` | `results/sts_external_clahe/metrics.json` |

## Mevcut tablonun durumu

`final_metrics.csv`, 27 Temmuz 2026 tarihinde Google Drive'daki altı deney
klasörünün `metrics.json` ve `per_image_metrics.csv` dosyaları karşılaştırılarak
tam hassasiyetle doğrulandı. Her deneyde görüntü sayısı ve üç metrik ortalaması
eşleşti. Checkpointler ve sonuç dosyalarının SHA-256 kimlikleri Drive'daki
`published/project_healthcheck.json` dosyasına kaydedildi.

Model checkpointleri, hasta kimlikleri ve görüntü bazlı dosyalar gizlilik ve
dosya boyutu nedeniyle GitHub'a yüklenmez; yalnızca kimliksiz toplu çıktı ve
doğrulama özeti yayımlanır.
