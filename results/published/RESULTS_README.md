# Yayımlanabilir Sonuçlar

Bu klasör yalnızca kimliksiz toplu metrikleri içerir. Ham görüntüler, hasta
kimlikleri, maskeler ve model ağırlıkları burada bulunmaz.

- `final_metrics.csv`: iç ve dış test sonuçları
- `conformal_comparison.csv`: piksel düzeyi conformal sonuçları
- `age_subgroup_metrics.csv`: yaş alt grup Dice ve bootstrap güven aralıkları
- `model_performance.svg`: iç/dış Dice ve IoU karşılaştırması
- `conformal_comparison.svg`: kapsama ve belirsizlik oranları
- `age_subgroup_dice.svg`: alt grup Dice ve %95 güven aralıkları
- `paired_bootstrap_holm.csv`: Baseline–CLAHE eşleştirilmiş farkları,
  bootstrap güven aralıkları ve Holm düzeltilmiş p-değerleri
- `PROJECT_HEALTHCHECK.md`: Colab/Drive uçtan uca doğrulama özeti
- `PROVENANCE.md`: sonuçların config, checkpoint ve ham metriklere kadar
  izlenebilir üretim zinciri

`final_metrics.csv`, 27 Temmuz 2026 tarihinde Drive'daki altı ham
`metrics.json` dosyasından tam hassasiyetle doğrulanmıştır. Ayrıntılar
`PROVENANCE.md` ve `PROJECT_HEALTHCHECK.md` içindedir.

Ham sonuçları birleştirip grafikleri yeniden üretmek için:

```bash
python -m src.collect_published_results
python -m src.generate_publication_figures
```
