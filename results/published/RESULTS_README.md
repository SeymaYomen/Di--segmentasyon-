# Yayımlanabilir Sonuçlar

Bu klasör yalnızca kimliksiz toplu metrikleri içerir. Ham görüntüler, hasta
kimlikleri, maskeler ve model ağırlıkları burada bulunmaz.

- `final_metrics.csv`: iç ve dış test sonuçları
- `conformal_comparison.csv`: piksel düzeyi conformal sonuçları
- `age_subgroup_metrics.csv`: yaş alt grup Dice ve bootstrap güven aralıkları
- `model_performance.svg`: iç/dış Dice ve IoU karşılaştırması
- `conformal_comparison.svg`: kapsama ve belirsizlik oranları
- `age_subgroup_dice.svg`: alt grup Dice ve %95 güven aralıkları

Yuvarlanmış değerler README'de gösterilir; analizlerde CSV dosyalarındaki daha
yüksek hassasiyetli değerler kullanılmalıdır.

Grafikleri yeniden üretmek için:

```bash
python -m src.generate_publication_figures
```
