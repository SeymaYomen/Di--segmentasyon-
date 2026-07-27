# Yayımlanabilir Sonuçlar

Bu klasör yalnızca kimliksiz toplu metrikleri ve makale figürlerini içerir. Ham
görüntüler, hasta/dosya kimlikleri, maskeler ve model ağırlıkları burada
bulunmaz.

- `final_metrics.csv`: iç ve dış test sonuçları
- `paired_bootstrap_holm.csv`: eşleştirilmiş farklar, %95 bootstrap güven
  aralıkları ve Holm-düzeltilmiş p değerleri
- `conformal_comparison.csv`: piksel düzeyi conformal sonuçları
- `age_subgroup_metrics.csv`: yaş alt grup Dice ve güven aralıkları
- `model_performance.svg`: iç/dış performans karşılaştırması
- `conformal_comparison.svg`: kapsama ve belirsizlik oranları
- `age_subgroup_dice.svg`: alt grup Dice ve %95 güven aralıkları
- `qualitative_good_hard_panel.csv/json`: dış OPG'den kimliksiz iki iyi ve iki
  zor örneğin seçim kaydı. Tıbbi görüntü içeren 6,2 MB PNG paneli hasta/dosya
  kimliği içermese de kamu reposuna otomatik eklenmemiş, doğrulanmış Drive
  sonuç klasöründe tutulmuştur.
- `PROJECT_HEALTHCHECK.md`: Colab/Drive uçtan uca doğrulama özeti
- `PROVENANCE.md`: config, checkpoint, ham metrik ve figür üretim zinciri

Altı deneyin görüntü-bazlı ve toplu metrik uyumu
`PROJECT_HEALTHCHECK.md` ile doğrulanmıştır. Nitel panelin seçimi
`PROVENANCE.md` içinde açıklanmıştır.
