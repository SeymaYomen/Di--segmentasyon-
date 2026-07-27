# OPG-DentalSeg dış test rehberi

Bu veri seti **eğitime karıştırılmaz**. CDPR üzerinde eğitilen modelin farklı bir merkezden gelen
panoramik görüntülerde ne kadar iyi genellediğini ölçmek için bağımsız dış test seti olarak kullanılır.

## 1. ZIP dosyasını yerleştirin

İndirilen `Panoramic_Dental_Xray_Segmentation_Dataset.zip` dosyasını açıp içeriğini şu klasöre koyun:

```text
data/raw/opg_external/
```

Beklenen alt klasörler `images/` ve `masks/` şeklindedir. Ham görüntüler ve maskeler GitHub'a
yüklenmez; `.gitignore` tarafından dışarıda tutulur.

## 2. Denetim ve dış test manifesti

```bash
python -m src.prepare_opg_external --raw-root data/raw/opg_external
```

Bu komut:

- görüntü ve maske adlarını eşleştirir,
- boyut uyuşmazlıklarını ve boş maskeleri dışlar,
- veri seti içindeki kesin görüntü tekrarlarını dışlar,
- mevcutsa `data/processed/cdpr/manifest.csv` ile kesin tekrar kontrolü yapar,
- 329 örneği yalnızca `test` bölümüne atar.

Üretilen temel dosyalar:

```text
data/processed/opg_external/manifest.csv
data/processed/opg_external/audit_summary.json
data/splits/opg_external.json
```

## 3. Mevcut CDPR modelini dış veride değerlendirme

```bash
python -m src.evaluate \
  --config configs/opg_external_baseline.yaml \
  --checkpoint checkpoints/cdpr_baseline_best.pth
```

Colab/Drive kontrol noktası kullanılıyorsa `--checkpoint` değerine Drive'daki tam model yolu verilir.

Çıktılar:

```text
results/opg_external_baseline/metrics.json
results/opg_external_baseline/per_image_metrics.csv
```

`metrics.json` dosyası ortalamalara ek olarak %95 bootstrap güven aralıklarını da içerir.

## 4. Örnek tahmin görselleri

```bash
python -m src.visualize_predictions \
  --config configs/opg_external_baseline.yaml \
  --checkpoint checkpoints/cdpr_baseline_best.pth \
  --count 8
```

Yeşil doğru tahmini, kırmızı yanlış pozitif alanı, mavi ise kaçırılan diş alanını gösterir.

## 5. İç ve dış test karşılaştırması

```bash
python -m src.compare_results \
  --internal results/cdpr_baseline/per_image_metrics.csv \
  --external results/opg_external_baseline/per_image_metrics.csv
```

Bu komut iki test seti arasındaki Dice, IoU ve piksel doğruluğu farkını %95 bootstrap güven
aralığıyla raporlar.

## 6. Bilimsel raporlama

CDPR test sonucu **iç test**, OPG-DentalSeg sonucu **dış test** olarak ayrı tablolar halinde verilir.
İki veri setinin örnekleri birleştirilerek yeniden rastgele bölünmez. Böylece dış merkez genellemesi
ölçülür ve veri sızıntısı önlenir.

Veri seti CC BY 4.0 lisanslıdır. Makalede veri seti DOI'si ve özgün yazarları kaynak gösterilmelidir.
