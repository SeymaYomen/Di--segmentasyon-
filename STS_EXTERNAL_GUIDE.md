# STS-2D-Tooth İkinci Dış Test Rehberi

STS-Tooth doğrudan dış test olarak kullanılmaz. Veri setinin bir bölümü CDPR'nin
önceki sürümünden türetildiği için önce kesin ve algısal tekrar kontrolü yapılır.
Yalnızca bu kontrolden geçen örnekler ikinci dış test kümesini oluşturur.

## 1. CPU çalışma zamanı kullanın

Bu hazırlık için GPU gerekmez.

## 2. Yalnızca 900 maskeli panoramik örneği indirin

```bash
pip install datasets
python -m src.download_sts_2d --output-root data/raw/sts_2d_labeled
```

İndirilen içerik yaklaşık 944 MB'lık STS-2D kaynağından gelir. 3B CBCT arşivi ve
maskesiz 3.100 görüntü bu proje için indirilmez.

## 3. CDPR ve mevcut dış veriyle sızıntı kontrolü yapın

```bash
python -m src.prepare_sts_external \
  --reference-manifest data/processed/cdpr/manifest.csv \
  --reference-manifest data/processed/opg_external/manifest.csv
```

Komut, kesin piksel tekrarlarını ve yeniden boyutlandırılmış/kontrastı değişmiş
olası tekrarları dışlar. Şüpheli benzerlikler de ihtiyatlı biçimde test dışında
bırakılır.

## 4. Denetim raporunu kontrol edin

Şu dosyada `n_clean_external_test` değeri sıfırdan büyük olmalıdır:

```text
data/processed/sts_external/audit_summary.json
```

Temiz örnek sayısı ve yetişkin/çocuk dağılımı görülmeden makalede ikinci dış test
iddiası yazılmaz.

## 5. Baseline ve CLAHE modellerini değerlendirin

```bash
python -m src.evaluate --config configs/sts_external_baseline.yaml \
  --checkpoint checkpoints/cdpr_baseline_best.pth

python -m src.evaluate --config configs/sts_external_clahe.yaml \
  --checkpoint checkpoints/cdpr_clahe_best.pth
```

STS görüntüleri eğitime veya kalibrasyona eklenmez; yalnızca dış testte kullanılır.
