# Colab'da OPG-DentalSeg dış testi — adım adım

Bu işlem yeni modeli eğitmez. Drive'da kayıtlı CDPR U-Net++ modelini 329 yeni panoramik
röntgende değerlendirir. Mümkünse T4 GPU kullanın.

## Önce Google Drive'a yüklenecek iki ZIP

1. `dis_segmentasyon_external_test_kodu.zip`
2. `Panoramic_Dental_Xray_Segmentation_Dataset.zip`

Mevcut model şu konumda kalmalıdır:

```text
MyDrive/dis_segmentasyon_sonuclar/checkpoints/cdpr_baseline_best.pth
```

## Hücre 1 — Drive bağlantısı ve dosya kontrolü

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

CODE_ZIP = Path("/content/drive/MyDrive/dis_segmentasyon_external_test_kodu.zip")
DATA_ZIP = Path("/content/drive/MyDrive/Panoramic_Dental_Xray_Segmentation_Dataset.zip")
MODEL = Path("/content/drive/MyDrive/dis_segmentasyon_sonuclar/checkpoints/cdpr_baseline_best.pth")

for path in [CODE_ZIP, DATA_ZIP, MODEL]:
    print(path.name, "->", "VAR" if path.exists() else "YOK")

assert CODE_ZIP.exists(), "Kod ZIP dosyası Drive'da bulunamadı."
assert DATA_ZIP.exists(), "Veri ZIP dosyası Drive'da bulunamadı."
assert MODEL.exists(), "Eğitilmiş model Drive'da bulunamadı."
```

Üç satırda da `VAR` görülmeden devam etmeyin.

## Hücre 2 — Kod ve veriyi geçici Colab alanına çıkarma

```python
!mkdir -p "/content/dis_external_test"
!unzip -oq "{CODE_ZIP}" -d "/content/dis_external_test"
!mkdir -p "/content/dis_external_test/data/raw/opg_external"
!unzip -oq "{DATA_ZIP}" -d "/content/dis_external_test/data/raw/opg_external"

%cd /content/dis_external_test
!find . -maxdepth 2 -type f | head -30
```

## Hücre 3 — Paket kurulumu

```python
!pip install -q -r requirements.txt
```

Kurulumdan sonra hata olmadan hücre tamamlanmalıdır.

## Hücre 4 — GPU kontrolü

```python
import torch
print("GPU aktif mi:", torch.cuda.is_available())
print("Aygıt:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

GPU yoksa veri hazırlığı yapılabilir; değerlendirme çok daha uzun sürer.

## Hücre 5 — 329 görüntünün kalite kontrolü

```python
!python -m src.prepare_opg_external \
  --raw-root data/raw/opg_external \
  --reference-manifest data/processed/cdpr/manifest.csv
```

Beklenen temel sonuçlar:

```text
n_images_found: 329
n_masks_found: 329
n_usable_pairs: 329
n_excluded: 0
```

Bu minimal pakette CDPR manifesti bulunmadığı için `reference_overlap_check: not_available` yazabilir.
Kesin tekrar kontrolü yerel hazırlık sırasında ayrıca yapılmış ve çakışma bulunmamıştır.

## Hücre 6 — Dış test değerlendirmesi

```python
!python -m src.evaluate \
  --config configs/opg_external_baseline.yaml \
  --checkpoint "/content/drive/MyDrive/dis_segmentasyon_sonuclar/checkpoints/cdpr_baseline_best.pth"
```

Tamamlandığında Dice, IoU, piksel doğruluğu ve %95 güven aralıkları ekranda görünür.

## Hücre 7 — Örnek tahmin görselleri

```python
!python -m src.visualize_predictions \
  --config configs/opg_external_baseline.yaml \
  --checkpoint "/content/drive/MyDrive/dis_segmentasyon_sonuclar/checkpoints/cdpr_baseline_best.pth" \
  --count 8
```

Renkler:

- yeşil: doğru diş tahmini,
- kırmızı: yanlış pozitif,
- mavi: modelin kaçırdığı diş alanı.

## Hücre 8 — Sonuçları kalıcı olarak Drive'a kopyalama

```python
!mkdir -p "/content/drive/MyDrive/dis_segmentasyon_sonuclar/external_opg"
!cp -r results/opg_external_baseline/. \
  "/content/drive/MyDrive/dis_segmentasyon_sonuclar/external_opg/"
!cp data/processed/opg_external/audit_summary.json \
  "/content/drive/MyDrive/dis_segmentasyon_sonuclar/external_opg/audit_summary.json"

!find "/content/drive/MyDrive/dis_segmentasyon_sonuclar/external_opg" -maxdepth 2 -type f
```

Son hücrede `metrics.json`, `per_image_metrics.csv` ve örnek PNG dosyaları listelenirse işlem
tamamlanmıştır. Bundan sonra Colab kapatılsa bile sonuçlar Drive'da kalır.
