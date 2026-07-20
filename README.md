# Diş Röntgeni Segmentasyon Projesi

Panoramik ve bitewing diş röntgenlerinde ikili diş segmentasyonu için çalıştırılabilir PyTorch projesi. U-Net++ baseline, hasta-bazlı veri ayrımı, kaynak dengeli batch sampler, Dice+BCE kaybı, değerlendirme ve piksel-bazlı conformal tahmin içerir.

> Bu yazılım araştırma/eğitim amaçlıdır; klinik tanı aracı değildir.

## 1. Kurulum

Python **3.10 veya 3.11** kullanın. Python 3.13 ile PyTorch paketleri her platformda uyumlu olmayabilir.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
```

NVIDIA GPU için `requirements-cpu.txt` yerine önce sisteminize uygun PyTorch komutunu [resmî seçiciden](https://pytorch.org/get-started/locally/) çalıştırın, sonra:

```powershell
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
```

## 2. Önce dummy veriyle uçtan uca deneyin

```bash
python -m src.make_dummy_data --output data/raw/dummy --patients 20
python -m src.prepare_data --manifest data/raw/dummy/manifest.csv --output data/processed/dummy --split-output data/splits/dummy.json
python -m src.train --config configs/dummy.yaml
python -m src.evaluate --config configs/dummy.yaml --checkpoint checkpoints/dummy_best.pth
python -m src.conformal --config configs/dummy.yaml --checkpoint checkpoints/dummy_best.pth
```

Çıktılar `checkpoints/` ve `results/` altında oluşur. Dummy veri yalnızca tesisat testidir; bilimsel sonuç sayılmaz.

## 3. Gerçek veriyi yerleştirme

Ham verileri aşağıdaki dizinlere koyun:

```text
data/raw/odontoai/
data/raw/cdpr/
data/raw/isbi_bitewing/
```

Her veri setini önce ikili PNG maskelere dönüştürün. Bu proje model girişinde şu manifest şemasını kullanır:

```csv
image_path,mask_path,patient_id,source
C:/data/img001.png,C:/data/mask001.png,P001,panoramic
C:/data/img002.png,C:/data/mask002.png,P002,bitewing
```

`source` yalnızca `panoramic` veya `bitewing` olmalıdır. Aynı hastaya ait bütün görüntüler aynı `patient_id` değerini taşımalıdır. Ardından:

```bash
python -m src.prepare_data --manifest data/raw/manifest.csv --output data/processed/main --split-output data/splits/main.json
```

Hazırlama komutu panoramik görüntüleri ve maskeleri aynı orta çizgiden ikiye böler, yeniden boyutlandırır ve hasta bazında `%60/%10/%15/%15` train/calibration/validation/test ayrımı yapar. Bitewing örnekleri bölünmeden yeniden boyutlandırılır.

## 4. Eğitim ve değerlendirme

`configs/default.yaml` içindeki yolları ve ayarları düzenleyin:

```bash
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pth
python -m src.conformal --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

TransUNet seçeneği için bu proje `segmentation-models-pytorch` içindeki transformer encoder'lı U-Net yaklaşımını kullanır (`model.name: transunet`). Bu, Beckschen/TransUNet kodunun birebir kopyası değildir; bakım ve giriş boyutu uyumluluğu daha kolay bir transformer tabanlı karşılaştırmadır. Tezde model adını ve implementasyonu açıkça belirtin.

## 5. Üretilen dosyalar

- `results/history.csv`, `results/loss_curve.png`: eğitim geçmişi
- `results/metrics.json`, `results/per_image_metrics.csv`: test metrikleri
- `results/conformal.json`: kalibrasyon eşiği ve kapsama bilgisi
- `results/conformal_examples/`: belirsiz bölgelerin görselleri

## 6. Veri erişimi ve etik

- OdontoAI için veri sahibinin lisans ve erişim prosedürünü izleyin.
- “CDPR” kısaltmasının hangi veri setini ifade ettiğini danışmanınızla kesinleştirin.
- ISBI challenge verisinin resmî lisans/erişim koşullarını doğrulayın.
- Hasta verilerini repoya commit etmeyin; kimliksizleştirme ve kurum etik kurul kurallarına uyun.
- Bölmeyi görüntü bazında değil hasta bazında yapın; aksi halde veri sızıntısı olur.

## Kontrol listesi

- [ ] Python 3.10/3.11 ortamı kuruldu
- [ ] Dummy akış başarıyla tamamlandı
- [ ] Veri izinleri ve lisanslar doğrulandı
- [ ] Maskeler görsel olarak kontrol edildi
- [ ] Hasta-bazlı split üretildi
- [ ] U-Net++ baseline eğitildi
- [ ] Transformer tabanlı model eğitildi
- [ ] Genel ve veri-kaynağı bazlı metrikler raporlandı
- [ ] Conformal kalibrasyon yalnızca calibration setinde yapıldı
- [ ] Sonuçlar farklı seed'lerle tekrarlandı

