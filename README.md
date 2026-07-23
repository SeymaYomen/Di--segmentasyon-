# Diş Röntgeni Segmentasyon Projesi

Panoramik diş röntgenlerinde ikili diş segmentasyonu için çalıştırılabilir PyTorch projesi. U-Net++ baseline, hasta-bazlı veri ayrımı, yinelenen örnek denetimi, CLAHE karşılaştırması, iç/dış test değerlendirmesi, Dice+BCE kaybı ve piksel-bazlı conformal tahmin içerir.

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

## 3. Gerçek veri kaynakları ve yerleşim

Bu çalışmanın güncel deney düzeninde iki panoramik diş röntgeni kaynağı kullanılır:

- **CDPR veri paketi:** Model geliştirme verisidir. Eğitim, kalibrasyon, doğrulama ve iç test kümeleri bu paketten üretilir. İndirilen paketteki yetişkin ve çocuk panoramik alt kümeleri birlikte denetlenir.
- **External OPG (Mendeley Data):** 329 görüntü-maske çiftinden oluşan bağımsız dış test kümesidir. Model bu veriyle eğitilmez; yalnızca farklı bir kaynaktaki genelleme performansını ölçmek için kullanılır.
- **OdontoAI:** Erişim platformunun sonlandırılması nedeniyle zorunlu veri kaynağı değildir ve mevcut deneylere dahil edilmemiştir.
- **ISBI bitewing:** Güncel deney kapsamına dahil değildir.

Kaynaklar:

- [Children's Dental Panoramic Radiographs Dataset makalesi](https://www.nature.com/articles/s41597-023-02237-5)
- [CDPR veri indirme sayfası (Figshare)](https://springernature.figshare.com/articles/dataset/Children_s_Dental_Panoramic_Radiographs_Dataset/21621705)
- [External OPG veri seti (Mendeley Data)](https://data.mendeley.com/datasets/jrz4nj82zv/1)

Ham verileri aşağıdaki dizinlerde tutun:

```text
data/raw/cdpr_bundle/
data/raw/opg_external/
```

CDPR verisini denetlemek, kesin yinelenenleri/boş maskeleri çıkarmak ve bölmeleri üretmek için:

```bash
python -m src.audit_cdpr --raw-root data/raw/cdpr_bundle --output-dir data/processed/cdpr_audit
python -m src.build_cdpr_splits --audit-manifest data/processed/cdpr_audit/manifest_audited.csv --output-manifest data/processed/cdpr/manifest.csv --output-splits data/splits/cdpr.json --seed 42
```

Modelin kullandığı asgari manifest şeması şöyledir:

```csv
image_path,mask_path,patient_id,source
C:/data/img001.png,C:/data/mask001.png,P001,panoramic
```

Aynı hastaya ait bütün örnekler aynı `patient_id` değerini taşımalı ve tek bir bölmede kalmalıdır. Mevcut iş akışı eğitim/kalibrasyon/doğrulama/test ayrımını hasta bazında üretir ve görüntü karmasıyla veri sızıntısını ayrıca kontrol eder. External OPG kümesi bu bölmelere karıştırılmaz.

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

- CDPR ve External OPG verilerinin kaynak sayfalarındaki lisans, atıf ve kullanım koşullarını izleyin.
- OdontoAI mevcut deney düzeninde zorunlu değildir ve erişilemeyen veri sonuçlara dahil edilmez.
- Ham röntgenleri, maskeleri ve model ağırlıklarını repoya commit etmeyin.
- Hasta verilerini kimliksizleştirin ve kurum/etik kurul kurallarına uyun.
- Bölmeleri hasta bazında üretin; kesin yinelenen görüntüleri bölme işleminden önce çıkarın.
- External OPG kümesini yalnızca dış test için kullanın; eğitim veya hiperparametre seçimine dahil etmeyin.

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
