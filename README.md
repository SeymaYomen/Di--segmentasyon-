# Diş Panoramik Röntgen Segmentasyonu

Panoramik diş röntgenlerinde diş/arka plan segmentasyonu, dış-merkez
genellenebilirlik ve conformal belirsizlik analizi için hazırlanmış PyTorch
projesidir.

> Araştırma ve eğitim amaçlıdır. Klinik tanı aracı değildir.

## Araştırma soruları

1. U-Net++ modeli iç testten bağımsız dış testlere geçtiğinde ne kadar performans kaybediyor?
2. CLAHE ön işlemesi bu genelleme farkını azaltıyor mu?
3. Piksel düzeyindeki conformal kapsama ve belirsizlik oranı nasıl değişiyor?

## Veri düzeni

| Veri | Rol | Kullanılan örnek | Lisans |
|---|---|---:|---|
| [Children's Dental Panoramic Radiographs (CDPR)](https://springernature.figshare.com/articles/dataset/Children_s_Dental_Panoramic_Radiographs_Dataset/21621705) | Eğitim, kalibrasyon, doğrulama ve iç test | 2.398 temiz çift; iç test 360 | CC0 |
| [Mendeley Panoramic Dental X-ray Segmentation](https://data.mendeley.com/datasets/jrz4nj82zv/1) | Birinci bağımsız dış test | 329 | CC BY 4.0 |
| [STS-2D-Tooth](https://huggingface.co/datasets/MedOtter/STS-2D-Tooth) | İkinci, keşifsel dış test | Sızıntı taramasından sonra 52 | CC BY 4.0 |

OdontoAI erişilemediği için deneylerde kullanılmadı. STS verisindeki 900 etiketli
örnek CDPR ve Mendeley verilerine karşı hash/perceptual-hash taramasından
geçirildi; 848 olası örtüşme dışlandı. Kalan 52 örnek küçük olduğu için bu sonuç
yalnızca keşifsel olarak yorumlanır. Dış test verileri eğitime veya model
seçimine dahil edilmedi.

Ham tıbbi görüntüler, maskeler ve `.pth` ağırlıkları bu repoda yayımlanmaz.

## Sonuç özeti

| Test kümesi | Yöntem | Dice | IoU | Piksel doğruluğu |
|---|---|---:|---:|---:|
| İç CDPR (n=360) | Baseline | 0.9440 | 0.8952 | 0.9804 |
| İç CDPR (n=360) | CLAHE | 0.9392 | 0.8866 | 0.9788 |
| Dış OPG (n=329) | Baseline | 0.8924 | 0.8077 | 0.9730 |
| Dış OPG (n=329) | CLAHE | 0.8993 | 0.8184 | 0.9744 |
| Temiz STS (n=52, keşifsel) | Baseline | 0.9000 | 0.8193 | 0.9696 |
| Temiz STS (n=52, keşifsel) | CLAHE | 0.8995 | 0.8186 | 0.9695 |

CLAHE iç CDPR performansını hafif düşürürken dış OPG performansını artırdı.
Temiz STS alt kümesindeki fark ihmal edilebilir düzeydeydi. Eşleştirilmiş
bootstrap ve Holm düzeltmeli sonuçlar `results/published/` altındadır.

Piksel düzeyinde %90 hedefli conformal analizde:

- Baseline: kapsama 0.9008, belirsiz/boş oranı 0.0980
- CLAHE: kapsama 0.9012, belirsiz/boş oranı 0.0973

Bu garanti marjinal piksel kapsamasıdır; görüntü içi uzamsal bağımlılık nedeniyle
bölge veya görüntü düzeyinde garanti olarak yorumlanmamalıdır.

## Kurulum

Python 3.10 veya 3.11 önerilir.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
```

GPU kurulumu için önce [PyTorch resmi seçicisindeki](https://pytorch.org/get-started/locally/)
sisteminize uygun komutu, ardından:

```powershell
pip install -r requirements.txt
```

## Dummy veriyle doğrulama

```bash
python -m src.make_dummy_data --output data/raw/dummy --patients 20
python -m src.prepare_data --manifest data/raw/dummy/manifest.csv --output data/processed/dummy --split-output data/splits/dummy.json
python -m src.train --config configs/dummy.yaml
python -m src.evaluate --config configs/dummy.yaml --checkpoint checkpoints/dummy_best.pth
python -m src.conformal --config configs/dummy.yaml --checkpoint checkpoints/dummy_best.pth
```

Dummy sonuçlar bilimsel bulgu değildir; yalnızca kod akışını sınar.

## Gerçek veri manifesti

Model girişi aşağıdaki temel alanları kullanır:

```csv
image_path,mask_path,patient_id,source
C:/data/img001.png,C:/data/mask001.png,P001,panoramic
```

Aynı hastaya ait bütün görüntüler aynı `patient_id` değerini taşımalıdır.
Ayırma hasta bazında yapılır; görüntü bazında rastgele ayırma veri sızıntısına
yol açar.

## Eğitim ve değerlendirme

```bash
python -m src.train --config configs/cdpr_baseline.yaml
python -m src.evaluate --config configs/cdpr_baseline.yaml --checkpoint checkpoints/cdpr_baseline_best.pth
python -m src.conformal --config configs/cdpr_baseline.yaml --checkpoint checkpoints/cdpr_baseline_best.pth
```

`src.evaluate`, her çalışmada toplu metriklerin yanında görüntü bazlı metrikleri
ve config/checkpoint/veri ayrımı SHA-256 değerlerini içeren
`evaluation_provenance.json` dosyasını üretir. Altı deney tamamlandıktan sonra
yayımlanabilir tablo ve grafikler elle kopyalanmadan üretilir:

```bash
python -m src.collect_published_results
python -m src.generate_publication_figures
```

CLAHE karşılaştırması için `configs/cdpr_clahe.yaml` kullanılır. Ana model
U-Net++/ResNet34'tür. En iyi ağırlık doğrulama kaybına göre seçilir ve erken
durdurma uygulanır.

## Arayüz

Yerel araştırma demosu:

```bash
streamlit run app.py
```

Arayüz bir panoramik görüntü ve yerel checkpoint alarak olasılık maskesi,
ikili maske ve bindirme görüntüsü üretir. Klinik kullanım için doğrulanmamıştır.

## Repo içeriği

- `src/`: veri denetimi, eğitim, değerlendirme ve conformal analiz
- `configs/`: baseline, CLAHE ve dış test yapılandırmaları
- `results/published/`: kimliksiz toplu metrikler ve istatistik tabloları
- `PROJE_DURUMU.md`: tamamlanan ve kalan işler
- `docs/MAKALE_TASLAGI.md`: makale iskeleti

Yayımlanabilir sonuç grafiklerini yeniden üretmek için:

```bash
python -m src.generate_publication_figures
```

## Etik ve yeniden üretilebilirlik

- Lisans ve atıf koşulları her veri kaynağı için ayrıca izlenmelidir.
- Hasta verileri ve model ağırlıkları GitHub'a yüklenmez.
- Dış test kümeleri eğitim, hiperparametre seçimi veya eşik ayarında kullanılmaz.
- Çocuk alt grubu n=30 olduğu için sonuçlar keşifsel ve bootstrap güven aralığıyla raporlanır.
- Cinsiyet bilgisi bulunmadığından cinsiyet alt grup analizi yapılmaz.
