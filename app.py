from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.models import build_model


IMAGE_SIZE = (512, 512)
METHODS = {
    "Baseline": {"preprocessing": "baseline"},
    "CLAHE": {"preprocessing": "clahe"},
}


def preprocess(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    resized = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if mode == "clahe":
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(lightness)
        resized = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2RGB,
        )
    return resized


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_bytes: bytes, device_name: str):
    device = torch.device(device_name)
    model = build_model(
        {
            "name": "unetplusplus",
            "encoder": "resnet34",
            "encoder_weights": None,
        }
    )
    payload = torch.load(
        BytesIO(checkpoint_bytes), map_location=device, weights_only=True
    )
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def predict(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = (
        torch.from_numpy(image_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
        .to(device)
    )
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return probability, (probability >= threshold).astype(np.uint8)


def overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = image_rgb.copy()
    green = np.zeros_like(color)
    green[..., 1] = 255
    selected = mask.astype(bool)
    color[selected] = (
        0.55 * color[selected] + 0.45 * green[selected]
    ).astype(np.uint8)
    return color


st.set_page_config(
    page_title="Diş Segmentasyonu",
    page_icon="🦷",
    layout="wide",
)
st.title("Diş Panoramik Röntgen Segmentasyonu")
st.caption("Araştırma demosu — klinik tanı amacıyla kullanılamaz.")

with st.sidebar:
    method = st.radio(
        "Deney yöntemi",
        options=list(METHODS),
        help=(
            "İç testte Baseline; bağımsız OPG dış testinde CLAHE daha yüksek "
            "Dice üretmiştir. Evrensel bir final model seçilmemiştir."
        ),
    )
    checkpoint = st.file_uploader(
        f"{method} checkpoint (.pth)",
        type=["pth"],
    )
    threshold = st.slider("Maske eşiği", 0.05, 0.95, 0.50, 0.05)
    st.info(
        f"Seçili yöntem: {method}. Ön işleme otomatik olarak "
        f"`{METHODS[method]['preprocessing']}` uygulanır."
    )

image_file = st.file_uploader(
    "Panoramik röntgen yükleyin",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if image_file and checkpoint:
    source = np.asarray(Image.open(image_file).convert("RGB"))
    processed = preprocess(source, METHODS[method]["preprocessing"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with st.spinner(f"Model yükleniyor ve tahmin yapılıyor ({device})..."):
            model = load_model(checkpoint.getvalue(), str(device))
            probability, mask = predict(model, processed, device, threshold)
    except Exception as exc:
        st.error(f"Checkpoint yüklenemedi: {exc}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.image(processed, caption="Model girdisi", use_container_width=True)
    col2.image(
        mask * 255,
        caption="İkili diş maskesi",
        clamp=True,
        use_container_width=True,
    )
    col3.image(
        overlay(processed, mask),
        caption="Bindirme",
        use_container_width=True,
    )
    st.metric(
        "Diş olarak sınıflanan piksel oranı",
        f"%{100 * mask.mean():.2f}",
    )
    st.image(
        probability,
        caption="Diş olasılık haritası",
        clamp=True,
        use_container_width=True,
    )
elif image_file or checkpoint:
    st.warning("Tahmin için hem röntgeni hem checkpoint dosyasını seçin.")
else:
    st.write(
        "Başlamak için deney yöntemini seçin, ona ait model ağırlığını ve bir "
        "röntgen görüntüsünü yükleyin."
    )
