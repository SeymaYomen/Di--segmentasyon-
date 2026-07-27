from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


FILTER_MODES = {"baseline", "clahe", "gamma", "clahe_bilateral"}


def enhance_image(image: np.ndarray, mode: str = "baseline", gamma: float = 1.0) -> np.ndarray:
    """Apply deterministic radiograph enhancement without changing geometry."""
    if mode not in FILTER_MODES:
        raise ValueError(f"Bilinmeyen filtre modu: {mode}; seçenekler: {sorted(FILTER_MODES)}")
    if gamma <= 0:
        raise ValueError("gamma sıfırdan büyük olmalıdır.")
    if mode == "baseline":
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    if mode in {"clahe", "clahe_bilateral"}:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if mode == "gamma":
        table = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)], dtype=np.uint8)
        gray = cv2.LUT(gray, table)
    if mode == "clahe_bilateral":
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def read_pair(image_path: str | Path, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Maske okunamadı: {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Boyut uyuşmazlığı: {image.shape[:2]} != {mask.shape[:2]}")
    return image, (mask > 0).astype(np.uint8) * 255


def split_patch2(image: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Görüntü ve maske aynı boyutta olmalıdır.")
    mid = image.shape[1] // 2
    return [(image[:, :mid], mask[:, :mid], "left"), (image[:, mid:], mask[:, mid:], "right")]


def resize_pair(image: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = size
    image_out = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask_out = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return image_out, (mask_out > 0).astype(np.uint8) * 255


def save_pair(image: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"Yazılamadı: {image_path}")
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Yazılamadı: {mask_path}")
