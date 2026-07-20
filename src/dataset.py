from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DentalSegDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def source_labels(self) -> list[str]:
        return self.frame.source.tolist()

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Örnek okunamadı: {row.image_path}, {row.mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.float32)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        mask = np.ascontiguousarray(mask[None, ...], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask), str(row.source), str(row.patient_id)


def subset_from_split(manifest_path: str, split: dict, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str})
    allowed = set(map(str, split[split_name]))
    return frame[frame.patient_id.astype(str).isin(allowed)].reset_index(drop=True)

