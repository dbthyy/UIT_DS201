import torch
import idx2numpy
import numpy as np
from torch.utils.data import Dataset

class MinstDataset(Dataset):
    def __init__(self, img_path: str, label_path: str):
        images = idx2numpy.convert_from_file(img_path)
        labels = idx2numpy.convert_from_file(label_path)

        images = images.astype(np.float32) / 255.0 
        images = np.expand_dims(images, axis=1)
        self._data = [
            {"image": images[i], "label": labels[i]} for i in range(len(labels))
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
    images = np.stack([item["image"] for item in items], axis=0)
    labels = np.stack([item["label"] for item in items], axis=0)

    return {
        "image": torch.tensor(images, dtype=torch.float32),
        "label": torch.tensor(labels, dtype=torch.long)
    }