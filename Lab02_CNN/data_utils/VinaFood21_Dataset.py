import os
import cv2 
import torch
from torch.utils.data import Dataset

class VinaFood21(Dataset):
    def __init__(self, path: str, image_size: tuple[int] = (224, 224)):
        super().__init__()
        self.label2idx = {}
        self.idx2label = {}
        self._data = []
        self.image_size = image_size

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        label_idx = 0
        for folder in os.listdir(path): 
            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_idx
                self.idx2label[label_idx] = label

                label_idx += 1
            
            folder_path = os.path.join(path, folder)
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)

                self._data.append({
                    "image_path": image_path,
                    "label": self.label2idx[label]
                })

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index: int) -> dict:
        image_path = self._data[index]["image_path"]
        label = self._data[index]["label"]

        image = cv2.imread(image_path)
        image = cv2.resize(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = image / 255.0
        image = (image - self.mean) / self.std

        return {
            "image": image,
            "label": label
        }
    
def collate_fn(samples: list[dict]) -> dict:
    images = [sample["image"] for sample in samples]
    labels = [sample["label"] for sample in samples]

    images = torch.stack(images, dim=0) # (bs, 3, h, w)
    labels = torch.tensor(labels, dtype=torch.long) # (bs, )

    return {
        "image": images, 
        "label": labels
    }