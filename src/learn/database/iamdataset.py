import numpy as np
import torch 

from datasets import load_dataset
from torch.utils.data import Dataset

class IAMLineDataset(Dataset):
    """
    Adapter for Hugging Face Teklia/IAM-line dataset.
    Returns the same format as OcrDataset:
    (image, label)
    """

    def __init__(self, split: str = "validation"):
        self.dataset = load_dataset("Teklia/IAM-line", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"].convert("RGB")
        image = np.array(image)                     # [H, W, 3]
        image = np.transpose(image, (2, 0, 1))      # [3, H, W]
        image = torch.from_numpy(image).float()
        
        label = sample["text"]

        return image, label