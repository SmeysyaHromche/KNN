import math
import torch
import torch.nn.functional as F


class OcrCollateFn:
    """
    Collate function for OCR batches with variable-width images.

    Behavior:
    - resizes every image to a fixed height
    - pads width to the maximum width in the batch
    - returns:
        images: Tensor [B, C, target_height, max_width]
        labels: list[str]
        widths: Tensor [B]   # widths after resize, before padding

    Args:
        target_height (int): Final height for all images.
        pad_value (float): Value used for padding. 0.0 is black, 1.0 is white.
    """
    def __init__(self, target_height: int, pad_value: float):
        self.target_height = target_height
        self.pad_value = pad_value

    def _resize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Resize image to target height.
        Args:
            img: Tensor [C, H, W]
        Returns:
            Tensor [C, target_height, new_width]
        """
        if img.ndim != 3:
            raise ValueError(f"Expected image tensor with shape [C, H, W], got {img.shape}")

        _, h, w = img.shape
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image shape: {img.shape}")

        scale = self.target_height / h
        new_width = max(1, int(round(w * scale)))

        img = img.unsqueeze(0)
        img = F.interpolate(img, size=(self.target_height, new_width),mode="bilinear", align_corners=False)
        
        return img.squeeze(0)

    def __call__(self, batch)-> tuple[torch.Tensor, list[str], torch.Tensor]:
        """
        Args:
            batch: list of tuples [(img_tensor, label), ...]
        Returns:
            images: Tensor [B, C, H, W_max]
            labels: list[str]
            widths: Tensor [B]
        """
        if batch is None or len(batch) == 0:
            raise ValueError("Batch is empty")

        resized_images = []
        labels = []
        widths = []

        for img, label in batch:
            resized_img = self._resize(img)
            resized_images.append(resized_img)
            labels.append(label)
            widths.append(resized_img.shape[-1])

        max_width = max(widths)
        batch_size = len(resized_images)
        channels = resized_images[0].shape[0]

        batch_images = torch.full(
            (batch_size, channels, self.target_height, max_width),
            fill_value=self.pad_value,
            dtype=resized_images[0].dtype,
        )

        for i, img in enumerate(resized_images):
            _, _, w = img.shape
            batch_images[i, :, :, :w] = img

        widths = torch.tensor(widths, dtype=torch.long)

        return batch_images, labels, widths