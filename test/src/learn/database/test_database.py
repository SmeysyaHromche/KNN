from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from learn.database import OcrDataset, OcrCollateFn


def save_batch_images_with_padding(
    dataset,
    collate_fn,
    output_dir: str = "saved_batches_padded",
    batch_size: int = 8,
    num_workers: int = 0,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    try:
        image_index = 0

        for batch_idx, batch in enumerate(loader):
            batch_images, labels, widths = batch
            # shape: [B, C, H, MAX_WIDTH]

            for i in range(batch_images.size(0)):
                img = batch_images[i]
                label = labels[i]
                width = int(widths[i])

                # keep padding
                img = img.detach().cpu()
                img = torch.clamp(img, 0, 1)

                pil_img = to_pil_image(img)

                save_path = (
                    output_path
                    / f"batch_{batch_idx:04d}_img_{i:04d}_{image_index:06d}_w{width}.jpg"
                )

                pil_img.save(save_path, format="JPEG")

                print(
                    f"saved (PADDED): {save_path} | label={label} | real_width={width}"
                )

                image_index += 1

            print("First batch processed. Stopping.")
            break

    except Exception as e:
        print(f"Error during batch processing: {e}")

    finally:
        dataset.close_resources()


if __name__ == "__main__":
    dataset = OcrDataset(
        path_to_db="/home/xkukht01/Dev/KNN/.data/.data",
        path_to_meta_db="/home/xkukht01/Dev/KNN/.data/.meta",
    )

    my_collate_fn = OcrCollateFn(32, 1.0)

    save_batch_images_with_padding(
        dataset,
        my_collate_fn,
        output_dir="debug_images_padded",
        batch_size=6,
        num_workers=2,
    )