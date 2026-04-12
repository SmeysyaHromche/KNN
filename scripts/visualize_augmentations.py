"""
Script to help visualize how the augmentations change the
image that is passed to ViT, so they can be tweaked easier.

Run the script from the project root directory with this command:
python3 -m scripts.visualize_augmentations

This script will load some image from the dataset, apply
the augmentations and open a new window with the final image.
"""
import cv2 as cv
from torch.utils.data import DataLoader
from src.learn.augmentations import (
    RandomBrightness,
    RandomContrast,
    Compose,
    GaussianBlur,
    GaussianNoise,
    ElasticTransform,
    RandomSkew,
    RandomMorphology,
)
from src.learn.database import OcrCollateFn, OcrDataset


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                     Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# All images will be transformed to have this fixed height
HEIGHT = 32

# Paths to the dataset
PATH_TO_DB = ""
PATH_TO_META_DB = ""

# No need to change this parameter
BATCH_SIZE = 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Augmentations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
augmentations = Compose(
    [
        # RandomMorphology(probability=0.3, kernel_size_range=(1, 3)),
        GaussianBlur(probability=0.2, kernel_size_range=(1, 3)),
        GaussianNoise(probability=0.2, mean=0, std=3),
        RandomSkew(probability=0.3, skew_range=(-0.5, 0.5)),
        ElasticTransform(probability=0.1, alpha=2, sigma=1),
        RandomBrightness(probability=0.3, brightness=30),
        RandomContrast(probability=0.3, contrast=10),
    ]
)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dataset = OcrDataset(
    path_to_db=PATH_TO_DB,
    path_to_meta_db=PATH_TO_META_DB,
    transform=augmentations,
)

collate_fn = OcrCollateFn(target_height=HEIGHT, pad_value=1.0)
train_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

images, labels, _ = next(iter(train_loader))

# Take first image from batch (C, H, W)
img = images[0]

# Change dimensions to (H, W, C) and convert to numpy
img = img.detach().permute(1, 2, 0).numpy()

# Convert RGB -> BGR for OpenCV
img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

# Show the image
cv.imshow("Augmented", img)
cv.waitKey(0)
cv.destroyAllWindows()
