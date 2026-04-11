import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from learn.augmentations import RandomBrightness, RandomContrast, Compose, GaussianBlur, GaussianNoise, ElasticTransform, RandomSkew, RandomMorphology

from common.tokenizer import Tokenizer

from learn.components.swin_feature_extractor import SwinFeatureExtractor

from learn.database.ocrcollatefn import OcrCollateFn
from learn.database.ocrdataset import OcrDataset


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                     Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IMAGE_TARGET_HEIGHT=32              # Target height of the resized images
BATCH_SIZE = 64                     # Number of images in a single batch
PATH_TO_DB = ""                     # Path to the image DB
PATH_TO_META_DB = ""                # Path to the image meta DB
NUM_OF_EPOCHS = 5                   # Number of epochs to be done in the training loop

SAVE_MODEL_PER_EPOCH = False        # Boolean value indicating whether the trained model should be saved each epoch (checkpoints)
OUTPUT_DIR = "src/learn/output"     # Directory of the output models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer
VOCABULARY_FILE = "src/common/vocabulary.txt"
tokenizer = Tokenizer(Path(VOCABULARY_FILE))

PAD_IDX = tokenizer.encode_special_token("<pad>")
BOS_IDX = tokenizer.encode_special_token("<bos>")
EOS_IDX = tokenizer.encode_special_token("<eos>")

# Timestamp is created to differentiate models from multiple runs
timestamp = time.strftime("%d%m%Y_%H%M%S")
MODEL_RUN_DIR = os.path.join(OUTPUT_DIR, timestamp)
os.makedirs(MODEL_RUN_DIR, exist_ok=True)

if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                   Augmentations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Augmentations are set to not be so aggressive for now
    augmentations = Compose([
        RandomMorphology(probability=0.3, kernel_size_range=(1, 3)),
        GaussianBlur(probability=0.2, kernel_size_range=(5, 9)),
        GaussianNoise(probability=0.2, mean=0, std=1),
        RandomSkew(probability=0.3, skew_range=(-0.5, 0.5)),
        ElasticTransform(probability=0.2),
        RandomBrightness(probability=0.3, brightness=30),
        RandomContrast(probability=0.3, contrast=30),
    ])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                      Dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dataset = OcrDataset(
        path_to_db=PATH_TO_DB,
        path_to_meta_db=PATH_TO_META_DB,
    )

    collate_fn = OcrCollateFn(target_height=IMAGE_TARGET_HEIGHT, pad_value=1.0)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # TODO: apply augmentations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                       Model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    swin_extractor = SwinFeatureExtractor(freeze=True).to(DEVICE)
    # TODO: initialize decoder

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                       Loss
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # TODO: add optimizer for decoder

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                     Training
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for epoch in range(NUM_OF_EPOCHS):
        # Only evaluate - Swin is frozen
        swin_extractor.eval()

        # TODO: train decoder

        for images, labels, _ in train_loader:
            # TODO: implement train loop
            pass

        # TODO: print loss
        print(f"Epoch {epoch}/{NUM_OF_EPOCHS}")

        # Save per-epoch checkpoint
        if SAVE_MODEL_PER_EPOCH:
            epoch_path = os.path.join(MODEL_RUN_DIR, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "swin_state_dict": swin_extractor.state_dict(),
                # TODO: add decoder optimizer state dict
            }, epoch_path)

    # Save the final model
    final_path = os.path.join(MODEL_RUN_DIR, "final.pt")
    torch.save({
        "epoch": NUM_OF_EPOCHS,
        "swin_state_dict": swin_extractor.state_dict(),
        # TODO: add decoder optimizer state dict
    }, final_path)

    print(f"Final model saved to {final_path}")
