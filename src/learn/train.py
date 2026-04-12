import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.learn.augmentations import RandomBrightness, RandomContrast, Compose, GaussianBlur, GaussianNoise, ElasticTransform, RandomSkew, RandomMorphology

from src.common.tokenizer import Tokenizer

from src.learn.config.learnconfig import LearnConfig

from src.learn.database.ocrcollatefn import OcrCollateFn
from src.learn.database.ocrdataset import OcrDataset

from src.model.knn import Knn


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                     Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load JSON config file
config_path = Path("learnconfig.json")
config = LearnConfig.model_validate_json(config_path.read_text())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer
tokenizer = Tokenizer(Path(config.data.path_to_vocabulary_file))

PAD_IDX = tokenizer.encode_special_token("<pad>")
BOS_IDX = tokenizer.encode_special_token("<bos>")
EOS_IDX = tokenizer.encode_special_token("<eos>")

# Timestamp is created to differentiate models from multiple runs
timestamp = time.strftime("%d%m%Y_%H%M%S")
MODEL_RUN_DIR = os.path.join(config.train.output_model_dir, timestamp)
os.makedirs(MODEL_RUN_DIR, exist_ok=True)

if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                   Augmentations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Augmentations are set to not be so aggressive for now
    augmentations = Compose([
        # RandomMorphology(probability=0.3, kernel_size_range=(1, 3)),
        GaussianBlur(probability=0.2, kernel_size_range=(1, 3)),
        GaussianNoise(probability=0.2, mean=0, std=3),
        RandomSkew(probability=0.3, skew_range=(-0.5, 0.5)),
        ElasticTransform(probability=0.1, alpha=2, sigma=1),
        RandomBrightness(probability=0.3, brightness=30),
        RandomContrast(probability=0.3, contrast=10),
    ])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                      Dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dataset = OcrDataset(
        path_to_db=config.data.path_to_db,
        path_to_meta_db=config.data.path_to_trn_meta_db,
        transform=augmentations,
    )

    collate_fn = OcrCollateFn(target_height=config.data.image_target_height, pad_value=1.0)
    train_loader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True, collate_fn=collate_fn)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                       Model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model = Knn(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=PAD_IDX,
        bos_token_id=BOS_IDX,
        eos_token_id=EOS_IDX,
        is_pretrain_swin=True
    ).to(DEVICE)

    # Freeze Swin backbone
    for param in model.visual_tokenizer.parameters():
        param.requires_grad = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                       Loss
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.optimizer_lr)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                     Training
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for epoch in range(config.train.num_of_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(DEVICE)

            # Encode each label into token IDs with BOS and EOS
            target_indices = [
                torch.tensor([BOS_IDX] + tokenizer.encode(label) + [EOS_IDX]) for label in labels
            ]

            # Pad sequences with PAD in batch to the same length
            target_indices = torch.nn.utils.rnn.pad_sequence(
                target_indices, batch_first=True, padding_value=PAD_IDX
            ).to(DEVICE)

            # Teacher forcing
            target_input = target_indices[:, :-1]   # Decoder input excludes last token (EOS)
            target_output = target_indices[:, 1:]   # Decoder target excludes first token (BOS)

            # Forward pass
            logits = model(images, target_input)    # [batch, sequence_len, vocab_size]

            # Flatten for cross-entropy loss
            logits_flat = logits.reshape(-1, logits.size(-1))   # [batch * sequence_len, vocab_size] - cross-entropy expects [samples, num_of_classes]
            target_output_flat = target_output.reshape(-1)      # [batch * sequence_len] - each element is the correct token ID for that position

            # Compute loss, ignoring PAD tokens
            loss = criterion(logits_flat, target_output_flat)
            total_loss += loss.item()

            # Print learning status each 10 batches (the line rewrites itself in the console)
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}]", end="\r", flush=True)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print progress info
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{config.train.num_of_epochs}, Avg Loss: {avg_loss:.4f}")

        # Save per-epoch checkpoint
        if config.train.save_model_per_epoch:
            epoch_path = os.path.join(MODEL_RUN_DIR, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, epoch_path)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #               Save the final model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    final_path = os.path.join(MODEL_RUN_DIR, "final.pt")
    torch.save({
        "epoch": config.train.num_of_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, final_path)

    print(f"Final model saved to {final_path}")
