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

DEVICE = config.train.device if torch.cuda.is_available() else "cpu"

# Tokenizer
tokenizer = Tokenizer(Path(config.data.path_to_vocabulary_file))

PAD_IDX = tokenizer.encode_special_token("<pad>")
BOS_IDX = tokenizer.encode_special_token("<bos>")
EOS_IDX = tokenizer.encode_special_token("<eos>")

# Timestamp is created to differentiate models from multiple runs
timestamp = time.strftime("%d%m%Y_%H%M%S")
MODEL_RUN_DIR = os.path.join(config.train.output_model_dir, timestamp)
os.makedirs(MODEL_RUN_DIR, exist_ok=True)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_IDX)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                 Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def unfreeze_swin_stage3(model):
    print("-> Unfreezing Swin stage 3")

    layers = list(model.visual_tokenizer._feature_extractor)
    stage3 = layers[-1]  # always last stage

    for param in stage3.parameters():
        param.requires_grad = True

def unfreeze_norm_layers(model):
    print("-> Unfreezing Swin norms")

    # Unfreeze all LayerNorms inside transformer blocks
    for name, param in model.visual_tokenizer.named_parameters():
        if "norm" in name:
            param.requires_grad = True

    # Unfreeze final output norm
    model.visual_tokenizer._norm.requires_grad_(True)

def build_eos_mask(targets, eos_id, pad_id):
    mask = targets != pad_id

    eos_pos = (targets == eos_id).float().cumsum(dim=1)
    mask = mask & (eos_pos <= 1)

    return mask

def run_epoch(model, loader, optimizer=None, epoch=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch_idx, (images, labels, _) in enumerate(loader):
            images = images.to(DEVICE)

            # Encode each label into token IDs with BOS and EOS
            target_indices = [
                torch.tensor([BOS_IDX] + tokenizer.encode(label) + [EOS_IDX], device=DEVICE)
                for label in labels
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
            loss_raw = loss_fn(logits_flat, target_output_flat)
            loss_mask = build_eos_mask(target_output, EOS_IDX, PAD_IDX).reshape(-1)
            loss = (loss_raw * loss_mask).sum() / loss_mask.sum()

            # Backpropagation only if training (not on validation)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Print learning status each 10 batches (the line rewrites itself in the console)
            if is_training and batch_idx % 10 == 0:
                print(f"[Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}]", end="\r", flush=True)

    return total_loss / len(loader)

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
    train_loader = DataLoader(dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers_train,
        pin_memory=True,
        persistent_workers=True
    )

    # Validation dataset
    val_dataset = OcrDataset(
        path_to_db=config.data.path_to_db,
        path_to_meta_db=config.data.path_to_vld_meta_db,
        transform=None
    )

    val_loader = DataLoader(val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers_validation,
        pin_memory=True,
        persistent_workers=True
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                       Model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model = Knn(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=PAD_IDX,
        bos_token_id=BOS_IDX,
        eos_token_id=EOS_IDX,
        is_pretrain_swin=config.model.is_pretrain_swin
    ).to(DEVICE)

    # Freeze Swin backbone
    for param in model.visual_tokenizer.parameters():
        param.requires_grad = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                       Loss
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam([
        {"params": model.decoder.parameters(), "lr": config.train.optimizer_lr},
        {"params": model.visual_adapter.parameters(), "lr": config.train.optimizer_lr}
        # {"params": model.visual_tokenizer.parameters(), "lr": config.train.swin_optimizer_lr}
    ])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                     Training
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for epoch in range(1, config.train.num_of_epochs + 1):
        if epoch == config.train.unfreeze_swin_norms_epoch:
            unfreeze_norm_layers(model)

            # optimizer.add_param_group(
            #     {"params": model.visual_tokenizer.parameters(), "lr": config.train.swin_optimizer_lr}
            # )

            trainable = [
                p for p in model.visual_tokenizer.parameters()
                if p.requires_grad
            ]

            optimizer.add_param_group({
                "params": trainable,
                "lr": config.train.swin_optimizer_lr
            })

        if epoch == config.train.unfreeze_swin_epoch:
            unfreeze_swin_stage3(model)        

        # Print progress info
        train_loss = run_epoch(model, train_loader, optimizer=optimizer, epoch=epoch)
        val_loss = run_epoch(model, val_loader)

        print(f"Epoch {epoch}/{config.train.num_of_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save per-epoch checkpoint
        if config.train.save_model_per_epoch:
            epoch_path = os.path.join(MODEL_RUN_DIR, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, epoch_path)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #               Save the final model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    final_path = os.path.join(MODEL_RUN_DIR, "final.pt")
    torch.save({
        "epoch": config.train.num_of_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, final_path)

    print(f"Final model saved to {final_path}")
