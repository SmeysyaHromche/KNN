from typing import Literal
from pydantic import BaseModel

class LearnDataConfig(BaseModel):
    path_to_trn_meta_db: str
    path_to_vld_meta_db: str
    path_to_tst_meta_db: str
    path_to_db: str
    path_to_vocabulary_file: str
    image_target_height: int
    batch_size: int
    num_workers_train: int
    num_workers_validation: int


class LearnTrainConfig(BaseModel):
    num_of_epochs: int
    save_model_per_epoch: bool
    output_model_dir: str
    optimizer_lr: float
    device: str

    # Swin-specific configurations
    unfreeze_swin_epoch: int
    unfreeze_swin_norms_epoch: int
    swin_optimizer_lr: float

    # VGG-specific configurations
    unfreeze_vgg_epoch: int
    vgg_optimizer_lr: float

    # ConvNeXt-specific configurations
    unfreeze_convnext_epoch: int
    convnext_optimizer_lr: float


class LearnModelConfig(BaseModel):
    feature_extractor: Literal["swin", "vgg", "convnext"] = "swin"

    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    max_seq_len: int

    is_pretrain_swin: bool = True
    is_pretrain_vgg: bool = True
    is_pretrained_convnext: bool = True


class LearnConfig(BaseModel):
    data: LearnDataConfig
    train: LearnTrainConfig
    model: LearnModelConfig


# Example
#print(LearnConfig.model_validate_json(open("learnconfig.json").read()))

