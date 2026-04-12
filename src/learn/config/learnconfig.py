from pydantic import BaseModel

class LearnDataConfig(BaseModel):
    path_to_trn_meta_db: str
    path_to_vld_meta_db: str
    path_to_tst_meta_db: str
    path_to_db: str
    path_to_vocabulary_file: str
    image_target_height: int
    batch_size: int


class LearnTrainConfig(BaseModel):
    num_of_epochs: int
    unfreeze_swin_epoch: int
    unfreeze_swin_norms_epoch: int
    save_model_per_epoch: bool
    output_model_dir: str
    optimizer_lr: float
    swin_optimizer_lr: float


class LearnModelConfig(BaseModel):
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    max_seq_len: int
    is_pretrain_swin: bool


class LearnConfig(BaseModel):
    data: LearnDataConfig
    train: LearnTrainConfig
    model: LearnModelConfig


# Example
#print(LearnConfig.model_validate_json(open("learnconfig.json").read()))

