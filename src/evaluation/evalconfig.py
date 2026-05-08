from pydantic import BaseModel
from typing import Literal
from enum import Enum


class PadColor(float, Enum):
    WHITE = 1.0
    BLACK = 0.0

class EvalDataConfig(BaseModel):
    dataset: str
    path_to_tst_meta_db: str
    path_to_db: str
    path_to_vocabulary_file: str
    image_target_height: int
    batch_size: int


class EvalModelConfig(BaseModel):
    path_to_model: str
    device: str
    is_pretrain_swin: bool
    max_seq_len: int
    img_pad_value: PadColor


class EvalOutputConfig(BaseModel):
    path_to_output_file: str


class EvalConfig(BaseModel):
    data: EvalDataConfig
    model: EvalModelConfig
    output: EvalOutputConfig
