import torch
import pytest

from model import Knn
from tests.utils import make_mock_images, make_mock_text_tokens


def test_full_model_forward_shape():
    vocab_size = 40
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    model = Knn(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        is_pretrained=False,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=1024,
    )

    images = make_mock_images(batch_size=2, h=224, w=224)
    text_tokens = make_mock_text_tokens(batch_size=2, seq_len=5, vocab_size=vocab_size, pad_token_id=pad_token_id)

    logits = model(images, text_tokens)

    assert logits.shape == (2, 5, vocab_size)


def test_full_model_generate_shape():
    vocab_size = 40
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    model = Knn(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        is_pretrained=False,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=1024,
    )

    images = make_mock_images(batch_size=2, h=224, w=224)
    generated = model.generate(images, max_new_tokens=8)

    assert generated.ndim == 2
    assert generated.shape[0] == 2
    assert generated.shape[1] >= 1
    assert generated.shape[1] <= 9
    assert torch.all(generated[:, 0] == bos_token_id)