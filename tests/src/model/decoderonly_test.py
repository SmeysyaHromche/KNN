import torch
import pytest

from model import DecoderOnly
from tests.utils import make_mock_text_tokens

def test_decoder_build_causal_mask():
    seq_len = 4
    mask = DecoderOnly.build_causal_mask(seq_len, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [False, True,  True,  True ],
            [False, False, True,  True ],
            [False, False, False, True ],
            [False, False, False, False],
        ],dtype=torch.bool,
    )

    assert mask.shape == (seq_len, seq_len)
    assert torch.equal(mask, expected)


def test_decoder_forward_shape():
    vocab_size = 30
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    model = DecoderOnly(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=512,
        visual_token_dim=768,
    )

    image_tokens = torch.randn(2, 16, 768)
    text_tokens = make_mock_text_tokens(batch_size=2, seq_len=5, vocab_size=vocab_size, pad_token_id=pad_token_id)

    logits = model(image_tokens, text_tokens)

    assert logits.shape == (2, 5, vocab_size)


def test_decoder_raises_on_wrong_visual_dim():
    vocab_size = 30
    model = DecoderOnly(
        vocab_size=vocab_size,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=512,
        visual_token_dim=768,
    )

    bad_image_tokens = torch.randn(2, 16, 512)  # wrong last dim
    text_tokens = torch.randint(1, vocab_size, (2, 5), dtype=torch.long)

    with pytest.raises(ValueError):
        _ = model(bad_image_tokens, text_tokens)


def test_decoder_generate_shape():
    vocab_size = 30
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    model = DecoderOnly(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=512,
        visual_token_dim=768,
    )

    image_tokens = torch.randn(2, 16, 768)
    generated = model.generate(image_tokens, max_new_tokens=10)

    assert generated.ndim == 2      # [B, T]
    assert generated.shape[0] == 2  # 2 batches
    assert generated.shape[1] >= 1  # min len of seq
    assert generated.shape[1] <= 11  # thresshold under max seq len
    assert torch.all(generated[:, 0] == bos_token_id) # first is bos always