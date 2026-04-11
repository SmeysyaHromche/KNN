import torch


def make_mock_images(batch_size: int = 2, h: int = 224, w: int = 224) -> torch.Tensor:
    return torch.randn(batch_size, 3, h, w)

def make_mock_text_tokens(
    batch_size: int = 2,
    seq_len: int = 6,
    vocab_size: int = 20,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Creates token ids in range [1, vocab_size - 1],
    then pads last positions for one sample to test padding path.
    """
    x = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    if seq_len >= 2:
        x[0, -1] = pad_token_id
    return x