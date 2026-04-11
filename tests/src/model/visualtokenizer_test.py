import torch
import pytest

from model import VisualTokenizer
from tests.utils import make_mock_images

def test_visual_tokenizer_output_shape():
    model = VisualTokenizer(is_pretrained=False)
    images = make_mock_images(batch_size=2, h=224, w=224)

    out = model(images)

    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[2] == model.get_out_dim() == 768
    assert out.shape[1] > 0


def test_visual_tokenizer_batch_size():
    model = VisualTokenizer(is_pretrained=False)
    images = make_mock_images(batch_size=4, h=224, w=224)

    out = model(images)

    assert out.shape[0] == 4
