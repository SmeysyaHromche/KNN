import torch
import torch.nn as nn
from torchvision.models import swin_v2_s, Swin_T_Weights


class VisualTokenizer(nn.Module):
    """
    Vision tokenizer based on Swin Transformer V2 small.

    Extracts high-level spatial features from an input image and converts them
    into a sequence of feature vectors suitable for transformer-based decoders
    or other sequence models.
    Args:
        is_pretrained (bool): If True, initializes the encoder with pretrained
            weights from the Swin Transformer V2 Small model. Default is True.
    Input:
        x (torch.Tensor): A batch of input images with shape [B, 3, H, W].
    Output:
        torch.Tensor: A batch of spatial feature sequences with shape [B, N, C],
            where N is the number of spatial locations (H' * W') and C is the
            feature dimension (768 for Swin V2 Small).
    """

    VISUAL_TOKEN_DIM = 768  # default for Swin V2 Small

    def __init__(self, is_pretrained: bool = True):
        super().__init__()
        
        _weight = Swin_T_Weights.DEFAULT if is_pretrained else None
        _vision_model = swin_v2_s(weights=_weight)

        self._feature_extractor = _vision_model.features
        self._norm = _vision_model.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self._feature_extractor(x)
        x = self._norm(x)
        # x: [B, H', W', C]
        
        b, h, w, c = x.shape
        return x.reshape(b, h * w, c)  # [B, N, C]
        
    def get_out_dim(self) -> int:
        return 768
