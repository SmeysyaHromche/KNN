import torch
import torch.nn as nn

from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class VisualTokenizer(nn.Module):
    """
    Backbone feature extractor, either:
        - Swin V2 Transformer Tiny
        - VGG16 with batch normalization
        - ConvNeXt Tiny

    Extracts high-level spatial features from an input image and converts them
    into a sequence of feature vectors suitable for transformer-based decoders
    or other sequence models.
    Args:
        is_pretrained (bool): If True, initializes the encoder with pretrained
            weights from the model. Default is True.
    Input:
        x (torch.Tensor): A batch of input images with shape [B, 3, H, W].
    Output:
        torch.Tensor: A batch of spatial feature sequences with shape [B, N, C],
            where N is the number of spatial locations (H' * W') and C is the
            feature dimension (768 for Swin V2 Small and ConvNeXt | 512 for VGG16).
    """

    def __init__(self, backbone: str = "swin", is_pretrained: bool = True):
        super().__init__()

        self.backbone = backbone
        
        if backbone == "swin":
            _weights = Swin_V2_T_Weights.DEFAULT if is_pretrained else None
            _vision_model = swin_v2_t(weights=_weights)

            self._feature_extractor = _vision_model.features
            self._norm = _vision_model.norm
            self.visual_token_dim = 768     # default for Swin V2 Small

        elif backbone == "vgg":
            _weights = VGG16_BN_Weights.DEFAULT if is_pretrained else None
            _model = vgg16_bn(weights=_weights)

            self._feature_extractor = _model.features
            self._norm = nn.Identity()      # VGG has no final norm
            self.visual_token_dim = 512     # default for VGG16

        elif backbone == "convnext":
            _weights = ConvNeXt_Tiny_Weights.DEFAULT if is_pretrained else None
            _model = convnext_tiny(weights=_weights)

            self._feature_extractor = _model.features
            self._norm = _model.avgpool     # ConvNeXt uses norm / avgpool pipeline
            self.visual_token_dim = 768     # default for ConvNeXt Tiny

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self._feature_extractor(x)

        # Swin case
        if self.backbone == "swin":
            x = self._norm(x)
            return x    # x: [B, H', W', C]
        
        # ConvNeXt case
        if self.backbone == "convnext":
            # [B, C, H, W] -> [B, H, W, C]
            if x.ndim == 4:
                x = x.permute(0, 2, 3, 1)
            return x
        
        # VGG case
        # x is still: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)   # x: [B, H, W, C]
        return x

