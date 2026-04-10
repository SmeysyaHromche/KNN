import torch
import timm


class SwinFeatureExtractor(torch.nn.Module):
    """
    Swin2 transformer feature extractor for image lines.
    Converts an image into a sequence of feature vectors.
    """

    def __init__(self,
                 model_name: str = "swin2_tiny_patch4_window8_256",
                 pretrained: bool = True,
                 freeze: bool = True) -> None:
        """
        Initialize the Swin transformer.

        :param model_name: Name of the Swin model in timm.
        :param pretrained: Load pretrained weights.
        :param freeze: If true, freezes all parameters.
        """

        super().__init__()
        self.swin = timm.create_model(model_name, pretrained=pretrained)

        # Remove classifier head, so only features are extracted
        if hasattr(self.swin, "head"):
            self.swin.head = torch.nn.Identity()

        # Freeze parameters if needed (no gradients computed) - Swin is not trained
        if freeze:
            for p in self.swin.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Swin2 feature extractor.

        :param x: Input image tensor, shape [B, C, H, W], where
            B = batch size
            C = number of channels (3 for RGB images)
            H = image height
            W = image width
        :return: Feature tensor of shape [B, sequence_len, feature_dim], where
            B = batch size
            sequence_len = number of spatial patches
            feature_dim = channel dimension of Swin output

        Notes:
            - Forward pass is changed because the classifier head is removed.
            - Backward pass is computed automatically by PyTorch.
        """

        # Pass images through Swin2 backbone
        features = self.swin(x)

        # Flatten H * W into sequence_len
        # Transpose [B, C, sequence_len] into [B, sequence_len, C]
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        
        return features
