import torch
import torch.nn as nn

from .decoderonly import DecoderOnly
from .visualtokenizer import VisualTokenizer
from .visualadapter import VisualAdapter

# TODO: delete max seq len as param
# TODO: test with 300 num of img tokens


class Knn(nn.Module):

    """
    Full KNN-OCR model:
        image -> VisualTokenizer -> DecoderOnly -> text

    Args:
        vocab_size (int): Size of output vocabulary.
        pad_token_id (int): Padding token id.
        bos_token_id (int): Begin-of-sequence token id.
        eos_token_id (int): End-of-sequence token id.
        is_pretrained (bool): Whether to use pretrained backbone weights.
        d_model (int): Transformer hidden size.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dim_feedforward (int): Feedforward hidden size.
        dropout (float): Dropout probability.
        max_seq_len (int): Maximum total sequence length.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,

        feature_extractor: str = "swin",
        is_pretrain_backbone: bool = True,

        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = DecoderOnly.MAX_SEQ_LEN,
    ):
        super().__init__()

        # Backbone
        self.visual_tokenizer = VisualTokenizer(
            backbone=feature_extractor,
            is_pretrained=is_pretrain_backbone
        )

        self.visual_adapter = VisualAdapter(
            in_dim = self.visual_tokenizer.visual_token_dim, 
            out_dim = d_model, 
            num_tokens = DecoderOnly.NUM_OF_IMG_TOKENS
        )

        self.decoder = DecoderOnly(
            vocab_size = vocab_size,
            pad_token_id = pad_token_id,
            bos_token_id = bos_token_id,
            eos_token_id = eos_token_id,
            d_model = d_model,
            nhead = nhead,
            num_layers = num_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            max_seq_len = max_seq_len,
        )

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
            text_tokens: [B, T]

        Returns:
            logits: [B, T, vocab_size]
        """
        image_tokens = self.visual_tokenizer(images)        # [B, H', W', 768]
        img_prefix = self.visual_adapter(image_tokens)      # [B, K, D]
        logits = self.decoder(img_prefix, text_tokens)      # [B, T, vocab_size]

        return logits

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            images: [B, 3, H, W]
            max_new_tokens: maximum number of generated tokens after <bos>

        Returns:
            generated tokens: [B, T_generated]
        """
        image_tokens = self.visual_tokenizer(images)
        img_prefix = self.visual_adapter(image_tokens)
        return self.decoder.generate(image_tokens=img_prefix, max_new_tokens=max_new_tokens)