import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderOnly(nn.Module):
    """
    Decoder-only model based on tarnsformer encoder architecture.

    Works with visual tokens produced by VisualTokenizer:
        image_tokens: [B, N_img, 768]

    The model builds a single autoregressive sequence:
        [image_tokens] + [text_tokens]

    and processes it with causal self-attention only.

    Args:
        vocab_size (int): Size of output vocabulary.
        pad_token_id (int): Padding token id.
        bos_token_id (int): Begin-of-sequence token id.
        eos_token_id (int): End-of-sequence token id.
        d_model (int): Transformer hidden size.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dim_feedforward (int): Feedforward hidden size.
        dropout (float): Dropout probability.
        max_seq_len (int): Maximum total sequence length:
            N_img + N_txt <= max_seq_len.
        visual_token_dim (int): Input dim of visual tokens.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        visual_token_dim: int = 768,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.visual_token_dim = visual_token_dim

        # project image tokens [B, N_img, 768] ->  to model dim [B, N_img, d_model]
        self.visual_proj = nn.Linear(visual_token_dim, d_model)

        # text token embeddings
        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_token_id)

        # positional embeddings for whole concatenated sequence [max_len, d_model]
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # transformer encoder with causal self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # post transformer processing
        self.norm = nn.LayerNorm(d_model)
        self.l_logits = nn.Linear(d_model, vocab_size) # [B, T, d_model] -> [B, T, vocab_size]

    @staticmethod
    def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns attention mask of shape [seq, seq].
        True means attention is forbidden.
        """
        # up right tokens in matrix is shadow
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tokens: [B, N_img, 768]
            text_tokens:  [B, T]

        Returns:
            logits: [B, T, vocab_size].
        """
        if image_tokens.ndim != 3:
            raise ValueError(f"image_tokens must have shape [B, N_img, C], got {image_tokens.shape}")
        if text_tokens.ndim != 2:
            raise ValueError(f"text_tokens must have shape [B, T], got {text_tokens.shape}")

        batch_size, n_img, c_img = image_tokens.shape
        _, t = text_tokens.shape

        if c_img != self.visual_token_dim:
            raise ValueError(f"Expected visual token dim = {self.visual_token_dim}, got {c_img}")

        total_len = n_img + t
        if total_len > self.max_seq_len:
            raise ValueError(f"Total sequence length {total_len} exceeds max_seq_len={self.max_seq_len}")

        # 1. trasnform visual prefix to model dimension
        visual_emb = self.visual_proj(image_tokens)  # [B, N_img, d_model]

        # 2. embedding for text prefix
        text_emb = self.token_embed(text_tokens)     # [B, T, d_model]

        # 3. concatenate: [image_prefix] + [text_prefix]
        x = torch.cat([visual_emb, text_emb], dim=1)  # [B, N_img + T, d_model] !!! N_img + T = seq

        # 4. add positional embeddings
        positions = torch.arange(total_len, device=x.device).unsqueeze(0)  # [1, seq] with [0, 1, 2, ..., seq - 1]
        x = x + self.pos_embed(positions)

        # 5. causal mask
        causal_mask = self.build_causal_mask(total_len, x.device)  # [seq, seq]

        # 6. padding mask
        # uncommit image tokens from masking => False
        image_padding_mask = torch.zeros(batch_size, n_img, dtype=torch.bool, device=x.device)

        # text pad positions => True
        text_padding_mask = (text_tokens == self.pad_token_id)

        key_padding_mask = torch.cat([image_padding_mask, text_padding_mask], dim=1)  # [B, L]

        # 7. transformer
        x = self.transformer(src=x, mask=causal_mask, src_key_padding_mask=key_padding_mask,)

        x = self.norm(x)

        # 8. keep only text positions for seq modeling
        text_hidden = x[:, n_img:, :]  # [B, T, d_model]
        return self.l_logits(text_hidden)  # [B, T, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        image_tokens: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation.

        Args:
            image_tokens: [B, N_img, 768]
            max_new_tokens: maximum number of generated tokens after <bos>

        Returns:
            generated tokens: [B, T_generated] (Includes <bos>).
        """
        batch_size = image_tokens.size(0)

        generated = torch.full(
            (batch_size, 1),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=image_tokens.device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=image_tokens.device)

        for _ in range(max_new_tokens):
            # [B, T, vocab_size]
            logits = self.forward(image_tokens=image_tokens, text_tokens=generated,)

            # [B, vocab_size]
            next_token_logits = logits[:, -1, :]
            # [B, 1] give token with max prob in vocab
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == self.eos_token_id)

            if finished.all():
                break

        return generated