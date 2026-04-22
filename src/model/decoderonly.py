import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOnly(nn.Module):

    NUM_OF_IMG_TOKENS = 128  # if max = 1024
    MAX_SEQ_LEN = 1024
    MODEL_DIMENSION = 512
    
    """
    Decoder-only model based on tarnsformer encoder architecture.

    Works with visual tokens produced by VisualTokenizer:
        image_tokens: [B, N_img, d_model]

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
        img_prefix_len (int): Length of visual prefix.
        max_seq_len (int): Maximum total sequence length:
            N_img + N_txt <= max_seq_len.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        d_model: int = MODEL_DIMENSION,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        img_prefix_len: int = NUM_OF_IMG_TOKENS,
        max_seq_len: int = MAX_SEQ_LEN
    ):
        super().__init__()

        if img_prefix_len >= max_seq_len :
            raise ValueError(f"Error! The number of visual tokens ({img_prefix_len}) exceeds the limit for the line itself ({max_seq_len})")

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.img_prefix_len = img_prefix_len
        self.max_seq_len = max_seq_len
        self.text_seq_len = self.max_seq_len - self.img_prefix_len

        # text token embeddings
        self.token_embed = nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim = self.d_model, 
            padding_idx = self.pad_token_id
        )

        # typed embeddings (0 = img, 1 = text)
        self.type_emb = nn.Embedding(num_embeddings = 2, embedding_dim = self.d_model)
        # pos embedding based on token type
        self.img_pos_emb = nn.Embedding(self.img_prefix_len, self.d_model)
        self.txt_pos_emb = nn.Embedding(self.text_seq_len, self.d_model)
        # embedding dropout
        self.emd_drop = nn.Dropout(self.dropout)

        # transformer encoder with causal self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout,
            activation = "gelu",
            batch_first = True,
            norm_first = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = self.num_layers)

        # post transformer processing
        self.norm = nn.LayerNorm(normalized_shape = d_model)
        self.l_logits = nn.Linear(in_features = d_model, out_features = vocab_size) # [B, T, d_model] -> [B, T, vocab_size]

    @staticmethod
    def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns attention mask of shape [seq, seq].
        True means attention is forbidden.

        Args:
            seq_len: expected length of full input
            device: CPU/GPU
        """
        # up right tokens in matrix is shadow
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


    def _build_embedded_input(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply embeddings on separete tokens inputs and concat to one tensor.

        Args:
            image_tokens: [B, K, D]
            text_tokens:  [B, T]

        Returns:
            full sequence: [B, seq, d_model].
        """
        device = image_tokens.device
        # 1. data reading
        if image_tokens.ndim != 3:
            raise ValueError(f"Error! The image_tokens must have shape [B, K, D], got {image_tokens.shape}.")
        if text_tokens.ndim != 2:
            raise ValueError(f"Error! The text_tokens must have shape [B, T], got {text_tokens.shape}.")

        batch_size, n_img, c_img = image_tokens.shape
        if n_img != self.img_prefix_len :
            raise ValueError(f"Error! The number of visual tokens {n_img} differs from the expected number {self.img_prefix_len}.")
        if c_img != self.d_model:
            raise ValueError(f"Error! The dimension of visual tokens {c_img} differs from the expected number {self.d_model}.")
        batch_text, t = text_tokens.shape

        if batch_size != batch_text :
            raise ValueError(f"Error! The inconsistency between batch size of images tokens ({batch_size}) and text tokens ({batch_text})")

        total_len = n_img + t
        if total_len > self.max_seq_len:
            raise ValueError(f"Total sequence length {total_len} exceeds max_seq_len={self.max_seq_len}")

        img_pos_ids = torch.arange(n_img, device=device).unsqueeze(0).expand(batch_size, -1)
        txt_pos_ids = torch.arange(t, device=device).unsqueeze(0).expand(batch_size, -1)

        img_type_ids = torch.zeros((batch_size, n_img), dtype=torch.long, device=device)
        txt_type_ids = torch.ones((batch_size, t), dtype=torch.long, device=device)

        text_emb = self.token_embed(text_tokens)

        img_emb = image_tokens + self.img_pos_emb(img_pos_ids) + self.type_emb(img_type_ids)
        txt_emb = text_emb + self.txt_pos_emb(txt_pos_ids) + self.type_emb(txt_type_ids)



        # 5. concatenate: [image] + [text]
        x = torch.cat([img_emb, txt_emb], dim=1)  # [B, N_img + T, d_model] !!! N_img + T = seq

        # 6. embedding dropout
        x = self.emd_drop(x)

        return x



    def forward(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tokens: [B, K, D]
            text_tokens:  [B, T]

        Returns:
            logits: [B, T, vocab_size].
        """
        # 1. add positional information and concat data
        x = self._build_embedded_input(image_tokens, text_tokens)  # [B, seq, d_model]
        batch_size, total_len, _ = x.shape

        # 2. causal mask
        causal_mask = self.build_causal_mask(total_len, x.device)  # [seq, seq]

        # 3. padding mask
        # uncommit image tokens from masking => False
        image_padding_mask = torch.zeros(batch_size, self.img_prefix_len, dtype=torch.bool, device=x.device)

        # text pad positions => True
        text_padding_mask = (text_tokens == self.pad_token_id)

        key_padding_mask = torch.cat([image_padding_mask, text_padding_mask], dim=1)  # [B, L]

        # 4. transformer
        x = self.transformer(src=x, mask=causal_mask, src_key_padding_mask=key_padding_mask,)

        # 5. norm
        x = self.norm(x)

        # 6. keep only text positions for seq modeling
        text_hidden = x[:, self.img_prefix_len:, :]  # [B, T, d_model]
        
        # 7. transform to vocab dimension
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