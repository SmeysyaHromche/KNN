from torch import nn

import torch
import torch.nn.functional as F

class VisualAdapter(nn.Module):

    def __init__(
            self, 
            in_dim: int,
            out_dim: int,
            num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.norm = nn.LayerNorm(out_dim)
    

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x [B, H', W', C]

        x = x.mean(dim=1)                               # x = [B, W', C] , make mean by H
        x = x.transpose(1, 2)                           # x = [B, C, W']
        x = F.adaptive_avg_pool1d(x, self.num_tokens)   # x = [B, C, K]
        x = x.transpose(1, 2)                           # x = [B, K, C]
        x = self.proj(x)                                # x = [B, K, D]
        x = self.norm(x)

        return x