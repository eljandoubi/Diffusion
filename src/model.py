import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 12,
        attn_p: float = 0,
        proj_p: float = 0,
        fused_attn: bool = True,
    ):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = int(in_channels / num_heads)
        self.scale: float = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.attn_p = attn_p
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, embed_dim = x.shape

        qkv: torch.Tensor = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
