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


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mlp_ratio: float = 4,
        act_layer=nn.GELU,
        mlp_p: float = 0,
    ):
        super().__init__()
        hidden_features = int(in_channels * mlp_ratio)
        self.fc1 = nn.Linear(in_channels, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(hidden_features, in_channels)
        self.drop2 = nn.Dropout(mlp_p)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        fused_attention: bool = True,
        num_heads: int = 4,
        mlp_ratio: float = 2,
        proj_p: float = 0,
        attn_p: float = 0,
        mlp_p: float = 0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(in_channels, eps=1e-6)

        self.attn = SelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            attn_p=attn_p,
            proj_p=proj_p,
            fused_attn=fused_attention,
        )

        self.norm2 = norm_layer(in_channels, eps=1e-6)
        self.mlp = MLP(
            in_channels=in_channels,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            mlp_p=mlp_p,
        )

    def forward(self, x: torch.Tensor):
        batch_size, channels, height, width = x.shape

        ### Reshape to batch_size x (height*width) x channels
        x = x.reshape(batch_size, channels, height * width).permute(0, 2, 1)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim, scaled_time_embed_dim):
        super().__init__()
        self.inv_freqs = self.register_buffer(
            "inv_freqs",
            1.0
            / (
                10000
                ** (torch.arange(0, time_embed_dim, 2).float() / (time_embed_dim / 2))
            ),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, scaled_time_embed_dim),
            nn.SiLU(),
            nn.Linear(scaled_time_embed_dim, scaled_time_embed_dim),
            nn.SiLU(),
        )

    def forward(self, timesteps: torch.Tensor):
        timestep_freqs = timesteps.unsqueeze(1) * self.inv_freqs.unsqueeze(0)
        embeddings = torch.cat(
            [torch.sin(timestep_freqs), torch.cos(timestep_freqs)], axis=-1
        )
        embeddings = self.time_mlp(embeddings)
        return embeddings
