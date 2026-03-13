import torch
import torch.nn as nn

class FSQ(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))

    def forward(self, z):
        z_bound = torch.tanh(z)
        half_l = (self.levels - 1) / 2
        z_scaled = z_bound * half_l
        z_rounded = torch.round(z_scaled)

        # We skip the STE here and do it in the main LightningModule forward pass
        # for cleaner loss tracking.
        return z_rounded / half_l


class Phi(nn.Module):
    def __init__(self, dim_in, dim_alpha: int, dim_mu: int):
        super().__init__()
        self.ff_alpha = nn.Sequential(
            nn.Linear(dim_in, 2*dim_in),
            nn.GELU(),
            nn.Linear(2*dim_in , dim_alpha)
        )
        self.ff_mu = nn.Sequential(
            nn.Linear(dim_in, 2*dim_in),
            nn.GELU(),
            nn.Linear(2*dim_in, dim_mu)
        )

    def forward(self, z):
        # WELD: Actually pass the data through the layers
        z_alpha = self.ff_alpha(z)
        z_mu = self.ff_mu(z)
        return z_mu, z_alpha




class Psi(nn.Module):
    def __init__(self, dim_mu: int, dim_alpha: int, dim_out):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim_mu + dim_alpha, (dim_mu + dim_alpha) * 2),
            nn.GELU(),
            nn.Linear((dim_mu + dim_alpha) * 2 , dim_out)
        )

    def forward(self, z_mu, z_alpha):
        # WELD: Replaced NumPy concat with PyTorch cat
        z = torch.cat([z_mu, z_alpha], dim=-1)
        z = self.ff(z)
        return z




class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, y, key_padding_mask=None):
        attn_out, _ = self.mha(query=x, key=y, value=y, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x



class ParticleSetEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, latent_nodes=32, out_channels=9, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(in_channels, hidden_dim)
        self.self_attn_layers = nn.ModuleList([MAB(hidden_dim, num_heads) for _ in range(num_layers)])

        self.latent_queries = nn.Parameter(torch.randn(1, latent_nodes, hidden_dim))
        self.cross_attn_pool = MAB(hidden_dim, num_heads)
        self.to_latent = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, mask=None):
        x = self.embed(x)
        # PyTorch expects True for ignored padding. Our mask is True for real data.
        pad_mask = ~mask if mask is not None else None

        for layer in self.self_attn_layers:
            x = layer(x, x, key_padding_mask=pad_mask)

        q = self.latent_queries.expand(x.size(0), -1, -1)
        latent = self.cross_attn_pool(q, x, key_padding_mask=pad_mask)
        return self.to_latent(latent)



class ParticleSetDecoder(nn.Module):
    def __init__(self, latent_channels=9, hidden_dim=64, out_nodes=256, out_channels=3, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(latent_channels, hidden_dim)
        self.out_queries = nn.Parameter(torch.randn(1, out_nodes, hidden_dim))
        self.cross_attn_expand = MAB(hidden_dim, num_heads)
        self.self_attn_layers = nn.ModuleList([MAB(hidden_dim, num_heads) for _ in range(num_layers)])
        self.to_physical = nn.Linear(hidden_dim, out_channels)

    def forward(self, z):
        z = self.embed(z)
        q = self.out_queries.expand(z.size(0), -1, -1)
        x = self.cross_attn_expand(q, z) 

        for layer in self.self_attn_layers:
            x = layer(x, x)

        return self.to_physical(x)

