import math
import torch
import torch.nn as nn


def apply_rope(x):
    """
    Applies Rotary Position Embeddings (RoPE) to the input tensor.

    Args:
        x: Input tensor of shape (batch_size, seqlen, hidden_dim)

    Returns:
        x_rotated: Tensor with RoPE applied, shape (batch_size, seqlen, hidden_dim)
    """
    # x: (batch_size, seqlen, hidden_dim)
    batch_size, seqlen, hidden_dim = x.size()
    device = x.device
    dtype = x.dtype

    # Ensure hidden_dim is even for splitting into two halves
    assert hidden_dim % 2 == 0, "hidden_dim must be even"

    # Compute half of the hidden dimension
    half_dim = hidden_dim // 2  # scalar

    # Create a sequence of frequencies (inverse frequencies)
    # freq_seq: (half_dim,)
    freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
    freq_seq = freq_seq / half_dim  # Normalize to range [0, 1)

    # Compute inverse frequencies using exponential decay
    # inv_freq: (half_dim,)
    inv_freq = 10000 ** (-freq_seq)

    # Create a sequence of positions
    # positions: (seqlen,)
    positions = torch.arange(seqlen, dtype=dtype, device=device)

    # Compute the angles using outer product of positions and inverse frequencies
    # angles: (seqlen, half_dim)
    angles = torch.einsum("i,j->ij", positions, inv_freq)

    # Compute sine and cosine of the angles
    # sin_angles: (seqlen, half_dim)
    # cos_angles: (seqlen, half_dim)
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    # Reshape sin_angles and cos_angles for broadcasting
    # sin_angles: (1, seqlen, half_dim)
    # cos_angles: (1, seqlen, half_dim)
    sin_angles = sin_angles.unsqueeze(0)
    cos_angles = cos_angles.unsqueeze(0)

    # Reshape x to separate even and odd dimensions
    # x_reshaped: (batch_size, seqlen, half_dim, 2)
    x_reshaped = x.view(batch_size, seqlen, half_dim, 2)

    # Split x into even and odd parts
    # x1: (batch_size, seqlen, half_dim) - even indices
    # x2: (batch_size, seqlen, half_dim) - odd indices
    x1 = x_reshaped[..., 0]
    x2 = x_reshaped[..., 1]

    # Apply RoPE rotations
    # x1_rotated: (batch_size, seqlen, half_dim)
    # x2_rotated: (batch_size, seqlen, half_dim)
    x1_rotated = x1 * cos_angles - x2 * sin_angles
    x2_rotated = x1 * sin_angles + x2 * cos_angles

    # Combine rotated parts and reshape back to original shape
    # x_rotated: (batch_size, seqlen, hidden_dim)
    x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1).view(batch_size, seqlen, hidden_dim)

    return x_rotated


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(TimeEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, t):
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


class CustomTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(CustomTransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x_t, x_cond):
        # Concatenate x_t and x_cond for cross-attention
        combined = torch.cat(
            [x_cond, x_t], dim=1
        )  # Shape: (batch_size, seqlen_cond + seqlen_t, hidden_dim)

        # Self-attention
        attn_output, _ = self.attention(query=x_t, key=combined, value=combined, need_weights=False)
        x_t = self.norm1(x_t + attn_output)

        # Feed-forward
        ff_output = self.ff(x_t)
        x_t = self.norm2(x_t + ff_output)

        return x_t


class CustomTransformer(nn.Module):
    def __init__(self, input_dim=40000, hidden_dim=128, num_heads=1, num_layers=1):
        super(CustomTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Projection layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        self.conditional_projection = nn.Linear(1, hidden_dim)

        # Time embedding
        self.time_embedding = TimeEmbedding(hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [CustomTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x_t, t, x_cond):
        """
        x_t: Noisy input at timestep t, shape (batch_size, seqlen1, input_dim)
        t: Timestep tensor, shape (batch_size,)
        x_cond: Conditional prior, shape (batch_size, seqlen2, input_dim)
        """
        # Project inputs
        x_t_proj = self.input_projection(x_t)  # Shape: (batch_size, seqlen1, hidden_dim)
        x_cond = x_cond.unsqueeze(2)  # Shape: (batch_size, seqlen2, 1)
        x_cond_proj = self.conditional_projection(
            x_cond
        )  # Shape: (batch_size, seqlen2, hidden_dim)

        # Apply RoPE positional embeddings
        x_t_proj = apply_rope(x_t_proj)
        x_cond_proj = apply_rope(x_cond_proj)

        # Time embeddings
        t_emb = self.time_embedding(t)  # Shape: (batch_size, hidden_dim)
        t_emb = t_emb[:, None, :]  # Shape: (batch_size, 1, hidden_dim)

        # Add time embeddings to x_t_proj
        x_t_proj = x_t_proj + t_emb

        # Pass through Transformer layers
        for layer in self.layers:
            x_t_proj = layer(x_t_proj, x_cond_proj)

        # Project back to original input dimension
        eps_pred = self.output_projection(x_t_proj)  # Shape: (batch_size, seqlen1, input_dim)

        return eps_pred
