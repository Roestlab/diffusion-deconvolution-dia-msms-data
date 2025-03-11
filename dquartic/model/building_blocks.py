import math
import torch
import torch.nn as nn


def apply_rope(x):
    """
    Applies Rotary Position Embeddings (RoPE) to the input tensor.

    RoPE is a technique for incorporating position information into the model by
    rotating the feature space. Here, we split the hidden dimension into two equal
    halves (x1, x2) and apply sinusoidal rotations to encode positional information.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seqlen, hidden_dim).
                          The hidden_dim must be even.

    Returns:
        torch.Tensor: Tensor of the same shape as input (batch_size, seqlen, hidden_dim),
                      with RoPE applied.
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
    freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
    freq_seq = freq_seq / half_dim  # Normalize to range [0, 1)

    # Compute inverse frequencies using exponential decay
    inv_freq = 10000 ** (-freq_seq)

    # Create a sequence of positions
    positions = torch.arange(seqlen, dtype=dtype, device=device)

    # Compute the angles using outer product of positions and inverse frequencies
    angles = torch.einsum("i,j->ij", positions, inv_freq)

    # Compute sine and cosine of the angles
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    # Reshape for broadcasting
    sin_angles = sin_angles.unsqueeze(0)  # (1, seqlen, half_dim)
    cos_angles = cos_angles.unsqueeze(0)  # (1, seqlen, half_dim)

    # Separate the tensor into two halves
    x_reshaped = x.view(batch_size, seqlen, half_dim, 2)
    x1 = x_reshaped[..., 0]  # even indices
    x2 = x_reshaped[..., 1]  # odd indices

    # Apply the RoPE rotation
    x1_rotated = x1 * cos_angles - x2 * sin_angles
    x2_rotated = x1 * sin_angles + x2 * cos_angles

    # Combine the halves back together
    x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1).view(batch_size, seqlen, hidden_dim)

    return x_rotated


class TimeEmbedding(nn.Module):
    """
    A learnable time embedding module that encodes the timestep `t` into a hidden representation.

    This uses sinusoidal features (sin and cos) combined with two linear projections and a GELU
    activation to produce a learned time embedding.

    Args:
        hidden_dim (int): The dimensionality of the hidden representation.

    Example:
        >>> time_emb = TimeEmbedding(hidden_dim=128)
        >>> t = torch.tensor([0, 1, 2])  # shape: (3,)
        >>> output = time_emb(t)
        >>> print(output.shape)  # (3, 128)
    """
    def __init__(self, hidden_dim):
        super(TimeEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, t):
        """
        Forward pass of the time embedding.

        Args:
            t (torch.Tensor): 1D tensor of shape (batch_size,) containing the timesteps
                              for each sample in the batch.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_dim) representing
                          the learned time embeddings.
        """
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
    """
    A single layer of a custom Transformer architecture that performs cross-attention.

    This layer uses a MultiheadAttention mechanism to attend to the combination of
    the conditional sequence (x_cond) and the input sequence (x_t). Then it applies
    a feed-forward network with GELU activation.

    Args:
        hidden_dim (int): The dimensionality of the hidden representation.
        num_heads (int): The number of attention heads.

    Example:
        >>> layer = CustomTransformerLayer(hidden_dim=128, num_heads=4)
        >>> x_t = torch.randn(2, 10, 128)
        >>> x_cond = torch.randn(2, 5, 128)
        >>> out = layer(x_t, x_cond)
        >>> print(out.shape)  # (2, 10, 128)
    """
    def __init__(self, hidden_dim, num_heads):
        super(CustomTransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x_t, x_cond):
        """
        Forward pass of the custom Transformer layer.

        This performs cross-attention between the input sequence x_t and the
        conditional sequence x_cond, followed by a feed-forward sub-layer.

        Args:
            x_t (torch.Tensor): Input sequence of shape (batch_size, seqlen_t, hidden_dim).
            x_cond (torch.Tensor): Conditional sequence of shape (batch_size, seqlen_cond, hidden_dim).

        Returns:
            torch.Tensor: Updated x_t sequence with attention applied, shape:
                          (batch_size, seqlen_t, hidden_dim).
        """
        # Concatenate x_t and x_cond for cross-attention
        combined = torch.cat([x_cond, x_t], dim=1)

        # Self-attention (query = x_t, key = combined, value = combined)
        attn_output, _ = self.attention(
            query=x_t, key=combined, value=combined, need_weights=False
        )
        # Residual connection + layer normalization
        x_t = self.norm1(x_t + attn_output)

        # Feed-forward sub-layer
        ff_output = self.ff(x_t)
        x_t = self.norm2(x_t + ff_output)

        return x_t


class CustomTransformer(nn.Module):
    """
    A custom Transformer model that predicts the noise (epsilon) given a noisy input, time step, and condition.

    This network:
      1. Projects the noisy input (x_t) and the conditional input (x_cond) to a hidden_dim space.
      2. Applies Rotary Position Embeddings (RoPE) to encode positional information.
      3. Adds a time embedding to x_t to incorporate the timestep information.
      4. Passes the representations through multiple layers of cross-attention (CustomTransformerLayer).
      5. Projects the final hidden states back to the original dimension (input_dim).

    Args:
        input_dim (int): Dimension of the input (and output) sequences.
        hidden_dim (int): Dimension of the hidden representation in the Transformer.
        num_heads (int): Number of heads in each attention layer.
        num_layers (int): Number of Transformer layers to stack.

    Example:
        >>> model = CustomTransformer(input_dim=40000, hidden_dim=128, num_heads=1, num_layers=1)
        >>> x_t = torch.randn(2, 10, 40000)       # Noisy input
        >>> t = torch.tensor([0, 1])              # Timesteps
        >>> x_cond = torch.randn(2, 5, 40000)     # Conditional prior
        >>> eps_pred = model(x_t, t, x_cond)
        >>> print(eps_pred.shape)  # (2, 10, 40000)
    """
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
        self.layers = nn.ModuleList([
            CustomTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x_t, t, x_cond):
        """
        Forward pass of the CustomTransformer.

        Args:
            x_t (torch.Tensor): Noisy input at timestep t, shape (batch_size, seqlen1, input_dim).
            t (torch.Tensor): Timestep tensor of shape (batch_size,).
            x_cond (torch.Tensor): Conditional prior, shape (batch_size, seqlen2, input_dim).

        Returns:
            torch.Tensor: Predicted noise (epsilon) of shape (batch_size, seqlen1, input_dim).
        """
        # Project inputs
        x_t_proj = self.input_projection(x_t)
        # We squeeze out one dimension so that x_cond can be projected from shape (batch_size, seqlen2, 1)
        x_cond = x_cond.unsqueeze(2)
        x_cond_proj = self.conditional_projection(x_cond)

        # Apply RoPE positional embeddings
        x_t_proj = apply_rope(x_t_proj)
        x_cond_proj = apply_rope(x_cond_proj)

        # Time embeddings
        t_emb = self.time_embedding(t)  # (batch_size, hidden_dim)
        t_emb = t_emb[:, None, :]       # (batch_size, 1, hidden_dim)

        # Add time embeddings to x_t_proj
        x_t_proj = x_t_proj + t_emb

        # Pass through each Transformer layer
        for layer in self.layers:
            x_t_proj = layer(x_t_proj, x_cond_proj)

        # Project back to the original dimension
        eps_pred = self.output_projection(x_t_proj)

        return eps_pred
