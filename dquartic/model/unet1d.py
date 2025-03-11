""" adapted from @lucidrains 's DDPM implementation """

""" https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py """
import math
from collections import namedtuple
from functools import partial, wraps
from packaging import version

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange

from rotary_embedding_torch import RotaryEmbedding


# constants


AttentionConfig = namedtuple(
    "AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val

    return d() if callable(d) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called

        if called:
            return

        called = True

        return fn(x)

    return inner


print_once = once(print)


# small helper modules


class Residual(Module):
    """
    A residual block that adds the input to the output of the given function.

    Args:
        fn (Module): The function to apply to the input.

    Methods:
        forward(x, *args, **kwargs): Applies the function and adds the input to the output.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    """
    Creates an upsampling layer followed by a convolutional layer.

    Args:
        dim (int): The number of input channels.
        dim_out (int, optional): The number of output channels. Defaults to the input channels.

    Returns:
        nn.Sequential: A sequential container with upsampling and convolutional layers.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """
    Creates a downsampling layer using a convolutional layer.

    Args:
        dim (int): The number of input channels.
        dim_out (int, optional): The number of output channels. Defaults to the input channels.

    Returns:
        nn.Conv1d: A convolutional layer for downsampling.
    """
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.

    Args:
        dim (int): The number of input channels.

    Attributes:
        g (Parameter): The scaling parameter.

    Methods:
        forward(x): Applies RMS normalization to the input.
    """
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        """
        Applies RMS normalization to the input.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized tensor.
        """
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(Module):
    """
    Pre-normalization module that applies RMS normalization before the given function.

    Args:
        dim (int): The number of input channels.
        fn (Module): The function to apply after normalization.

    Attributes:
        fn (Module): The function to apply after normalization.
        norm (RMSNorm): The RMS normalization layer.

    Methods:
        forward(x, *args, **kwargs): Applies normalization and then the function.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        """
        Applies normalization and then the function.

        Args:
            x (Tensor): The input tensor.
            *args: Additional arguments for the function.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            Tensor: The output tensor after applying normalization and the function.
        """
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# sinusoidal positional embeds


class SinusoidalPosEmb(Module):
    """
    Sinusoidal positional embedding.

    Args:
        dim (int): The dimension of the embedding.
        theta (int, optional): The theta value for the embedding. Defaults to 10000.

    Attributes:
        dim (int): The dimension of the embedding.
        theta (int): The theta value for the embedding.

    Methods:
        forward(x): Applies sinusoidal positional embedding to the input.
    """
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """
        Applies sinusoidal positional embedding to the input.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The tensor with sinusoidal positional embedding applied.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


# building block modules

class Block(Module):
    """
    A basic building block with convolution, normalization, activation, and dropout.

    Args:
        dim (int): The number of input channels.
        dim_out (int): The number of output channels.
        dropout (float, optional): The dropout rate. Defaults to 0.0.

    Attributes:
        proj (nn.Conv1d): The convolutional layer.
        norm (RMSNorm): The normalization layer.
        act (nn.SiLU): The activation function.
        dropout (nn.Dropout): The dropout layer.

    Methods:
        forward(x, scale_shift=None): Applies the block operations to the input.
    """
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        """
        Applies the block operations to the input.

        Args:
            x (Tensor): The input tensor.
            scale_shift (tuple, optional): A tuple containing scale and shift tensors. Defaults to None.

        Returns:
            Tensor: The output tensor after applying the block operations.
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)

        return self.dropout(x)


class ResnetBlock(Module):
    """
    A ResNet block with optional time embedding.

    Args:
        dim (int): The number of input channels.
        dim_out (int): The number of output channels.
        time_emb_dim (int, optional): The dimension of the time embedding. Defaults to None.
        dropout (float, optional): The dropout rate. Defaults to 0.0.

    Attributes:
        mlp (nn.Sequential or None): The MLP for time embedding, if provided.
        block1 (Block): The first block.
        block2 (Block): The second block.
        res_conv (nn.Conv1d or nn.Identity): The residual convolution layer.

    Methods:
        forward(x, time_emb=None): Applies the ResNet block operations to the input.
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Applies the ResNet block operations to the input.

        Args:
            x (Tensor): The input tensor.
            time_emb (Tensor, optional): The time embedding tensor. Defaults to None.

        Returns:
            Tensor: The output tensor after applying the ResNet block operations.
        """
        scale_shift = None

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)
    
# attention

class Attend(Module):
    """
    Attention mechanism with optional flash attention.

    Args:
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        flash (bool, optional): Whether to use flash attention. Defaults to False.
        scale (float, optional): The scaling factor for the attention. Defaults to None.

    Attributes:
        dropout (float): The dropout rate.
        scale (float or None): The scaling factor for the attention.
        attn_dropout (nn.Dropout): The dropout layer for attention.
        flash (bool): Whether to use flash attention.
        cpu_config (AttentionConfig): The attention configuration for CPU.
        cuda_config (AttentionConfig or None): The attention configuration for CUDA.

    Methods:
        flash_attn(q, k, v): Applies flash attention to the input queries, keys, and values.
        forward(q, k, v): Applies the attention mechanism to the input queries, keys, and values.
    """
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        device_version = version.parse(f"{device_properties.major}.{device_properties.minor}")

        if device_version > version.parse("8.0"):
            print_once("A100 GPU detected, using flash attention if input tensor is on cuda")
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        """
        Applies flash attention to the input queries, keys, and values.

        Args:
            q (Tensor): The queries.
            k (Tensor): The keys.
            v (Tensor): The values.

        Returns:
            Tensor: The output of the flash attention.
        """
        is_cuda = q.is_cuda

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with nn.attention.sdpa_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        return out

    def forward(self, q, k, v):
        """
        Applies the attention mechanism to the input queries, keys, and values.

        Args:
            q (Tensor): The queries.
            k (Tensor): The keys.
            v (Tensor): The values.

        Returns:
            Tensor: The output of the attention mechanism.
        """
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class LinearAttention(Module):
    """
    Linear attention mechanism.

    Args:
        dim (int): The number of input channels.
        heads (int, optional): The number of attention heads. Defaults to 4.
        dim_head (int, optional): The dimension of each attention head. Defaults to 32.

    Attributes:
        scale (float): The scaling factor for the attention.
        heads (int): The number of attention heads.
        to_qkv (nn.Conv1d): The convolutional layer to generate queries, keys, and values.
        to_out (nn.Sequential): The output layer with convolution and normalization.

    Methods:
        forward(x): Applies the linear attention mechanism to the input.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        """
        Applies the linear attention mechanism to the input.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the linear attention mechanism.
        """
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)

        return self.to_out(out)

class Attention(Module):
    """
    Attention mechanism with optional cross-attention.

    Args:
        dim (int): The number of input channels.
        heads (int, optional): The number of attention heads. Defaults to 4.
        dim_head (int, optional): The dimension of each attention head. Defaults to 32.
        flash (bool, optional): Whether to use flash attention. Defaults to False.
        use_xattn (bool, optional): Whether to use cross-attention. Defaults to False.
        cond_dim (int, optional): The dimension of the conditioning input. Defaults to 1.

    Attributes:
        use_xattn (bool): Whether to use cross-attention.
        heads (int): The number of attention heads.
        rotary_emb (RotaryEmbedding): The rotary embedding for queries and keys.
        attend (Attend): The attention mechanism.
        to_qv (nn.Conv1d or None): The convolutional layer to generate queries and values for cross-attention.
        to_k (nn.Conv1d or None): The convolutional layer to generate keys for cross-attention.
        to_qkv (nn.Conv1d or None): The convolutional layer to generate queries, keys, and values for self-attention.
        to_out (nn.Conv1d): The output convolutional layer.

    Methods:
        forward(x, cond=None): Applies the attention mechanism to the input.
    """
    def __init__(self, dim, heads=4, dim_head=32, flash=False, use_xattn=False, cond_dim=1):
        super().__init__()
        self.use_xattn = use_xattn
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim=dim_head // 2)

        self.attend = Attend(flash=flash)

        if self.use_xattn:
            self.to_qv = nn.Conv1d(dim, hidden_dim * 2, 1, bias=False)
            self.to_k = nn.Conv1d(cond_dim, hidden_dim, 1, bias=False)
        else:
            self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, cond=None):
        """
        Applies the attention mechanism to the input.

        Args:
            x (Tensor): The input tensor.
            cond (Tensor, optional): The conditioning tensor for cross-attention. Defaults to None.

        Returns:
            Tensor: The output tensor after applying the attention mechanism.
        """
        if self.use_xattn and exists(cond):
            qv = self.to_qv(x).chunk(2, dim=1)
            q, v = map(lambda t: rearrange(t, "b (h c) n -> b h n c", h=self.heads), qv)
            k = rearrange(self.to_k(cond), "b (h c) n -> b h n c", h=self.heads)
        else:
            qkv = self.to_qkv(x).chunk(3, dim=1)
            q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h n c", h=self.heads), qkv)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b (h d) n")

        return self.to_out(out)


class HybridSelfAndCrossAttention(Module):
    """
    Hybrid self-attention and cross-attention mechanism.

    Args:
        dim (int): The number of input channels.
        heads (int, optional): The number of attention heads. Defaults to 4.
        dim_head (int, optional): The dimension of each attention head. Defaults to 32.
        flash (bool, optional): Whether to use flash attention. Defaults to False.
        cond_dim (int, optional): The dimension of the conditioning input. Defaults to 1.

    Attributes:
        heads (int): The number of attention heads.
        rotary_emb (RotaryEmbedding): The rotary embedding for queries and keys.
        attend (Attend): The attention mechanism.
        to_qkv (nn.Conv1d): The convolutional layer to generate queries, keys, and values for self-attention.
        to_qv (nn.Conv1d): The convolutional layer to generate queries and values for cross-attention.
        to_k (nn.Conv1d): The convolutional layer to generate keys for cross-attention.
        to_mid (nn.Conv1d): The intermediate convolutional layer.
        to_out (nn.Conv1d): The output convolutional layer.

    Methods:
        forward(x, cond): Applies the hybrid self-attention and cross-attention mechanism to the input.
    """
    def __init__(self, dim, heads=4, dim_head=32, flash=False, cond_dim=1):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim=dim_head // 2)

        self.attend = Attend(flash=flash)

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_qv = nn.Conv1d(dim, hidden_dim * 2, 1, bias=False)
        self.to_k = nn.Conv1d(cond_dim, hidden_dim, 1, bias=False)

        self.to_mid = nn.Conv1d(hidden_dim, dim, 1)

        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, cond):
        """
        Applies the hybrid self-attention and cross-attention mechanism to the input.

        Args:
            x (Tensor): The input tensor.
            cond (Tensor): The conditioning tensor for cross-attention.

        Returns:
            Tensor: The output tensor after applying the hybrid self-attention and cross-attention mechanism.
        """
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h n c", h=self.heads), qkv)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        x = self.attend(q, k, v)
        x = rearrange(x, "b h n d -> b (h d) n")

        mid = self.to_mid(x)

        qv = self.to_qv(mid).chunk(2, dim=1)
        q, v = map(lambda t: rearrange(t, "b (h c) n -> b h n c", h=self.heads), qv)
        k = rearrange(self.to_k(cond), "b (h c) n -> b h n c", h=self.heads)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)
        out = rearrange(out, "b h n d -> b (h d) n")

        return self.to_out(out)

# embedding model and helper classes


class ConditionalScaleShift(Module):
    """
    Applies a conditional scale and shift transformation to the input tensor.

    Args:
        time_emb_dim (int): The dimension of the time embedding.
        dim (int): The number of input channels.

    Attributes:
        to_scale_shift (nn.Sequential): The sequential layer to generate scale and shift values.

    Methods:
        forward(x, t): Applies the scale and shift transformation to the input tensor.
    """
    def __init__(self, time_emb_dim, dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim * 2))

    def forward(self, x, t):
        """
        Applies the scale and shift transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.
            t (Tensor): The time embedding tensor.

        Returns:
            Tensor: The transformed tensor.
        """
        scale, shift = self.to_scale_shift(t).chunk(2, dim=-1)
        return x * (scale + 1) + shift


class LayerNorm1d(Module):
    """
    Applies layer normalization to the input tensor.

    Args:
        channels (int): The number of input channels.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.

    Attributes:
        bias (bool): Whether to include a bias term.
        eps (float): A small value to avoid division by zero.
        g (Parameter): The scaling parameter.
        b (Parameter or None): The bias parameter.

    Methods:
        forward(x): Applies layer normalization to the input tensor.
    """
    def __init__(self, channels, *, bias=True, eps=1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, channels, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1)) if bias else None

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized tensor.
        """
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm
    

class FeedForward1d(Module):
    """
    A feedforward neural network for 1D data with layer normalization, convolution, and GELU activation.

    Args:
        channels (int): The number of input channels.
        ch_mult (int, optional): The channel multiplier for the hidden layer. Defaults to 2.

    Attributes:
        net (nn.Sequential): The sequential container with layer normalization, convolution, and GELU activation.

    Methods:
        forward(x): Applies the feedforward network to the input tensor.
    """
    def __init__(self, channels, ch_mult=2):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm1d(channels=channels),
            nn.Conv1d(channels, channels * ch_mult, 1),
            nn.GELU(),
            nn.Conv1d(channels * ch_mult, channels, 1),
        )

    def forward(self, x):
        """
        Applies the feedforward network to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the feedforward network.
        """
        return self.net(x)


class Transformer1d(Module):
    """
    A 1D Transformer model with optional cross-attention.

    Args:
        dim (int): The number of input channels.
        depth (int, optional): The number of layers in the transformer. Defaults to 4.
        heads (int, optional): The number of attention heads. Defaults to 4.
        dim_head (int, optional): The dimension of each attention head. Defaults to 32.
        mlp_dim (int, optional): The dimension of the feedforward network. Defaults to None.
        use_xattn (bool, optional): Whether to use cross-attention. Defaults to False.
        cond_dim (int, optional): The dimension of the conditioning input. Defaults to 1.

    Attributes:
        layers (ModuleList): The list of transformer layers.

    Methods:
        forward(x, cond=None): Applies the transformer model to the input tensor.
    """
    def __init__(
        self, dim, depth=4, heads=4, dim_head=32, mlp_dim=None, use_xattn=False, cond_dim=1
    ):
        super().__init__()
        self.layers = ModuleList([])
        for i in range(depth):
            if i < depth // 2 or not use_xattn:
                self.layers.append(
                    ModuleList(
                        [
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                            ),
                            FeedForward1d(dim, default(mlp_dim, dim * 2)),
                        ]
                    )
                )
            elif use_xattn:
                self.layers.append(
                    ModuleList(
                        [
                            HybridSelfAndCrossAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                cond_dim=cond_dim,
                            ),
                            FeedForward1d(dim, default(mlp_dim, dim * 2)),
                        ]
                    )
                )

    def forward(self, x, cond=None):
        """
        Applies the transformer model to the input tensor.

        Args:
            x (Tensor): The input tensor.
            cond (Tensor, optional): The conditioning tensor for cross-attention. Defaults to None.

        Returns:
            Tensor: The output tensor after applying the transformer model.
        """
        for attn, ra1, ff, ra2 in self.layers:
            x = attn(x, cond=cond) + x
            x = ra2(ff(ra1(x))) + x

        return x
    
# Fourier Space


class FourierFeatures(Module):
    """
    Applies Fourier features to the input tensor.

    Args:
        dim (int): The number of input channels.
        h (int, optional): The height of the Fourier feature grid. Defaults to 10000.
        w (int, optional): The width of the Fourier feature grid. Defaults to 34.

    Attributes:
        complex_weight (Parameter): The complex weight parameter for Fourier features.
        h (int): The height of the Fourier feature grid.
        w (int): The width of the Fourier feature grid.

    Methods:
        forward(x): Applies the Fourier features to the input tensor.
    """
    def __init__(self, dim, h=10000, w=34):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2) * 0.02)
        self.h = h
        self.w = w

    def forward(self, x):
        """
        Applies the Fourier features to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The tensor with Fourier features applied.
        """
        _, _, h, w = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(h, w), dim=(2, 3), norm="ortho")

        return x

# model

class UNet1d(Module):
    """
    A 1D U-Net model for diffusion-based deconvolution.

    Args:
        dim (int): The base dimension of the model.
        init_dim (int, optional): The initial dimension. Defaults to the base dimension.
        out_dim (int, optional): The output dimension. Defaults to the number of channels.
        dim_mults (tuple, optional): Multipliers for the dimensions at each level. Defaults to (1, 2, 4, 8).
        channels (int, optional): The number of input channels. Defaults to 3.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        conditional (bool, optional): Whether to use conditional inputs. Defaults to True.
        init_cond_channels (int, optional): The number of initial condition channels. Defaults to None.
        attn_cond_channels (int, optional): The number of attention condition channels. Defaults to None.
        attn_cond_init_dim (int, optional): The initial dimension for attention condition. Defaults to None.
        learned_variance (bool, optional): Whether to learn the variance. Defaults to False.
        sinusoidal_pos_emb_theta (int, optional): The theta value for sinusoidal positional embedding. Defaults to 10000.
        attn_heads (int, optional): The number of attention heads. Defaults to 4.
        attn_dim_head (int, optional): The dimension of each attention head. Defaults to 32.
        tfer_dim_mult (int, optional): The dimension multiplier for the transformer. Defaults to 620.
        tfer_depth (int, optional): The depth of the transformer. Defaults to 4.
        downsample_dim (int, optional): The dimension for downsampling. Defaults to 40000.
        simple (bool, optional): Whether to use a simple architecture. Defaults to True.
        pos_output_only (bool, optional): Whether to use only positive outputs. Defaults to False.

    Attributes:
        channels (int): The number of input channels.
        conditional (bool): Whether to use conditional inputs.
        init_conv (nn.Conv1d): The initial convolutional layer.
        time_mlp (nn.Sequential): The time embedding MLP.
        init_cond_proj (ConditionalScaleShift or None): The initial condition projection layer.
        attn_cond_proj (ModuleList or None): The attention condition projection layers.
        downs (ModuleList): The downsampling layers.
        ups (ModuleList): The upsampling layers.
        mid_block1 (ResnetBlock): The first middle block.
        mid_attn (Residual): The middle attention block.
        mid_block2 (ResnetBlock): The second middle block.
        final_res_block (ResnetBlock): The final residual block.
        final_conv (nn.Conv1d): The final convolutional layer.
        final_act (nn.Module): The final activation function.

    Methods:
        forward(x, time, init_cond=None, attn_cond=None): Forward pass of the model.
    """
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        dropout=0.0,
        conditional=True,
        init_cond_channels=None,
        attn_cond_channels=None,
        attn_cond_init_dim=None,
        learned_variance=False,
        sinusoidal_pos_emb_theta=10000,
        attn_heads=4,
        attn_dim_head=32,
        tfer_dim_mult=620,
        tfer_depth=4,
        downsample_dim=40000,
        simple=True,
        pos_output_only=False,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.conditional = conditional
        input_channels = channels + default(init_cond_channels, 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # conditioning signals

        if self.conditional:
            self.init_cond_proj = ConditionalScaleShift(
                time_emb_dim=time_dim, dim=init_cond_channels
            )
            attn_cond_init_dim = default(attn_cond_init_dim, dim * 2)
            self.attn_cond_proj = (
                ModuleList(
                    [
                        nn.Identity(),
                        nn.Sequential(
                            nn.Conv1d(attn_cond_channels, attn_cond_init_dim, 7, padding=3),
                            nn.GELU(),
                            nn.Conv1d(attn_cond_init_dim, attn_cond_init_dim, 1),
                        ),
                    ]
                )
                if simple
                else ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv1d(attn_cond_channels, attn_cond_init_dim, 7, padding=3),
                            resnet_block(attn_cond_init_dim, attn_cond_init_dim),
                            resnet_block(attn_cond_init_dim, attn_cond_init_dim),
                            Residual(
                                PreNorm(attn_cond_init_dim, LinearAttention(attn_cond_init_dim))
                            ),
                        ),
                        Transformer1d(
                            attn_cond_init_dim * tfer_dim_mult,
                            depth=tfer_depth // 2,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                        ),
                    ]
                )
            )

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        self.downsampled_n = downsample_dim // (2 ** (len(dim_mults) - 1))
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim * self.downsampled_n, mid_dim * self.downsampled_n)
        self.mid_attn = (
            Residual(
                PreNorm(
                    mid_dim * self.downsampled_n,
                    Attention(
                        mid_dim * self.downsampled_n,
                        heads=attn_heads,
                        dim_head=attn_dim_head,
                        use_xattn=self.conditional,
                        cond_dim=attn_cond_init_dim,
                    ),
                )
            )
            if simple
            else Residual(
                PreNorm(
                    mid_dim * self.downsampled_n,
                    Transformer1d(
                        mid_dim * self.downsampled_n,
                        depth=tfer_depth,
                        heads=attn_heads,
                        dim_head=attn_dim_head,
                        use_xattn=self.conditional,
                        cond_dim=attn_cond_init_dim,
                    ),
                )
            )
        )
        self.mid_block2 = resnet_block(mid_dim * self.downsampled_n, mid_dim * self.downsampled_n)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv1d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

        self.final_act = nn.Softplus() if pos_output_only else nn.Identity()

    def forward(self, x, time, init_cond=None, attn_cond=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): The input tensor.
            time (Tensor): The time embedding tensor.
            init_cond (Tensor, optional): The initial condition tensor. Defaults to None.
            attn_cond (Tensor, optional): The attention condition tensor. Defaults to None.

        Returns:
            Tensor: The output tensor after applying the model.
        """
        b = x.shape[0] if x.dim() == 3 else 1
        x = (
            rearrange(x, "b rt mz -> (b rt) () mz")
            if x.dim() == 3
            else rearrange(x, "rt mz -> rt () mz")
        )
        t = self.time_mlp(time)

        if self.conditional:
            init_cond = default(init_cond, lambda: torch.zeros_like(x))
            init_cond = (
                rearrange(init_cond, "b rt mz -> (b rt) () mz")
                if init_cond.dim() == 3
                else rearrange(init_cond, "rt mz -> rt () mz")
            )
            init_cond = self.init_cond_proj(init_cond, t)
            x = torch.cat((init_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        if self.conditional:
            attn_cond = default(attn_cond, lambda: torch.zeros_like(x))
            attn_cond = (
                rearrange(attn_cond, "b rt -> (b rt) () ()")
                if attn_cond.dim() == 2
                else rearrange(attn_cond, "b rt mz -> (b rt) () mz")
            )
            mz_net, rt_net = self.attn_cond_proj
            attn_cond = mz_net(attn_cond)
            attn_cond = rearrange(attn_cond, "(b rt) d mz -> b (d mz) rt", b=b)
            attn_cond = rt_net(attn_cond)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = rearrange(x, "(b rt) d mz -> b (d mz) rt", b=b)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, cond=attn_cond)
        x = self.mid_block2(x, t)
        x = rearrange(x, "b (d mz) rt -> (b rt) d mz", mz=self.downsampled_n)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = rearrange(x, "(b rt) d mz -> b (rt d) mz", b=b)

        return self.final_act(x)