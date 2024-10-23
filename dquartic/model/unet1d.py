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
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# sinusoidal positional embeds


class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# building block modules


class Block(Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
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

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


# attention


class Attend(nn.Module):
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
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
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
    def __init__(self, dim, heads=2, dim_head=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
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
    def __init__(self, dim, heads=2, dim_head=8, flash=False, use_xattn=False, cond_dim=1):
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


# model


class Unet1D(Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        dropout=0.0,
        has_condition=False,
        cond_channels=None,
        cond_init_dim=None,
        learned_variance=False,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=8,
        attn_heads=2,
        attn_downsampled_l=625,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.has_condition = has_condition
        input_channels = channels

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

        # conditioning signal

        if self.has_condition:
            self.cond_proj = nn.Sequential(
                nn.Conv1d(cond_channels, cond_init_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(cond_init_dim, cond_init_dim, 1),
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

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim * attn_downsampled_l, mid_dim * attn_downsampled_l)
        self.mid_attn = Residual(
            PreNorm(
                mid_dim * attn_downsampled_l,
                Attention(
                    mid_dim * attn_downsampled_l,
                    dim_head=attn_dim_head,
                    heads=attn_heads,
                    use_xattn=self.has_condition,
                    cond_dim=cond_init_dim,
                ),
            )
        )
        self.mid_block2 = resnet_block(mid_dim * attn_downsampled_l, mid_dim * attn_downsampled_l)

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

    def forward(self, x, x_cond, time):
        x = rearrange(x, "b c n -> c b n")
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        c = (
            self.cond_proj(rearrange(x_cond, "b n -> b () n") if len(x_cond.shape) == 2 else x_cond)
            if self.has_condition
            else None
        )

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        c, _, n = x.shape
        x = rearrange(x, "c b n -> 1 (b n) c", c=c, n=n)
        x = self.mid_block1(x, t[0].unsqueeze(0))
        x = self.mid_attn(x, cond=c)
        x = self.mid_block2(x, t[0].unsqueeze(0))
        x = rearrange(x, "1 (b n) c -> c b n", c=c, n=n)

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
        x = rearrange(x, "c b n -> b c n")
        return x
