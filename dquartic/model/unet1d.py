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


class Attend(Module):
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
    def __init__(self, dim, heads=4, dim_head=32):
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
    def __init__(self, time_emb_dim, dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim * 2))

    def forward(self, x, t):
        scale, shift = self.to_scale_shift(t).chunk(2, dim=-1)

        return x * (scale + 1) + shift


class LayerNorm1d(Module):
    def __init__(self, channels, *, bias=True, eps=1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, channels, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1)) if bias else None

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g

        return norm + self.b if self.bias else norm


class FeedForward1d(Module):
    def __init__(self, channels, ch_mult=2):
        self.net = nn.Sequential(
            LayerNorm1d(channels=channels),
            nn.Conv1d(channels, channels * ch_mult, 1),
            nn.GELU(),
            nn.Conv1d(channels * ch_mult, channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class Transformer1d(Module):
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
        for attn, ra1, ff, ra2 in self.layers:
            x = attn(x, cond=cond) + x
            x = ra2(ff(ra1(x))) + x

        return x


# Fourier Space


class FourierFeatures(Module):
    def __init__(self, dim, h=10000, w=34):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2) * 0.02)
        self.h = h
        self.w = w

    def forward(self, x):
        _, _, h, w = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(h, w), dim=(2, 3), norm="ortho")

        return x


# model


class UNet1d(Module):
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
