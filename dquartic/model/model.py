import math

import torch
import torch.nn.functional as F

from einops import reduce

from .model_interface import ModelInterface


# beta schedule functions


def get_linear_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Returns a linear beta schedule for the diffusion process.
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def get_cosine_beta_schedule(num_timesteps, s=0.008):
    """
    Cosine beta schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_alphas(betas):
    return 1.0 - betas


def get_alpha_bars(alpha):
    return torch.cumprod(alpha, dim=0)


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def identity(t, *args, **kwargs):
    return t


# other helper functions


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMDiffusionModel(ModelInterface):
    def __init__(
        self,
        model_class,
        num_timesteps=1000,
        beta_schedule_type="cosine",
        pred_type="eps",
        auto_normalize=True,
        ms1_loss_weight=0.0,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.model = None
        self.build(model_class, **kwargs)
        self.num_timesteps = num_timesteps
        self.device = device

        # Define beta schedule and compute alpha and alpha_bar

        self.betas = (
            get_linear_beta_schedule(num_timesteps).to(device)
            if beta_schedule_type == "linear"
            else get_cosine_beta_schedule(num_timesteps).to(device)
        ).to(torch.float32)
        self.alphas = get_alphas(self.betas).to(torch.float32)
        self.alpha_bars = get_alpha_bars(self.alphas).to(torch.float32)

        # calculate loss weight

        snr = self.alpha_bars / (1 - self.alpha_bars)

        if pred_type == "eps":
            self.loss_weight = torch.ones_like(snr)
        elif pred_type == "x0":
            self.loss_weight = snr
        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        # other parameters

        self.pred_type = pred_type
        self.ms1_loss_weight = ms1_loss_weight

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0)
        """
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t])[:, None, None]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.alpha_bars[t])[:, None, None]

        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, x_t, t, init_cond=None, attn_cond=None):
        """
        Perform a reverse sampling step.
        Can switch between predicting initial input x0 or noise eps.

        Args:
            x_t: Current state tensor at time t.
            t: Current timestep.
            init_cond: Conditioning input tensor to anchor the prediction.
            attn_cond: Conditioning input tensor with semantic information.
        """
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Compute constants
        alpha_bar_t = self.alpha_bars[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        if self.pred_type == "eps":
            # Predict noise
            eps_pred = self.model(x_t, t_tensor, t_tensor, init_cond, attn_cond)
            # Compute x_0 prediction
            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        elif self.pred_type == "x0":
            # Predict x_0 directly
            x0_pred = self.model(x_t, t_tensor, init_cond, attn_cond)
            # Compute eps_pred from x0_pred
            eps_pred = (x_t - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type}")

        # Compute x_{t-1}
        if t > 0:
            alpha_bar_t_prev = self.alpha_bars[t - 1]
            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
            sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1.0 - alpha_bar_t_prev)
            x_t_prev = sqrt_alpha_bar_t_prev * x0_pred + sqrt_one_minus_alpha_bar_t_prev * eps_pred
        else:
            x_t_prev = x0_pred

        return x_t_prev, eps_pred

    def sample(self, x_t, ms2_cond=None, ms1_cond=None, num_steps=1000):
        """
        Generate samples from the model.

        Args:
            x_t: Initial input tensor.
            ms2_cond: The MS2 mixture data maps.
            ms1_cond: The clean MS1 data maps.
            num_steps: Number of sampling steps.
        """
        ms2_cond = self.normalize(ms2_cond) if ms2_cond is not None else None
        ms1_cond = self.normalize(ms1_cond) if ms1_cond is not None else None
        pred_noise = None
        time_steps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for t in time_steps:
            t = t.long()
            x_t, pred_noise = self.p_sample(x_t, t.item(), ms2_cond, ms1_cond)

        x_t, pred_noise = self.unnormalize(x_t), self.unnormalize(pred_noise)

        if ms2_cond is not None:
            pred_noise = self.unnormalize(ms2_cond) - x_t

        return x_t, pred_noise

    def train_step(self, x_0, ms2_cond=None, ms1_cond=None, noise=None, ms1_loss_weight=0.0):
        """
        Perform a single training step.

        Args:
            x_0: The clean MS2 data maps (x0).
            ms2_cond: The MS2 mixture data maps.
            ms1_cond: The clean MS1 data maps.
            noise: Optional noise tensor. If None, it will be sampled randomly.
            ms1_loss_weight: Weight for the additional loss component.
        """
        batch_size = x_0.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        noise = (
            torch.randn_like(x_0, device=self.device) if noise is None else self.normalize(noise)
        )

        x_0 = self.normalize(x_0)
        ms2_cond = self.normalize(ms2_cond) if ms2_cond is not None else None
        ms1_cond = self.normalize(ms1_cond) if ms1_cond is not None else None
        x_t = self.q_sample(x_0, t, noise=noise)
        primary_loss, additional_loss = torch.zeros((batch_size,), device=self.device), torch.zeros(
            (batch_size,), device=self.device
        )

        if self.pred_type == "eps":
            # Predict noise
            eps_pred = self.model(x_t, t, ms2_cond, ms1_cond)
            # Compute primary loss between predicted noise and true noise
            primary_loss = F.mse_loss(eps_pred, noise)

            # Additional loss term
            if ms1_loss_weight > 0.0:
                # Compute summary ion chromatograms and calculate additional loss
                for func in (torch.sum, torch.mean, torch.max):
                    sic = func(x_t - eps_pred, dim=-1)
                    ms1_sic = func(ms1_cond, dim=-1)
                    additional_loss = additional_loss + F.mse_loss(
                        sic / torch.max(sic), ms1_sic / torch.max(ms1_sic)
                    )
        elif self.pred_type == "x0":
            # Predict x0
            x0_pred = self.model(x_t, t, ms2_cond, ms1_cond)
            # Compute primary loss between predicted x0 and true x0
            primary_loss = F.mse_loss(x0_pred, x_0)

            # Additional loss term
            if ms1_loss_weight > 0.0:
                # Compute ms1 pseudo chromatograms and calculate additional loss
                for func in (torch.sum, torch.mean, torch.max):
                    sic = func(x0_pred, dim=-1)
                    ms1_sic = func(ms1_cond, dim=-1)
                    additional_loss = additional_loss + F.mse_loss(
                        sic / torch.max(sic), ms1_sic / torch.max(ms1_sic)
                    )
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type}")

        # Combine primary loss and additional loss
        primary_loss = (
            reduce(primary_loss, "b ... -> b", "mean") if primary_loss.dim() > 1 else primary_loss
        )
        additional_loss = (
            reduce(additional_loss, "b ... -> b", "mean")
            if additional_loss.dim() > 1
            else additional_loss
        )
        loss = (
            (1 - ms1_loss_weight) * primary_loss + ms1_loss_weight * additional_loss
            if ms1_loss_weight > 0.0
            else primary_loss
        )
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss
