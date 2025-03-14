import math

import torch
import torch.nn.functional as F

from einops import reduce

from .model_interface import ModelInterface


# beta schedule functions


def get_linear_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Creates a linear beta schedule between two endpoints over a set number of timesteps.

    Beta is used in diffusion models to control the noise level added at each timestep,
    gradually transitioning from data to pure noise.

    Args:
        num_timesteps (int): The number of timesteps for which to generate the beta values.
        beta_start (float): Starting value of beta.
        beta_end (float): Ending value of beta.

    Returns:
        torch.Tensor: A 1D tensor of length num_timesteps with linearly interpolated beta values.
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def get_cosine_beta_schedule(num_timesteps, s=0.008):
    """
    Cosine beta schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Computes a cosine beta schedule as described in the specified diffusion model literature.

    This schedule uses a cosine function to generate a non-linear progression of beta values,
    which can help control the variance of the process more effectively than a linear schedule.

    Args:
        num_timesteps (int): The number of timesteps for which to generate the beta values.
        s (float): A smoothing factor that adjusts the range of the cosine function.

    Returns:
        torch.Tensor: A 1D tensor of length num_timesteps with beta values computed using a cosine function.

    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_alphas(betas):
    """
    Calculates the alpha values from beta values for the diffusion process.

    Alpha values represent the proportion of the original data retained after adding noise.

    Args:
        betas (torch.Tensor): A tensor of beta values used in the diffusion process.

    Returns:
        torch.Tensor: A tensor of alpha values calculated as 1 - betas.
    """
    return 1.0 - betas


def get_alpha_bars(alpha):
    """
    Computes the cumulative product of alpha values across timesteps.

    Alpha bars represent the total proportion of the original data preserved over multiple timesteps.

    Args:
        alpha (torch.Tensor): A tensor of alpha values for each timestep.

    Returns:
        torch.Tensor: A tensor of cumulative product of alpha values.
    """
    return torch.cumprod(alpha, dim=0)


# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    Normalizes image data from [0, 1] to [-1, 1].

    Args:
        img (torch.Tensor): The image tensor to normalize.

    Returns:
        torch.Tensor: The normalized image tensor.
    """
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """
    Converts data from [-1, 1] back to [0, 1].

    Args:
        t (torch.Tensor): The tensor to unnormalize.

    Returns:
        torch.Tensor: The unnormalized tensor.
    """
    return (t + 1) * 0.5


def identity(t, *args, **kwargs):
    """
    A placeholder function that returns the input without any changes.

    Args:
        t (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The same tensor as the input.
    """
    return t


# other helper functions


def extract(a, t, x_shape):
    """
    Extracts elements from tensor `a` according to indices tensor `t` for given shape `x_shape`.

    This is commonly used to select specific elements from a beta or alpha tensor corresponding to
    particular timesteps in a batch.

    Args:
        a (torch.Tensor): Source tensor from which to extract values.
        t (torch.Tensor): Indices tensor, typically representing timesteps.
        x_shape (tuple): The shape of the output tensor.

    Returns:
        torch.Tensor: Reshaped tensor after extraction.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMDiffusionModel(ModelInterface):
    """
    Implements a denoising diffusion implicit model (DDIM) with various options for beta scheduling and prediction types.

    This class can be used to generate or reverse samples from noisy data back to clean data using
    specified beta schedules and model configurations.

    Args:
        model_class (torch.nn.Module): The neural network model to use for prediction.
        num_timesteps (int): The total number of timesteps in the diffusion process.
        beta_schedule_type (str): Type of beta schedule to use ('linear' or 'cosine').
        pred_type (str): Type of prediction ('eps' for noise prediction or 'x0' for initial state prediction).
        auto_normalize (bool): Whether to automatically normalize and unnormalize the input and output data.
        ms1_loss_weight (float): The weight for MS1-specific loss calculation.
        device (str): The device to use for computations ('cuda' or 'cpu').

    Attributes:
        betas (torch.Tensor): Tensor of beta values for the diffusion process.
        alphas (torch.Tensor): Tensor of alpha values for the diffusion process.
        alpha_bars (torch.Tensor): Tensor of cumulative alpha values for the diffusion process.
        normalize (function): Function to normalize data.
        unnormalize (function): Function to unnormalize data.
        pred_type (str): Type of prediction used in the diffusion process.
        ms1_loss_weight (float): The weight for the additional MS1-specific loss component.
        model (torch.nn.Module): The neural network model for prediction.
    """
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
        Samples from the distribution q(x_t | x_0) given the clean input x_0 and timestep t.

        This function simulates the forward diffusion process, adding noise to the clean input.

        Args:
            x_0 (torch.Tensor): The clean input data at the start of the diffusion process.
            t (torch.Tensor): A tensor of timestep indices.
            noise (torch.Tensor, optional): Noise tensor to add to the data. If None, noise is sampled randomly.

        Returns:
            torch.Tensor: The noisy data at timestep t.
        """
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t])[:, None, None]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.alpha_bars[t])[:, None, None]

        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, x_t, t, init_cond=None, attn_cond=None, precursor_cond=None):
        """
        Performs a reverse sampling step, estimating the original input x_0 or the noise epsilon.

        Depending on the prediction type (`pred_type`), this function either estimates the original input
        or the noise added at each step of the reverse diffusion process.

        Args:
            x_t (torch.Tensor): The noisy data tensor at timestep t.
            t (torch.Tensor): The current timestep index.
            init_cond (torch.Tensor, optional): Initial conditions to use for prediction.
            attn_cond (torch.Tensor, optional): Attention conditions to use for semantic guidance.
            precursor_cond (torch.Tensor, optional): Precursor feature masks used as additional conditioning signals.

        Returns:
            torch.Tensor: The estimated previous state x_{t-1}.
            torch.Tensor: The estimated noise or original input, depending on `pred_type`.
        """
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Compute constants
        alpha_bar_t = self.alpha_bars[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        if self.pred_type == "eps":
            # Predict noise
            eps_pred = self.model(x_t, t_tensor, init_cond, attn_cond, precursor_cond)
            # Compute x_0 prediction
            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        elif self.pred_type == "x0":
            # Predict x_0 directly
            x0_pred = self.model(x_t, t_tensor, init_cond, attn_cond, precursor_cond)
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

    def sample(self, x_t, ms2_cond=None, ms1_cond=None, precursor_cond=None, num_steps=1000):
        """
        Generates samples from the model starting from a noisy input x_t, optionally conditioned on MS2, MS1,
        and precursor feature masks.

        This function iteratively applies the reverse diffusion process (`p_sample`) to generate samples moving
        from noisy data back towards the original data distribution.

        Args:
            x_t (torch.Tensor): The initial noisy input tensor.
            ms2_cond (torch.Tensor, optional): MS2 mixture data maps for conditioning.
            ms1_cond (torch.Tensor, optional): Clean MS1 data maps for conditioning.
            precursor_cond (torch.Tensor, optional): Precursor feature masks used as additional conditioning signals.
            num_steps (int): The number of reverse sampling steps to perform.

        Returns:
            torch.Tensor: The final predicted clean data.
            torch.Tensor: The final predicted noise or input, depending on the conditioning.
        """
        ms2_cond = self.normalize(ms2_cond) if ms2_cond is not None else None
        ms1_cond = self.normalize(ms1_cond) if ms1_cond is not None else None
        precursor_cond = self.normalize(precursor_cond) if precursor_cond is not None else None
        pred_noise = None
        time_steps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for t in time_steps:
            t = t.long()
            x_t, pred_noise = self.p_sample(x_t, t.item(), ms2_cond, ms1_cond)

        x_t, pred_noise = self.unnormalize(x_t), self.unnormalize(pred_noise)

        if ms2_cond is not None:
            pred_noise = self.unnormalize(ms2_cond) - x_t

        return x_t, pred_noise

    def train_step(self, x_0, ms2_cond=None, ms1_cond=None, precursor_cond=None, noise=None, ms1_loss_weight=0.0):
        """
        Performs a single training step using the specified input data, optionally with additional MS1 loss weighting.

        This method takes clean data and conditions, applies the forward diffusion process (`q_sample`), and then
        calculates loss based on the model's predictions during reverse diffusion.

        Args:
            x_0 (torch.Tensor): The clean MS2 data maps (original data).
            ms2_cond (torch.Tensor, optional): MS2 mixture data maps for additional conditioning.
            ms1_cond (torch.Tensor, optional): Clean MS1 data maps for additional conditioning.
            precursor_cond (torch.Tensor, optional): Precursor feature masks used as additional conditioning signals.
            noise (torch.Tensor, optional): Noise tensor to use during forward diffusion. If None, noise is sampled randomly.
            ms1_loss_weight (float): Weighting factor for an additional MS1-specific loss component.

        Returns:
            torch.Tensor: The calculated loss for this training step.
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
            eps_pred = self.model(x_t, t, ms2_cond, ms1_cond, precursor_cond)
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
            x0_pred = self.model(x_t, t, ms2_cond, ms1_cond, precursor_cond)
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
