import torch
import torch.nn.functional as F

from .model_interface import ModelInterface
from .building_blocks import get_beta_schedule, get_alpha, get_alpha_bar


class DDIMDiffusionModel(ModelInterface):
    def __init__(
        self,
        model_class,
        num_timesteps=1000,
        beta_start=0.001,
        beta_end=0.00125,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.model = None
        self.build(model_class, **kwargs)
        self.num_timesteps = num_timesteps
        self.device = device

        # Define beta schedule and compute alpha and alpha_bar
        self.beta = get_beta_schedule(num_timesteps, beta_start, beta_end).to(device)
        self.alpha = get_alpha(self.beta)
        self.alpha_bar = get_alpha_bar(self.alpha)

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t])[:, None, None]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t])[:, None, None]

        return (
            sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise,
            sqrt_one_minus_alpha_bar_t,
        )

    def p_sample(self, x_t, x_cond, t, eta=0.0):
        """
        Perform a reverse sampling step with DDIM update rule.
        """
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise
        eps_pred = self.model(x_t, x_cond, t_tensor)

        # Compute x_{t-1}
        alpha_bar_t = self.alpha_bar[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        # Compute x_0 prediction
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

        if t > 0:
            alpha_bar_t_prev = self.alpha_bar[t - 1]
            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
            sigma_t = (
                eta
                * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
            )
            noise = torch.randn_like(x_t)
            x_t_prev = (
                sqrt_alpha_bar_t_prev * x0_pred
                + torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps_pred
                + sigma_t * noise
            )
        else:
            x_t_prev = x0_pred

        return x_t_prev

    def sample(self, x, x_cond, num_steps=50, eta=0.0):
        """
        Generate samples from the model.
        """
        x_t = x

        time_steps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for t in time_steps:
            t = t.long()
            x_t = self.p_sample(x_t, x_cond, t.item(), eta)

        return x_t

    def train_step(self, x_start, x_cond, noise=None, ms1_loss_weight=0.0):
        """
        Perform a single training step.
        """
        batch_size = x_start.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x_start) if noise is None else noise
        x_t, noise_mult = self.q_sample(x_start, t, noise)

        # Predict noise
        eps_pred = self.model(x_t, x_cond, t)

        if ms1_loss_weight > 0.0:
            loss = (1 - ms1_loss_weight) * F.mse_loss(
                eps_pred, noise * noise_mult
            ) + ms1_loss_weight * F.mse_loss(torch.sum(x_t - eps_pred, dim=-1), x_cond)
        else:
            loss = F.mse_loss(eps_pred, noise * noise_mult)
        return loss
