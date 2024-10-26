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
        pred_type="eps",
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
        self.beta = get_beta_schedule(num_timesteps, beta_start, beta_end).to(device)
        self.alpha = get_alpha(self.beta)
        self.alpha_bar = get_alpha_bar(self.alpha)
        self.pred_type = pred_type
        self.ms1_loss_weight = ms1_loss_weight

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t])[:, None, None]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t])[:, None, None]

        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, x_t, x_cond, t):
        """
        Perform a reverse sampling step.
        Can switch between predicting initial input x0 or noise eps.

        Args:
            x_t: Current state tensor at time t.
            x_cond: Conditioning input tensor.
            t: Current timestep.
        """
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Compute constants
        alpha_bar_t = self.alpha_bar[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        if self.pred_type == "eps":
            # Predict noise
            eps_pred = self.model(x_t, x_cond, t_tensor)
            # Compute x_0 prediction
            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        elif self.pred_type == "x0":
            # Predict x_0 directly
            x0_pred = self.model(x_t, x_cond, t_tensor)
            # Compute eps_pred from x0_pred
            eps_pred = (x_t - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type}")

        # Compute x_{t-1}
        if t > 0:
            alpha_bar_t_prev = self.alpha_bar[t - 1]
            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
            sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)
            x_t_prev = sqrt_alpha_bar_t_prev * x0_pred + sqrt_one_minus_alpha_bar_t_prev * eps_pred
        else:
            x_t_prev = x0_pred

        return x_t_prev, eps_pred

    def sample(self, x, x_cond, num_steps=1000):
        """
        Generate samples from the model.

        Args:
            x: Initial input tensor.
            x_cond: Conditioning input tensor.
            num_steps: Number of sampling steps.
        """
        x_t = x
        pred_noise = None

        time_steps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for t in time_steps:
            t = t.long()
            x_t, pred_noise = self.p_sample(x_t, x_cond, t.item())

        return x_t, pred_noise

    def train_step(self, x_start, x_cond, noise=None, ms1_loss_weight=0.0):
        """
        Perform a single training step.

        Args:
            x_start: The initial data samples (x0).
            x_cond: Conditioning input tensor.
            noise: Optional noise tensor. If None, it will be sampled randomly.
            ms1_loss_weight: Weight for the additional loss component.
        """
        batch_size = x_start.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x_start) if noise is None else noise
        x_t = self.q_sample(x_start, t, noise)

        if self.pred_type == "eps":
            # Predict noise
            eps_pred = self.model(x_t, x_cond, t)
            # Compute primary loss between predicted noise and true noise
            primary_loss = F.mse_loss(eps_pred, noise)

            # Additional loss term
            if ms1_loss_weight > 0.0:
                # Compute normalized tic
                tic = torch.sum(x_t - eps_pred, dim=-1) / torch.max(tic)
                additional_loss = F.mse_loss(tic, x_cond)
            else:
                additional_loss = 0.0
        elif self.pred_type == "x0":
            # Predict x0
            x0_pred = self.model(x_t, x_cond, t)
            # Compute primary loss between predicted x0 and true x0
            primary_loss = F.mse_loss(x0_pred, x_start)

            # Additional loss term
            if ms1_loss_weight > 0.0:
                # Compute normalized tic
                tic = torch.sum(x_t - eps_pred, dim=-1) / torch.max(tic)
                additional_loss = F.mse_loss(tic, x_cond)
            else:
                additional_loss = 0.0
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type}")

        # Combine primary loss and additional loss
        loss = (1 - ms1_loss_weight) * primary_loss + ms1_loss_weight * additional_loss

        return loss
