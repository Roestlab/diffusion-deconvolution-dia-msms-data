import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from .model_interface import ModelInterface
from .building_blocks import get_beta_schedule, get_alpha, get_alpha_bar


class DDIMDiffusionModel(ModelInterface):
    def __init__(self, model, num_timesteps=1000, device="cuda"):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        # Define beta schedule and compute alpha and alpha_bar
        self.beta = get_beta_schedule(num_timesteps).to(device)
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

        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, x_t, x_cond, t, eta=0.0):
        """
        Perform a reverse sampling step with DDIM update rule.
        """
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise
        eps_pred = self.model(x_t, x_cond, t_tensor)

        # Compute x_{t-1}
        alpha_t = self.alpha[t]
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

    def sample(self, x_shape, x_cond, num_steps=50, eta=0.0):
        """
        Generate samples from the model.
        """
        batch_size = x_shape[0]
        x_t = torch.randn(x_shape, device=self.device)

        time_steps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for t in time_steps:
            t = t.long()
            x_t = self.p_sample(x_t, x_cond, t.item(), eta)

        return x_t

    def train_step(self, x_start, x_cond, noise=None):
        """
        Perform a single training step.
        """
        batch_size = x_start.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x_start) if noise is None else noise
        x_t = self.q_sample(x_start, t, noise)

        # Predict noise
        eps_pred = self.model(x_t, x_cond, t)

        loss = F.mse_loss(eps_pred, noise)
        return loss


def train_model(
    diffusion_model,
    dataloader,
    optimizer,
    num_epochs,
    device,
    use_wandb=True,
    checkpoint_path="best_model.pth",
    num_warmup_steps=0,
    num_training_steps=None,
):
    model = diffusion_model.model
    model.train()

    best_loss = float("inf")

    # Initialize the learning rate scheduler
    num_training_steps = (
        len(dataloader) * num_epochs if num_training_steps is None else num_training_steps
    )
    lr_scheduler = diffusion_model.lr_scheduler_class(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(num_epochs):
        dataloader.dataset.reset_epoch()
        epoch_losses = []

        for batch_idx, (ms2_1, ms1_1, ms2_2, ms1_2) in enumerate(dataloader):
            x_start, x_cond = ms2_1.to(device), ms1_1.to(device)  # Unpack and move to device
            x_noise = ms2_2.to(device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss = diffusion_model.train_step(x_start, x_cond, x_noise)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN detected at epoch {epoch}, batch {batch_idx}")
                continue  # Skip this iteration

            loss.backward()
            optimizer.step()

            # Step the learning rate scheduler
            lr_scheduler.step(epoch, loss.item())

            # Log batch loss
            if use_wandb:
                wandb.log(
                    {"batch/train_loss": loss.item(), "batch": batch_idx + epoch * len(dataloader)}
                )

            epoch_losses.append(loss.item())

        # Calculate average loss for the epoch
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")

        # Log epoch metrics
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(f"Epoch {epoch + 1}, Average Loss: {avg_train_loss:.4f}")

        # Check if this is the best loss so far
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            # Save the model checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1} with loss: {best_loss:.4f}")
