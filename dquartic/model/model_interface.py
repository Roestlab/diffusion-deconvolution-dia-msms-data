from typing import List
import math
import functools
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import wandb

class LR_SchedulerInterface(object):
    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs):
        raise NotImplementedError

    def step(self, epoch: int, loss: float):
        """
        This method must be implemented in the sub-class. It will be called to get the learning rate for the next epoch.
        While the one we are using here does not need the loss value, this is left in case of using something like the ReduceLROnPlateau scheduler.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        loss : float
            The loss value of the current epoch.
        """
        raise NotImplementedError

    def get_last_lr(self) -> List[float]:
        """
        Get the last learning rate.

        Returns
        -------
        List[float]
            The last learning rate.
        """
        raise NotImplementedError


class WarmupLR_Scheduler(LR_SchedulerInterface):
    """
    A learning rate scheduler that includes a warmup phase and then a cosine annealing phase.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.lambda_lr = self.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch
        )

    def step(self, epoch: int = None, loss=None):
        """
        Get the learning rate for the next epoch.

        Parameters
        ----------
        epoch : int (Deprecated)
            The current epoch number.

        """
        return self.lambda_lr.step()

    def get_last_lr(self) -> List[float]:
        """
        Get the last learning rate.

        Returns
        -------
        List[float]
            The last learning rate.
        """
        return self.lambda_lr.get_last_lr()

    # `transformers.optimization.get_cosine_schedule_with_warmup` will import tensorflow,
    # resulting in some package version issues.
    # Here we copy the code from transformers.optimization
    def _get_cosine_schedule_with_warmup_lr_lambda(
        self,
        current_step: int,
        *,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float,
    ):
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            1e-10, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    def get_cosine_schedule_with_warmup(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        """
        Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`float`, *optional*, defaults to 0.5):
                The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
                following a half-cosine).
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        lr_lambda = functools.partial(
            self._get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
        return LambdaLR(optimizer, lr_lambda, last_epoch)


class CallbackHandler:
    """
    A CallbackHandler class that can be used to add callbacks to the training process for both
    epoch-level and batch-level events. To have more control over the training process, you can
    create a subclass of this class and override the methods you need.
    """

    def epoch_callback(self, epoch: int, epoch_loss: float) -> bool:
        """
        This method will be called at the end of each epoch. The callback can also be used to
        stop the training by returning False. If the return value is None, or True, the training
        will continue.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        epoch_loss : float
            The loss value of the current epoch.

        Returns
        -------
        continue_training : bool
            If False, the training will stop.
        """
        continue_training = True
        return continue_training

    def batch_callback(self, batch: int, batch_loss: float):
        """
        This method will be called at the end of each batch.

        Parameters
        ----------
        batch : int
            The current batch number.
        batch_loss : float
            The loss value of the current batch.

        """
        pass
    

class ModelInterface(object):
    """
    Provides standardized methods to interact
    with ml models. Inherit into new class and override
    the abstract (i.e. not implemented) methods.
    """

    #################
    # Magic Methods #
    #################
    
    def __init__(
        self,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        min_pred_value: float = 0.0,
        **kwargs,
    ):
        """
        Parameters
        ----------

        min_pred_value : float, optional
            See :attr:`min_pred_value`, defaults to 0.0.
        """
        self.model: torch.nn.Module = None
        self.optimizer = None
        self.model_params: dict = {}
        self.min_pred_value = min_pred_value
        self.lr_scheduler_class = WarmupLR_Scheduler
        self.callback_handler = CallbackHandler()
        self.device = device
        
        # Logging
        self.use_wandb = False
        
    ##################
    # Public Methods #
    ##################
    
    def get_parameter_num(self):
        """
        Get total number of parameters in model.
        """
        return np.sum([p.numel() for p in self.model.parameters()])
    
    def train_with_warmup(
        self,
        dataloader,
        num_epochs,
        num_warmup_steps=5,
        learning_rate=1e-4,
        use_wandb=True,
        checkpoint_path="best_model.pth"
    ):
        """
        Train the model according to specifications. Includes a warumup
        phase with linear increasing and cosine decreasing for lr scheduling).
        """
        
        self._set_lr(learning_rate)
        # Initialize the learning rate scheduler
        lr_scheduler = self._get_lr_schedule_with_warmup(num_warmup_steps, num_epochs)
        
        # model = diffusion_model.model
        self.model.train()

        best_loss = float("inf")

        for epoch in range(num_epochs):
            dataloader.dataset.reset_epoch()

            batch_loss = self._train_one_epoch(epoch, dataloader)

            # Step the learning rate scheduler
            lr_scheduler.step(epoch, np.mean(batch_loss))


            # Calculate average loss for the epoch
            avg_train_loss = np.mean(batch_loss)

            # Log epoch metrics
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": avg_train_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    }
                )

            print(f"[Training] Epoch={epoch+1}, lr={lr_scheduler.get_last_lr()[0]}, loss={np.mean(batch_loss)}")

            # Check if this is the best loss so far
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                # Save the model checkpoint
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved at epoch {epoch + 1} with loss: {best_loss:.4f}")
            
            continue_training = self.callback_handler.epoch_callback(
                epoch=epoch, epoch_loss=np.mean(batch_loss)
            )
            if not continue_training:
                print(f"Training stopped at epoch {epoch}")
                break
            
    def train(
        self,
        dataloader,
        batch_size,
        epochs,
        warmup_epochs: int = 5,
        learning_rate: float = 1e-4,
        use_wandb: bool = False,
        checkpoint_path: str = "best_model.pth",
         **kwargs,
    ):
        """
        Train the model with the given dataloader for the given number of epochs.
        """
        if warmup_epochs > 0:
            self.train_with_warmup(
                dataloader,
                epochs,
                num_warmup_steps=warmup_epochs,
                learning_rate=learning_rate,
                use_wandb=use_wandb,
                checkpoint_path=checkpoint_path,
                 **kwargs,
            ) 
        else:
            self._prepare_training(learning_rate, **kwargs)
            best_loss = float("inf")
            
            for epoch in range(epochs):
                dataloader.dataset.reset_epoch()
                batch_loss = self._train_one_epoch(epoch, dataloader)
                
                # Log epoch metrics
                if use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/loss": np.mean(batch_loss),
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    )
                
                print(f"[Training] Epoch={epoch+1}, lr={self.optimizer.param_groups[0]['lr']}, loss={np.mean(batch_loss)}")
                
                if np.mean(batch_loss) < best_loss:
                    best_loss = np.mean(batch_loss)
                    # Save the model checkpoint
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"Model checkpoint saved at epoch {epoch + 1} with loss: {best_loss:.4f}")
                
                continue_training = self.callback_handler.epoch_callback(
                    epoch=epoch, epoch_loss=np.mean(batch_loss)
                )
                if not continue_training:
                    print(f"Training stopped at epoch {epoch}")
                    break
            
        
    ###################
    # Private Methods #
    ###################
    
    def _init_for_training(self):
        """
        Set the loss function, and more attributes for different tasks.
        The default loss function is nn.L1Loss.
        """
        self.loss_func = torch.nn.L1Loss()
        
    def _prepare_training(self, lr: float, **kwargs):
        self.model.train()
        self._set_lr(lr)
        
    def _set_optimizer(self, lr):
        """Set optimizer"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def _set_lr(self, lr: float):
        """Set learning rate"""
        if self.optimizer is None:
            self._set_optimizer(lr)
        else:
            for g in self.optimizer.param_groups:
                g["lr"] = lr
                
    def _get_lr_schedule_with_warmup(self, warmup_epoch, epoch):
        """
        Returns a learning rate scheduler with warmup.

        Args:
            warmup_epoch (int): The number of warmup epochs.
            epoch (int): The total number of training epochs.

        Returns:
            torch.optim.lr_scheduler: The learning rate scheduler with warmup.
        """
        if warmup_epoch > epoch:
            warmup_epoch = epoch // 2
        return self.lr_scheduler_class(
            self.optimizer, num_warmup_steps=warmup_epoch, num_training_steps=epoch
        )
        
    def _train_one_batch(self, x_start, x_cond, x_noise):
        """Train one batch"""
        self.optimizer.zero_grad()
        loss = self.model.train_step(x_start, x_cond, x_noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    
    def _train_one_epoch(self, epoch, dataloader):
        """Train one epoch"""
        self.model.train()
        batch_loss = []
        for batch_idx, (ms2_1, ms1_1, ms2_2, ms1_2) in enumerate(dataloader):
            x_start, x_cond = ms2_1.to(self.device), ms1_1.to(self.device)
            x_noise = ms2_2.to(self.device)
            loss = self._train_one_batch(x_start, x_cond, x_noise)
            batch_loss.append(loss)
            
            if self.use_wandb:
                wandb.log(
                    {"batch/train_loss": loss, "batch": batch_idx + epoch * len(dataloader)}
                )
    
        return batch_loss
            