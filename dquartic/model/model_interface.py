from typing import List
import os
import math
import functools
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
import wandb
from io import BytesIO
from PIL import Image as PILImage

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

    def __repr__(self):
        return f"{self.__class__.__name__} with {self.model.__class__.__name__} model with {self.get_parameter_num()} parameters on {self.device}"

    ##################
    # Public Methods #
    ##################

    def build(self, model_class, **kwargs):
        """
        Build the model.
        """
        self.model = model_class
        self._init_for_training()

    def get_parameter_num(self):
        """
        Get total number of parameters in model.
        """
        return np.sum([p.numel() for p in self.model.parameters()])

    def train_step(self, x_start, x_cond, noise=None):
        """
        Perform a single training step. Implemented in the subclass.
        """
        raise NotImplementedError

    def sample(self, x_start, x_cond, num_steps=100, eta=0.0):
        """
        Generate samples from the model. Implemented in the subclass.
        """
        raise NotImplementedError

    def train_with_warmup(
        self,
        dataloader,
        num_epochs,
        num_warmup_steps=5,
        learning_rate=1e-4,
        use_wandb=True,
        checkpoint_path="best_model.ckpt"
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

        # Load checkpoint if available
        start_epoch, best_loss, lr_scheduler = self.load_checkpoint(lr_scheduler, f"{os.path.dirname(checkpoint_path)}{os.path.sep}dquartic_latest_checkpoint.ckpt", self.device)

        best_epoch = start_epoch

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
            
            self.save_checkpoint(None, epoch, np.mean(batch_loss), f"{os.path.dirname(checkpoint_path)}{os.path.sep}dquartic_latest_checkpoint.ckpt")             
            
            # Check if this is the best loss so far
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                best_epoch = epoch + 1
                self.save_checkpoint(lr_scheduler, epoch, best_loss, checkpoint_path)
                if use_wandb:
                    self.log_single_prediction(best_epoch, best_loss, dataloader, num_steps=[100, 500, 1000], eta=0.0)

            continue_training = self.callback_handler.epoch_callback(
                epoch=epoch, epoch_loss=np.mean(batch_loss)
            )
            if not continue_training:
                print(f"Training stopped at epoch {epoch}")
                break
        print(f"Best model checkpoint saved at epoch {best_epoch} with loss: {best_loss:.6f}")

    def train(
        self,
        dataloader,
        batch_size,
        epochs,
        warmup_epochs: int = 5,
        learning_rate: float = 1e-4,
        use_wandb: bool = False,
        checkpoint_path: str = "best_model.ckpt",
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

            # Load checkpoint if available
            start_epoch, best_loss, _ = self.load_checkpoint(None, f"{os.path.dirname(checkpoint_path)}{os.path.sep}dquartic_latest_checkpoint.ckpt", self.device)

            best_epoch = start_epoch

            for epoch in range(start_epoch, epochs):
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
                
                self.save_checkpoint(None, epoch, np.mean(batch_loss), f"{os.path.dirname(checkpoint_path)}{os.path.sep}dquartic_latest_checkpoint.ckpt")
                
                if np.mean(batch_loss) < best_loss:
                    best_loss = np.mean(batch_loss)
                    best_epoch = epoch + 1
                    self.save_checkpoint(None, epoch, best_loss, checkpoint_path)

                    # Only log predictions if using wandb and for the firt epoch and then only after epoch 15 every 500 epochs if the loss is still the best
                    if use_wandb and (epoch == 0 or (epoch > 15 and epoch % 500 == 0)):
                        self.log_single_prediction(best_epoch, best_loss, dataloader, num_steps=[100, 500, 1000], eta=0.0)

                continue_training = self.callback_handler.epoch_callback(
                    epoch=epoch, epoch_loss=np.mean(batch_loss)
                )
                if not continue_training:
                    print(f"Training stopped at epoch {epoch}")
                    break
            print(f"Best model checkpoint saved at epoch {best_epoch} with loss: {best_loss:.6f}")

    def load_checkpoint(self, scheduler, checkpoint_path, device):
        """
        Load model, optimizer, and scheduler states from a checkpoint file if available.
        """
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.lambda_lr.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print(f"Resumed from ({checkpoint_path}) epoch {epoch}, best loss {best_loss:.6f}")
        else:
            print(f"No checkpoint ({checkpoint_path}) found. Starting from scratch.")
            epoch = 0
            best_loss = float('inf')
            scheduler = scheduler

        return epoch, best_loss, scheduler

    def save_checkpoint(self, scheduler, epoch, best_loss, checkpoint_path):
        """
        Save the current state of the model, optimizer, and scheduler to a checkpoint file.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.lambda_lr.state_dict() if scheduler is not None else None,
            'best_loss': best_loss,
        }, checkpoint_path)

    def predict(self, dataloader, mixture_weights=(0.5, 0.75), num_steps=100, eta=0.0):
        self.model.eval()
        preds = np.array([])
        for ms2_1, ms1_1, ms2_2, ms1_2 in dataloader:
            x_start, x_cond = ms2_1, ms1_1  
            x_start = x_start.to(self.device)
            x_cond = x_cond.to(self.device)
            # Simulated mixed spectra from target sample and other sample
            x_start_rand = (ms2_1*mixture_weights[0]) + (ms2_2*mixture_weights[1])
            x_start_rand = x_start_rand.to(self.device).unsqueeze(0)
            pred = self._predict_one_batch(x_start_rand, x_cond, num_steps, eta)
            # Store ms2_1, ms1_1, x_start_rand, pred
            pred_data = {'ms2_1': ms2_1, 'ms1_1': ms1_1, 'x_start_rand': x_start_rand.cpu().detach().numpy(), 'pred': pred}
            preds = np.append(preds, pred_data)
        return preds

    def log_single_prediction(
        self,
        epoch,
        loss,
        dataloader,
        sample_idx=None,
        mixture_weights=(0.5, 0.75),
        num_steps=[100, 500, 1000],
        eta=0.0,
    ):
        """Log a wandb.Table with matplotlib figures for Target MS2, Target MS1, Random MS2 Input, and Predicted MS2."""
        # Create a wandb Table to log
        table = wandb.Table(
            columns=[
                "Num Steps",
                "Epoch",
                "Loss",
                "Target MS2",
                "Target MS1",
                "Noise MS2 Input",
                "Predicted MS2",
            ]
        )
        
        for _num_steps in num_steps:
            # Get sample and prediction
            ms2_target_plot, ms1_plot, ms2_input_plot, pred_plot = self.plot_single_prediction(
                dataloader,
                sample_idx=sample_idx,
                mixture_weights=mixture_weights,
                num_steps=_num_steps,
                eta=eta,
            )
            
            table.add_data(
                _num_steps,
                epoch,
                loss,
                wandb.Image(PILImage.open(self._convert_mpl_fig_to_bytes(ms2_target_plot.superFig))),
                wandb.Image(PILImage.open(self._convert_mpl_fig_to_bytes(ms1_plot.superFig))),
                wandb.Image(PILImage.open(self._convert_mpl_fig_to_bytes(ms2_input_plot.superFig))),
                wandb.Image(PILImage.open(self._convert_mpl_fig_to_bytes(pred_plot.superFig))),
            )
        wandb.log({"predictions_table": table}, commit=False)

    def plot_single_prediction(
        self,
        dataloader,
        sample_idx=None,
        mixture_weights=(0.5, 0.75),
        num_steps=100,
        eta=0.0,
        plot_type="peakmap",
        plot_3d=True,
        backend="ms_matplotlib",
    ):
        """
        Plot a matplotlib figure with Target MS2, Target MS1, Random MS2 Input, and Predicted MS2.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader.
            sample_idx (int, optional): The index of the sample to plot. Defaults to None, in which case a random sample is chosen.
            mixture_weights (tuple, optional): The weights for the mixture of the two samples. Defaults to (0.5, 0.75).
            num_steps (int, optional): The number of steps to predict. Defaults to 100.
            eta (float, optional): The eta value. Defaults to 0.0.
            plot_type (str, optional): The type of plot. Defaults to 'peakmap'.
            plot_3d (bool, optional): Whether to plot in 3D. Defaults to True.
            backend (str, optional): The backend for plotting. Defaults to 'ms_matplotlib'.
        """

        try:
            import pyopenms_viz
        except ImportError:
            raise ImportError(
                "pyopenms_viz is required for plotting. Install it with `pip install pyopenms_viz`."
            )

        if sample_idx is None:
            sample_idx = np.random.randint(len(dataloader.dataset))
        ms2_1, ms1_1, ms2_2, ms1_2 = dataloader.dataset[sample_idx]
        x_start, x_cond = ms2_1, ms1_1
        x_start = x_start.to(self.device).unsqueeze(0)
        x_cond = x_cond.to(self.device).unsqueeze(0)

        # Simulated mixed spectra from target sample and other sample
        x_start_rand = (ms2_1*mixture_weights[0]) + (ms2_2*mixture_weights[1])
        x_start_rand = x_start_rand.to(self.device).unsqueeze(0)
            
        pred = self._predict_one_batch(x_start_rand, x_cond, num_steps, eta)

        pred_df = self._ms2_mesh_to_df(pred)
        pred_plot = pred_df.plot(
            x="y",
            y="x",
            z="intensity",
            title="Predicted MS2",
            kind=plot_type,
            xlabel="X Index",
            ylabel="Y Index",
            height=500,
            width=800,
            plot_3d=plot_3d,
            grid=False,
            show_plot=False,
            backend=backend,
        )

        ms2_mesh_df = self._ms2_mesh_to_df(ms2_1)
        ms2_target_plot = ms2_mesh_df.plot(
            x="y",
            y="x",
            z="intensity",
            title="Target MS2",
            kind=plot_type,
            xlabel="RT Index",
            ylabel="m/z Index",
            height=500,
            width=800,
            plot_3d=plot_3d,
            grid=False,
            show_plot=False,
            backend=backend,
        )

        ms2_input_mesh_df = self._ms2_mesh_to_df(x_start_rand.cpu().detach().numpy().squeeze(0))
        ms2_input_plot = ms2_input_mesh_df.plot(
            x="y",
            y="x",
            z="intensity",
            title="Random Noise MS2 Input",
            kind=plot_type,
            xlabel="RT Index",
            ylabel="m/z Index",
            height=500,
            width=800,
            plot_3d=plot_3d,
            grid=False,
            show_plot=False,
            backend=backend,
        )

        ms1_df = self._ms1_to_df(ms1_1.unsqueeze(0))
        ms1_plot = ms1_df.plot(
            kind="chromatogram",
            x="y",
            y="intensity",
            title="Query MS1",
            xlabel="RT Index",
            ylabel="Intensity",
            height=500,
            width=800,
            grid=False,
            show_plot=False,
            backend=backend,
        )

        return ms2_target_plot, ms1_plot, ms2_input_plot, pred_plot

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

    def _train_one_batch(self, x_start, x_cond, x_noise):
        """Train one batch"""
        self.optimizer.zero_grad()
        loss = self.train_step(x_start, x_cond, noise=x_noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def _predict_one_batch(self, x_start, x_cond, num_steps=100, eta=0.0):
        """Predict one batch"""
        self.model.eval()
        with torch.no_grad():
            sample = self.sample(x_start, x_cond, num_steps=num_steps, eta=eta)
        return sample[0].cpu().detach().numpy()

    @staticmethod            
    def _ms2_mesh_to_df(arr):
        """Convert MS2 mesh to DataFrame for 2D plotting"""
        rows, cols = arr.shape
        y, x = np.meshgrid(range(rows), range(cols), indexing='ij')
        x_flat = x.flatten()
        y_flat = y.flatten()
        intensity_flat = arr.flatten()

        return pd.DataFrame({
            'x': x_flat,
            'y': y_flat,
            'intensity': intensity_flat
        })

    @staticmethod
    def _ms2_to_df(batch_ms2):
        """Convert MS2 batch to DataFrame for 1D plotting"""
        ms2_df = pd.DataFrame()
        for i in range(batch_ms2.shape[0]):
            sample = batch_ms2[i]
            df = pd.DataFrame(sample.numpy())
            df['sample_id'] = i
            df = df.melt(id_vars=['sample_id'], var_name='x', value_name='intensity')
            df['y'] = df.index % 34
            ms2_df = pd.concat([ms2_df, df], ignore_index=True)
        return ms2_df

    @staticmethod
    def _ms1_to_df(batch_ms1):
        """Convert MS1 batch to DataFrame for 1D plotting"""
        ms1_df = pd.DataFrame()
        for i in range(batch_ms1.shape[0]):
            sample = batch_ms1[i]
            df = pd.DataFrame(sample.numpy().reshape(1, -1))
            df['sample_id'] = i
            df = df.melt(id_vars=['sample_id'], var_name='y', value_name='intensity')
            ms1_df = pd.concat([ms1_df, df], ignore_index=True)
        return ms1_df
    
    @staticmethod
    def _convert_mpl_fig_to_bytes(fig):
        """Convert a matplotlib figure to bytes"""
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf
