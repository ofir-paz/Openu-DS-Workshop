"""
base_model.py - Base model module.

Contains the base model class.

@Author: Ofir Paz
@Version: 17.07.2024
"""

# ================================== Imports ================================= #
import os
import glob
import re
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import Tensor
from torch.nn.modules.loss import _Loss as BaseNNLossModule
from typing import Tuple, Optional, Union
from src.config import MODELS_PATH
from tqdm import tqdm
import logging
# ============================== End Of Imports ============================== #

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s.%(levelname)s: %(message)s")
logger = logging.getLogger("model.base_model")

# ============================== BaseModel Class ============================= #
class BaseModel(nn.Module):
    """Base model class."""
    def __init__(self, name: str = "") -> None:
        """Constructor."""
        if not isinstance(name, str):
            raise TypeError("Model name must be a string.")
        elif name == "":
            raise ValueError("Model name must be provided.")
        elif "=" in name:
            raise ValueError("Model name cannot contain the '=' character.")
        self.make_model_dir(name, num_tries=5)
        super().__init__()
        self.best_weights: Optional[dict[str, Tensor]] = None
        self.global_epoch: int = 0
        
        self.train_costs: list[float] = []
        self.val_costs: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []

        self.criterion: BaseNNLossModule = nn.NLLLoss()
        self.name: str = name

    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass."""
        raise NotImplementedError

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
            num_epochs: int = 30, lr: float = 0.001, momentum: float = 0.9, wd: float = 0.,
            try_cuda: bool = True, verbose: bool = True, print_stride: int = 1) -> None:
        """
        Base function for training a model.

        Args:
            train_loader (DataLoader) - The dataloader to fit the model to.
            val_loader (DataLoader) - The dataloader to validate the model on.
            num_epochs (int) - Number of epochs.
            lr (float) - Learning rate.
            momentum (float) - Momentum for SGD.
            wd (float) - Weight decay.
            try_cuda (bool) - Try to use CUDA.
            verbose (bool) - Verbose flag.
            print_stride (int) - Print stride (in epochs).
        """
        use_cuda = try_cuda and torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            print("Using CUDA for training.")
        else:
            self.cpu()
            print("Using CPU for training.")

        # Create the optimizer.
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        start_epoch = self.global_epoch
        for epoch in range(num_epochs):
            running_loss: float = 0.
            total_corrects: int = 0
            self.global_epoch += 1
            for mb, (x, y) in enumerate(train_loader):
                if use_cuda:
                    x: Tensor = x.cuda(); y: Tensor = y.cuda()

                y_hat, loss = self._train_step(x, y, optimizer)

                running_loss, total_corrects = self._calc_running_metrics(
                    x, y_hat, y, loss, running_loss, total_corrects)

                # TODO: Use tqdm instead of print.
                if verbose:
                    print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d} "
                          f"mb: {mb + 1:03d}/{len(train_loader):03d}]  " 
                          f"[Train loss: {loss:.6f}]\033[J", end="")

            train_epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore
            train_total_acc = total_corrects / len(train_loader.dataset)  # type: ignore
            self.train_costs.append(train_epoch_loss)
            self.train_accs.append(train_total_acc)
            
            if val_loader is not None:
                val_epoch_loss, val_total_acc = self.calc_metrics(val_loader, use_cuda)
            else:
                val_epoch_loss, val_total_acc = torch.nan, torch.nan

            self.val_costs.append(val_epoch_loss)
            self.val_accs.append(val_total_acc)

            self.save_best_weights()

            if verbose and (epoch % print_stride == 0 or epoch == num_epochs - 1):
                print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d}] "
                      f"[Train loss: {train_epoch_loss:.6f} "
                      f" Train acc: {100 * train_total_acc:.6f}%]  "
                      f"[Val loss: {val_epoch_loss:.6f} "
                      f" Val acc: {100 * val_total_acc:.6f}%]")
        self.cpu()

    def _train_step(self, x: Tensor, y: Tensor, optimizer: optim.Optimizer) -> Tuple[Tensor, float]:
        """
        Performs a single training step.

        Args:
            x (Tensor) - Input tensor.
            y (Tensor) - Target tensor.
            optimizer (Optimizer) - Optimizer.
        
        Returns:
            Tuple[Tensor, float] - The model's output and the loss of the model.
        """
        # zero the parameter gradients.
        optimizer.zero_grad()

        # forward + backward + optimize.
        y_hat: Tensor = self(x)

        loss: Tensor = self.criterion(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        lloss = loss.item()

        return y_hat, lloss

    def calc_metrics(self, data_loader: DataLoader, use_cuda: bool) -> Tuple[float, float]:
        """
        Calculates and returns the loss and the accuracy of the model on a given dataset.

        Args:
            data_loader (DataLoader) - Data loader.
            use_cuda (bool) - Use CUDA flag.

        Returns:
            Tuple[float, float] - The loss and score of the model on the given dataset.            
        """
        running_loss: float = 0.
        total_corrects: int = 0
        self.eval()
        with torch.no_grad():
            for x, y in tqdm(data_loader, desc="Validation", total=len(data_loader)):
                if use_cuda:
                    x: Tensor = x.cuda(); y: Tensor = y.cuda()

                y_hat: Tensor = self(x)
                loss: float = self.criterion(y_hat.view(-1, y_hat.size(-1)), y.view(-1)).item()

                running_loss, total_corrects = self._calc_running_metrics(
                    x, y_hat, y, loss, running_loss, total_corrects)

        self.train()
        total_loss = running_loss / len(data_loader.dataset)  # type: ignore
        total_acc = total_corrects / len(data_loader.dataset)  # type: ignore
        return total_loss, total_acc

    def _calc_running_metrics(self, x: Tensor, y_hat: Tensor, y: Tensor, loss: float,
                              running_loss: float, total_corrects: int) -> Tuple[float, int]:
        corrects = int((y_hat.argmax(-1) == y).sum().item())

        running_loss += loss * x.size(0)
        total_corrects += corrects
        return running_loss, total_corrects

    def save_best_weights(self) -> None:
        """Saves the best weights of the model."""
        if self.best_weights is None or self.val_accs[-1] > max(self.val_accs[:-1]):
            self.best_weights = copy.deepcopy(self.state_dict())
            self._cleanup_old_models()
            logger.info(f"Saving model weights at epoch: {self.global_epoch}")
            torch.save(self.best_weights, self.model_dir / f"{self.name}_e={self.global_epoch}.pt")

    def load_best_weights(self):
        """Loads the best weights of the model."""
        assert self.best_weights is not None, "No weights to load."
        self.load_state_dict(self.best_weights)

    def _cleanup_old_models(self):
        """Removes old model files, keeping only the last 5."""
        model_files = glob.glob(str(self.model_dir / f"{self.name}_e=*.pt"))
        if len(model_files) > 5:
            # Extract integer values from filenames
            def extract_epoch(filename: str) -> Union[int, float]:
                match = re.search(r"_e=(\d+)\.pt", filename)
                return int(match.group(1)) if match else float('inf')

            # Sort files by extracted integer values
            model_files.sort(key=extract_epoch)

            # Delete old files, keeping only the last 5
            for old_model in model_files[:-5]:
                logger.warning(f"Too many model checkpoints, Removing old model: {old_model}")
                os.remove(old_model)

    def get_outputs(self, data_loader: DataLoader, try_cuda: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Calculates and returns the outputs of the model on a given dataset.

        Args:
            data_loader (DataLoader) - Data loader.
            try_cuda (bool) - Try to use CUDA flag.

        Returns:
            Tuple[Tensor, Tensor] - The outputs (class, probability) of the model on the given dataset.
        """
        use_cuda = try_cuda and torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        self.eval()
        outputs: list[Tensor] = []
        with torch.no_grad():
            for x, _ in data_loader:
                if use_cuda:
                    x: Tensor = x.cuda()
                y_hat: Tensor = self(x)
                outputs.append(y_hat)
        self.train()
        logits, preds = torch.cat(outputs).max()
        probs = torch.softmax(logits, dim=1)

        return preds, probs
    
    @classmethod
    def load_model(cls, name: str, epoch: Union[int, str]) -> "BaseModel":
        """
        Loads a model from a file.

        Args:
            name (str) - The model's name.
            epoch (Union[int, str]) - The epoch to load.

        Returns:
            BaseModel - The loaded model.
        """
        model = cls(name)
        model.load_state_dict(torch.load(MODELS_PATH / f"{name}_ckpts" / f"{name}_e={epoch}.pt"))
        return model

    def make_model_dir(self, name: str, num_tries: int = 3) -> None:
        """Creates the model directory."""
        model_dir = MODELS_PATH / f"{name}_ckpts"
        if not model_dir.exists():
            for try_num in range(num_tries + 1):
                if try_num == num_tries:
                    logger.error(f"Failed to create model directory: {model_dir}")
                    raise Exception(f"Failed to create model directory: {model_dir}")
                try:
                    model_dir.mkdir()
                    logger.info(f"Created model directory: {model_dir}")
                    break
                except Exception as e:
                    logger.error(f"[Try: {try_num+1}/{num_tries+1}] - Failed to create model directory: {model_dir}."
                                 f" {e}")
        else:
            logger.warning(f"Model directory already exists: {model_dir}."
                           f"\nMake sure you are not overwriting an existing model.")

        self.model_dir: Path = model_dir
# ========================== End Of BaseModel Class ========================== #
