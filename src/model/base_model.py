"""
base_model.py - Base model module.

Contains the base model class.

@Author: Ofir Paz
@Version: 17.07.2024
"""

# ================================== Imports ================================= #
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics import Accuracy
from torch import Tensor
from torch.nn.modules.loss import _Loss as BaseNNLossModule
from typing import Literal, Tuple, Union, Optional
# ============================== End Of Imports ============================== #


# ============================== BaseModel Class ============================= #
class BaseModel(nn.Module):
    """Base model class."""

    def __init__(self) -> None:
        """Constructor."""
        super(BaseModel, self).__init__()
        self.best_weights: Optional[dict[str, Tensor]] = None
        self.global_epoch: int = 0
        
        self.train_costs: list[float] = []
        self.val_costs: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []

        self.criterion: BaseNNLossModule = nn.CrossEntropyLoss()   

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

                y_hat, loss = self.__train_step(x, y, optimizer)

                running_loss, total_corrects = self._calc_running_matrics(
                    x, y_hat, y, loss, running_loss, total_corrects)

                # TODO: Use tqdm instead of print.
                if verbose:
                    print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d} "
                          f"mb: {mb + 1:03d}/{len(train_loader):03d}]  " 
                          f"[Train loss: {loss:.6f}]\033[J", end="")

            train_epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore
            train_total_acc = total_corrects / len(train_loader.dataset)  # type: ignore
            self.train_costs.append(train_epoch_loss); self.train_accs.append(train_total_acc)
            
            if val_loader is not None:
                val_epoch_loss, val_total_acc = self.calc_metrics(val_loader, use_cuda)
            else:
                val_epoch_loss, val_total_acc = torch.nan, torch.nan

            self.val_costs.append(val_epoch_loss); self.val_accs.append(val_total_acc)

            self.save_best_weights(val_total_acc)

            if verbose and (epoch % print_stride == 0 or epoch == num_epochs - 1):
                print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d}] "
                      f"[Train loss: {train_epoch_loss:.6f} "
                      f" Train acc: {100 * train_total_acc:.6f}%]  "
                      f"[Val loss: {val_epoch_loss:.6f} "
                      f" Val acc: {100 * val_total_acc:.6f}%]")
        self.cpu()

    def __train_step(self, x: Tensor, y: Tensor, optimizer: optim.Optimizer) -> Tuple[Tensor, float]:
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

        loss: Tensor = self.criterion(y_hat, y)
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
            for x, y in data_loader:
                if use_cuda:
                    x: Tensor = x.cuda(); y: Tensor = y.cuda()

                y_hat: Tensor = self(x)
                loss: float = self.criterion(y_hat, y).item()

                running_loss, total_corrects = self._calc_running_matrics(
                    x, y_hat, y, loss, running_loss, total_corrects)

        self.train()
        total_loss = running_loss / len(data_loader.dataset)  # type: ignore
        total_acc = total_corrects / len(data_loader.dataset)  # type: ignore
        return total_loss, total_acc

    def _calc_running_matrics(self, x: Tensor, y_hat: Tensor, y: Tensor, loss: float,
                              running_loss: float, total_corrects: int) -> Tuple[float, int]:
        corrects = int((y_hat.argmax(1) == y).sum().item())

        running_loss += loss * x.size(0)
        total_corrects += corrects
        return running_loss, total_corrects

    def save_best_weights(self, score: float) -> None:
        """Saves the best weights of the model."""
        if self.best_weights is None:
            self.best_weights = copy.deepcopy(self.state_dict())
        elif score > max(self.val_accs):
            self.best_weights = copy.deepcopy(self.state_dict())

    def load_best_weights(self):
        """Loads the best weights of the model."""
        assert self.best_weights is not None, "No weights to load."
        self.load_state_dict(self.best_weights)

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

# ========================== End Of BaseModel Class ========================== #