import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models.video import (
    r3d_18, R3D_18_Weights,
    s3d, S3D_Weights,
    mc3_18, MC3_18_Weights
)
from torch.utils.data import DataLoader
from src.model.base_model import BaseModel
from typing import Callable, List, Tuple, Dict, Union, Optional, Literal
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s.%(levelname)s: %(message)s")
logger = logging.getLogger("model.spine_cnn")


class LumbarSpineStenosisResNet(BaseModel):
    """Lumbar Spine Stenosis ResNet model."""
    num_levels: int = 5
    num_conditions: int = 5
    num_severities: int = 3
    num_total_classes: int = num_levels * num_conditions * num_severities
    single_channel_mean: List[float] = [sum([0.43216, 0.394666, 0.37645]) / 3]
    single_channel_std: List[float] = [sum([0.22803, 0.22145, 0.216989]) / 3]

    def __init__(
        self,
        architecture: Literal["R3D_18", "S3D", "MC3_18"] = "MC3_18",
        pretrained: bool = False,
        progress: bool = True,
        **kwargs
    ) -> None:
        super().__init__(name=kwargs.get("name", ""))
        assert isinstance(architecture, str), "architecture must be a string."
        _hidden_size = kwargs.get("hidden_size", 1024)
        _dropout_val = kwargs.get("dropout", kwargs.get("p", 0.5))
        _max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.architecture = architecture
        if architecture == "R3D_18":
            # Load the pre-trained 3D ResNet18 model.
            # See: https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html
            # Under: #torchvision.saved_models.video.R3D_18_Weights
            self.pre_trained_transforms = R3D_18_Weights.KINETICS400_V1.transforms(
                mean=self.single_channel_mean, std=self.single_channel_std
            )
            _weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
            self.model = r3d_18(weights=_weights, progress=progress)

            # Modify the first convolutional layer to accept 1 input channel instead of 3.
            # We do that by taking the average of the 3 input channels (ensemble).
            first_conv_layer: nn.Conv3d = self.model.stem._modules["0"]  # type: ignore
            first_conv_layer.weight = nn.Parameter(first_conv_layer.weight.mean(dim=1, keepdim=True))
            first_conv_layer.__dict__["in_channels"] = 1

            # Modify the fully connected layer.
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, _hidden_size),
                nn.ReLU(),
                nn.Dropout(p=_dropout_val),
                nn.Linear(_hidden_size, self.num_total_classes)
            )

        elif architecture == "S3D":
            raise NotImplementedError("S3D architecture is not supported due to mismatched input size.")
            # Load the pre-trained S3D model.
            # See: https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html
            # Under: #torchvision.saved_models.video.S3D_Weights
            self.pre_trained_transforms = S3D_Weights.KINETICS400_V1.transforms(
                mean=self.single_channel_mean, std=self.single_channel_std
            )
            _weights = S3D_Weights.KINETICS400_V1 if pretrained else None
            self.model = s3d(weights=_weights, progress=progress)

            # Modify the first convolutional layer to accept 1 input channel instead of 3.
            # We do that by taking the average of the 3 input channels (ensemble).
            first_conv_layer: nn.Conv3d = self.model.features[0][0][0]
            first_conv_layer.weight = nn.Parameter(first_conv_layer.weight.mean(dim=1, keepdim=True))
            first_conv_layer.__dict__["in_channels"] = 1

            # Modify the fully connected layer.
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=_dropout_val),
                nn.Conv3d(1024, self.num_total_classes, kernel_size=1, stride=1, bias=True)
            )
        elif architecture == "MC3_18":
            # Load the pre-trained MC3_18 model.
            # See: https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html
            # Under: #torchvision.saved_models.video.MC3_18_Weights
            self.pre_trained_transforms = MC3_18_Weights.KINETICS400_V1.transforms(
                mean=self.single_channel_mean, std=self.single_channel_std
            )

            _weights = MC3_18_Weights.KINETICS400_V1 if pretrained else None
            self.model = mc3_18(weights=_weights, progress=progress)

            # Modify the first convolutional layer to accept 1 input channel instead of 3.
            # We do that by taking the average of the 3 input channels (ensemble).
            first_conv_layer: nn.Conv3d = self.model.stem._modules["0"]  # type: ignore
            first_conv_layer.weight = nn.Parameter(first_conv_layer.weight.mean(dim=1, keepdim=True))
            first_conv_layer.__dict__["in_channels"] = 1

            # Modify the fully connected layer.
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, _hidden_size),
                nn.ReLU(),
                nn.Dropout(p=_dropout_val),
                nn.Linear(_hidden_size, self.num_total_classes)
            )

        else:
            raise NotImplementedError(f"architecture '{architecture}' is not supported.")

        self.log_softmax = nn.LogSoftmax(dim=2)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=_max_grad_norm)
        self._add_forward_transforms()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = x.view(-1, self.num_levels * self.num_conditions, self.num_severities)
        x = self.log_softmax(x)  # Apply softmax over the severity classes
        return x
    
    def _add_forward_transforms(self) -> None:
        if hasattr(self, "pre_trained_transforms"):
            def decorator(forward: Callable[[Tensor], Tensor]) -> Callable:
                def wrapper(x: Tensor) -> Tensor:
                    x = self.pre_trained_transforms(x)
                    return forward(x)
                return wrapper
            self.forward = decorator(self.forward)

    def _calc_running_metrics(self, x: Tensor, y_hat: Tensor, y: Tensor, loss: float,
                              running_loss: float, total_corrects: float) -> Tuple[float, float]:
        """Override the base class method to calculate the running metrics."""
        corrects = int((y_hat.argmax(-1) == y).sum().item())

        running_loss += loss
        total_corrects += corrects / self.num_levels / self.num_conditions
        return running_loss, total_corrects


class SingleModelSpineCNN(LumbarSpineStenosisResNet):
    """Single-Model Spine CNN model."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, name="do_not_save", **kwargs)
        _dropout_val = kwargs.get("dropout", kwargs.get("p", 0.5))
        _out_features_size = kwargs.get("out_features_size", 512)
        self.out_features_size = _out_features_size

        # Reinitialize the fully connected layer.
        if self.architecture == "R3D_18":
            self.model.fc = nn.Linear(self.model.fc[0].in_features, _out_features_size)
        elif self.architecture == "S3D":
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=_dropout_val),
                nn.Conv3d(1024, _out_features_size, kernel_size=1, stride=1, bias=True)
            )
        elif self.architecture == "MC3_18":
            self.model.fc = nn.Linear(self.model.fc[0].in_features, _out_features_size)
        else:
            raise RuntimeError("Architecture must be either 'R3D_18' or 'S3D' or 'MC3_18', this should be unreachable.")

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x).view(-1)  # Batch size must be 1.
        return x


class MultiModelSpineCNN(BaseModel):
    """Multi-Model Spine CNN model."""
    def __init__(self, *model_dicts, last_fc_dim: int = 1024, dropout: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        assert all(isinstance(model_dict, dict) for model_dict in model_dicts), \
            "All model arguments must be dictionaries for model initialization."
        self.models: List[SingleModelSpineCNN] = []
        for i, model_dict in enumerate(model_dicts):
            setattr(self, f"sub_model_{i}", SingleModelSpineCNN(**model_dict))
            self.models.append(getattr(self, f"sub_model_{i}"))
        self.fc = nn.Sequential(
            nn.Linear(sum([model_dict["out_features_size"] for model_dict in model_dicts]), last_fc_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(last_fc_dim, SingleModelSpineCNN.num_total_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[lr_scheduler.LRScheduler] = None
        self._device: Literal["cpu", "cuda"] = "cpu"

    def forward(self, data_dict: Dict[str, Union[List[Tensor], Tensor]]) -> Tensor:
        assert "data" in data_dict and "series_types" in data_dict, \
            "data_dict must contain 'data' and 'series_types' keys."
        assert len(data_dict["data"]) > 0 and len(data_dict["series_types"]) > 0, \
            "data and series_types must not be empty."
        assert len(data_dict["data"]) == len(data_dict["series_types"]), \
            "data and series_types must have the same length."

        # Assume series_type is encoded.
        out_features = [[] for _ in range(len(self.models))]
        for i, (data, series_type) in enumerate(zip(data_dict["data"], data_dict["series_types"])):
            out_features[series_type].append(self.models[series_type](data))

        # Handle case when there are no series of a certain type.
        for i in range(len(self.models)):
            if out_features[i] == []:
                out_features[i] = [torch.zeros(self.models[i].out_features_size, device=self._device)]

        # Handle multiple same series types.
        out_features = [torch.stack(out).mean(dim=0) for out in out_features]
        total_features = torch.cat(out_features, dim=0)

        y_hat = self.fc(total_features)
        y_hat = y_hat.view(SingleModelSpineCNN.num_levels * SingleModelSpineCNN.num_conditions,
                           SingleModelSpineCNN.num_severities)
        y_hat = self.log_softmax(y_hat)
        return y_hat.unsqueeze(0)  # Add a batch dimension.

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            num_epochs: int = 30, lr: float = 0.001, momentum: float = 0.9, wd: float = 0.,
            try_cuda: bool = True, verbose: bool = True, print_stride: int = 1) -> None:
        """Override function for training a multi-model spine 3D-CNN."""
        use_cuda = try_cuda and torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            self._device = "cuda"
            logger.info("Using CUDA for training.")
        else:
            self.cpu()
            self._device = "cpu"
            logger.info("Using CPU for training.")

        # Create the optimizer.
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        start_epoch = self.global_epoch
        for epoch in range(num_epochs):
            running_loss: float = 0.
            total_corrects: int = 0
            self.global_epoch += 1
            for mb, data_dict in enumerate(train_loader):
                data_dict: Dict[str, Union[List[Tensor], Tensor]]
                # Assume batch size is 1, remove the batch dimension.
                #data_dict["data"] = [x[0] for x in data_dict["data"]]
                data_dict["target"] = data_dict["target"][0]
                data_dict["series_types"] = data_dict["series_types"][0]
                if use_cuda:
                    data_dict["data"]: List[Tensor] = [x.cuda() for x in data_dict["data"]]
                    data_dict["target"]: Tensor = data_dict["target"].cuda()

                y_hat, loss = self.train_step(data_dict)

                running_loss, total_corrects = self.calc_running_metrics(
                    data_dict, y_hat, loss, running_loss, total_corrects)

                # TODO: Use tqdm instead of print.
                if verbose:
                    print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d} "
                          f"mb: {mb + 1:03d}/{len(train_loader):03d}  lr: {self.scheduler.get_last_lr()[0]:.6f}]  "
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
                logger.info(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d}"
                            f"  lr: {self.scheduler.get_last_lr()[0]:.6f}] "
                            f"[Train loss: {train_epoch_loss:.6f} "
                            f" Train acc: {100 * train_total_acc:.6f}%]  "
                            f"[Val loss: {val_epoch_loss:.6f} "
                            f" Val acc: {100 * val_total_acc:.6f}%]")
            self.scheduler.step()

        self._device = "cpu"
        self.cpu()

    def train_step(
        self, data_dict: Dict[str, Union[List[Tensor], Tensor]]
    ) -> Tuple[Tensor, float]:
        self.optimizer.zero_grad()
        y_hat = self(data_dict)
        loss = F.nll_loss(y_hat.view(-1, y_hat.size(-1)), data_dict["target"])
        loss.backward()
        self.optimizer.step()
        return y_hat, loss.item()

    def calc_metrics(self, data_loader: DataLoader, use_cuda: bool) -> Tuple[float, float]:
        """Override function for calculating the validation metrics."""
        running_loss: float = 0.
        total_corrects: int = 0
        self.eval()
        with torch.no_grad():
            for data_dict in tqdm(data_loader, desc="Validation", total=len(data_loader)):
                data_dict: Dict[str, Union[List[Tensor], Tensor]]
                # Assume batch size is 1, remove the batch dimension.
                # data_dict["data"] = [x[0] for x in data_dict["data"]]
                data_dict["target"] = data_dict["target"][0]
                data_dict["series_types"] = data_dict["series_types"][0]
                if use_cuda:
                    data_dict["data"]: List[Tensor] = [x.cuda() for x in data_dict["data"]]
                    data_dict["target"]: Tensor = data_dict["target"].cuda()

                y_hat: Tensor = self(data_dict)
                loss: float = F.nll_loss(
                    y_hat.view(-1, y_hat.size(-1)), data_dict["target"]
                ).item()

                running_loss, total_corrects = self.calc_running_metrics(
                    data_dict, y_hat, loss, running_loss, total_corrects)

        self.train()
        total_loss = running_loss / len(data_loader.dataset)  # type: ignore
        total_acc = total_corrects / len(data_loader.dataset)  # type: ignore
        return total_loss, total_acc

    def calc_running_metrics(
        self, data_dict: Dict[str, Union[List[Tensor], Tensor]], y_hat: Tensor, loss: float,
        running_loss: float, total_corrects: int
    ) -> Tuple[float, int]:
        """Override function for calculating the running metrics."""
        corrects = int((y_hat.argmax(-1) == data_dict["target"]).sum().item())
        running_loss += loss
        total_corrects += corrects / LumbarSpineStenosisResNet.num_levels / LumbarSpineStenosisResNet.num_conditions
        return running_loss, total_corrects
