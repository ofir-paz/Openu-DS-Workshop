import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.video import r3d_18, R3D_18_Weights
from src.model.base_model import BaseModel
from typing import Callable, List, Tuple


class LumbarSpineStenosisResNet(BaseModel):
    """Lumbar Spine Stenosis ResNet model."""
    num_levels: int = 5
    num_conditions: int = 5
    num_severities: int = 3
    num_total_classes: int = num_levels * num_conditions * num_severities
    single_channel_mean: List[float] = [sum([0.43216, 0.394666, 0.37645]) / 3]
    single_channel_std: List[float] = [sum([0.22803, 0.22145, 0.216989]) / 3]

    def __init__(self, pretrained: bool = False, progress: bool = True, **kwargs) -> None:
        super().__init__(name=kwargs.get("name", ""))
        _hidden_size = kwargs.get("hidden_size", 1024)
        _dropout_val = kwargs.get("dropout", kwargs.get("p", 0.5))
        if pretrained:
            # Load the pre-trained 3D ResNet model.
            # See: https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html
            # Under: #torchvision.saved_models.video.R3D_18_Weights
            self.pre_trained_transforms = R3D_18_Weights.KINETICS400_V1.transforms(
                mean=self.single_channel_mean, std=self.single_channel_std
            )
            self.resnet3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1, progress=progress)
        else:
            # TODO: Change the mean and std to the correct values.
            self.pre_trained_transforms = R3D_18_Weights.KINETICS400_V1.transforms(
                mean=self.single_channel_mean, std=self.single_channel_std
            )
            # Create a new 3D ResNet model and ignore the progress bar.
            self.resnet3d = r3d_18(weights=None, progress=False)
        
        # Modify the first convolutional layer to accept 1 input channel instead of 3.
        # We do that by taking the average of the 3 input channels (ensemble).
        first_conv_layer: nn.Conv3d = self.resnet3d.stem._modules["0"]  # type: ignore
        first_conv_layer.weight = nn.Parameter(first_conv_layer.weight.mean(dim=1, keepdim=True))
        
        # Modify the fully connected layer.
        self.resnet3d.fc = nn.Sequential(
            nn.Linear(self.resnet3d.fc.in_features, _hidden_size),
            nn.ReLU(),
            nn.Dropout(p=_dropout_val),
            nn.Linear(_hidden_size, self.num_total_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=2)

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self._add_forward_transforms()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet3d(x)
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
