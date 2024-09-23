import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.video import r3d_18, R3D_18_Weights
from src.model.base_model import BaseModel
from typing import Callable, Optional


class LumbarSpineStenosisResNet(BaseModel):
    """Lumbar Spine Stenosis ResNet model."""
    num_levels: int = 5
    num_conditions: int = 5
    num_severities: int = 3
    num_total_classes: int = num_levels * num_conditions * num_severities

    def __init__(self, pretrained: bool = False, progress: bool = True, **kwargs) -> None:
        super(LumbarSpineStenosisResNet, self).__init__(**kwargs)
        _hidden_size = kwargs.get("hidden_size", 1024)
        _dropout_val = kwargs.get("dropout", kwargs.get("p", 0.5))
        if pretrained:
            # Load the pre-trained 3D ResNet model.
            # See: https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html
            # Under: #torchvision.models.video.R3D_18_Weights
            self.pre_trained_transformrs = R3D_18_Weights.KINETICS400_V1.transforms
            self.resnet3d = r3d_18(pretrained=R3D_18_Weights.KINETICS400_V1, progress=progress)
        else:
            # Create a new 3D ResNet model and ignore the progress bar.
            self.resnet3d = r3d_18(pretrained=False, progress=False)
        
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

        self._add_forward_transforms()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet3d(x)
        x = x.view(-1, self.num_levels * self.num_conditions, self.num_severities)
        x = F.softmax(x, dim=2)  # Apply softmax over the severity classes
        return x.view(-1, self.num_total_classes)
    
    def _add_forward_transforms(self) -> None:
        if hasattr(self, "pre_trained_transformrs"):
            def decorator(forward: Callable[[Tensor], Tensor]) -> Callable:
                def wrapper(x: Tensor) -> Tensor:
                    x = self.pre_trained_transformrs(x)
                    return forward(x)
                return wrapper
            self.forward = decorator(self.forward)
