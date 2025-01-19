import torch
from torchvision import transforms


def ComposeSingle(func) -> transforms.Compose:
    if not callable(func):
        raise ValueError("The input must be a callable function.")
    return transforms.Compose([func])


@ComposeSingle
def add_gaussian_noise(image: torch.Tensor) -> torch.Tensor:
    std = torch.min(torch.std(image, dim=(1, 2, 3)) * 0.05, torch.tensor(50))
    noise = torch.randn(image.size(), dtype=torch.float32) * std.view(-1, 1, 1, 1)
    noisy_tensor = image + noise
    return torch.clamp(noisy_tensor, image.min(), image.max())  # Ensure values are within the original range