import torch
from torchvision import transforms


def ComposeSingle(func) -> transforms.Compose:
    if not callable(func):
        raise ValueError("The input must be a callable function.")
    return transforms.Compose([func])


@ComposeSingle
def add_gaussian_noise(image: torch.Tensor) -> torch.Tensor:
    var = image.max() * 0.05 + 1e-7  # TODO: Maybe add variance in each depth slice.
    noise = torch.randn(image.size(), dtype=torch.float32) * torch.sqrt(var)
    noisy_tensor = image + noise
    return torch.clamp(noisy_tensor, image.min(), image.max())  # Ensure values are within the original range