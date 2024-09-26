import sys
import os
from pathlib import Path


if __name__ == "__main__":
    # Add the root directory to the Python path
    PROJECT_PATH = Path(os.path.dirname(__file__)).parent
    sys.path.append(str(PROJECT_PATH))

    print(f"Added {PROJECT_PATH} to the Python path.")

    from torch.utils.data import DataLoader
    from src.model.spine_cnn import LumbarSpineStenosisResNet
    from src.dataset.spine_dataset import LumbarSpineDataset

    dataset = LumbarSpineDataset(True)
    train_dataset, val_dataset = dataset.split(val_size=0.95)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = LumbarSpineStenosisResNet(pretrained=False, progress=True, hidden_size=1024, dropout=0.2, name="test")
    model.fit(train_loader=train_dataloader, val_loader=train_dataloader, num_epochs=3,
              lr=0.005, momentum=0.9, wd=0., try_cuda=True, verbose=True, print_stride=1)
    model.fit(train_loader=train_dataloader, val_loader=train_dataloader, num_epochs=2,
              lr=0.0001, momentum=0.9, wd=0., try_cuda=True, verbose=True, print_stride=1)
    #model.fit(train_loader=train_dataloader, val_loader=train_dataloader, num_epochs=2,
    #          lr=0.0001, momentum=0.9, wd=0.0001, try_cuda=True, verbose=True, print_stride=1)
