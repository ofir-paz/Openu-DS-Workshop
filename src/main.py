import config  # For setting the root directory of the project.

from torch.utils.data import DataLoader
from src.model.spine_cnn import LumbarSpineStenosisResNet
from src.dataset.spine_dataset import LumbarSpineDataset


def main() -> None:
    dataset = LumbarSpineDataset(True)
    train_dataset, val_dataset = dataset.split(val_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=3)
    val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=3)

    model = LumbarSpineStenosisResNet(pretrained=True, progress=True, hidden_size=1024, dropout=0.2)
    model.fit(train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=3,
              lr=0.1, momentum=0.9, wd=0.01, try_cuda=True, verbose=True, print_stride=1)


if __name__ == '__main__':
    main()
