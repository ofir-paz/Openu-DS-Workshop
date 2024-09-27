import config  # For setting the root directory of the project.

from torch.utils.data import DataLoader
from src.model.spine_cnn import LumbarSpineStenosisResNet
from src.dataset.spine_dataset import LumbarSpineDataset


def main() -> None:
    dataset = LumbarSpineDataset(True)
    train_dataset, val_dataset = dataset.split(val_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=8)

    model = LumbarSpineStenosisResNet(pretrained=True, progress=True, hidden_size=2048, dropout=0.4,
                                      name="lumbar_spine_cnn_v1")
    # If you want to use SGD instead of Adam, you need to manually set the optimizer in base_model.py.
    # momentum parameter is only used in SGD.
    model.fit(train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=20,
              lr=0.0025, momentum=0.9, wd=0.001, try_cuda=True, verbose=True, print_stride=1)
    model.fit(train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=15,
              lr=0.0005, momentum=0.9, wd=0.001, try_cuda=True, verbose=True, print_stride=1)
    model.fit(train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=10,
              lr=0.0001, momentum=0.9, wd=0.001, try_cuda=True, verbose=True, print_stride=1)


if __name__ == '__main__':
    main()
