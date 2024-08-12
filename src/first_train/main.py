import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.first_train.spine_cnn import SpineCNN
from src.first_train.spine_dataset import SpineDataset

image_width = 300
image_height = 300
train_batch = 1
depth = 10

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.cuda()),
        transforms.Resize((image_height, image_width)),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = SpineDataset(image_height=image_height, image_width=image_width, transform=transform, depth=depth)
    #test_dataset = SpineDataset(project_path / "data", False, transform)

    train_dataloader = DataLoader(dataset, batch_size=train_batch, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=2)

    model = SpineCNN(image_height, image_width, depth)
    model.fit(train_loader=train_dataloader, num_epochs=1)


if __name__ == '__main__':
    main()
