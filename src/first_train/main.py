import os
from pathlib import Path
from torchvision import transforms
from SpineCNN import *
from SpineDataset import *

image_width = 100
image_height = 200
train_batch = 10
project_path = Path(os.path.dirname(__file__)).parent


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_height, image_width)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SpineDataset(project_path / "data", True, transform)
    #test_dataset = SpineDataset(project_path / "data", False, transform)

    train_dataloader = DataLoader(dataset, batch_size=train_batch, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=2)

    num_conditions = len(dataset.condition_to_one_hot)
    num_series_descriptions = 3  # Set this to the actual number of series descriptions
    model = SpineCNN(img_width=image_width, img_height=image_height, num_conditions=num_conditions,
                     num_series_descriptions=num_series_descriptions)
    model.fit(train_dataloader)


if __name__ == '__main__':
    main()

