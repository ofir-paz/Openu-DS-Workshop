from torchvision import transforms
from SpineCNN import *
from SpineDataset import *

image_width = 100
image_height = 200
train_batch = 100


def train_model(model, dataloader, criterion, optimizer, num_epochs, num_conditions):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, conditions, descriptions in dataloader:
            optimizer.zero_grad()

            # Handle conditions being None for test dataset by using a placeholder tensor
            if conditions is None:
                conditions = torch.zeros(images.size(0), num_conditions)

            outputs = model(images, conditions)
            loss = criterion(outputs, descriptions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_width, image_height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Add your own path
    train_dataset = SpineDataset(
        'M:\\Users\\netan\\Desktop\\rsna-2024-lumbar-spine-degenerative-classification', True, transform)
    test_dataset = SpineDataset(
        'M:\\Users\\netan\\Desktop\\rsna-2024-lumbar-spine-degenerative-classification', False, transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_dataset), num_workers=2)

    num_conditions = len(train_dataset.condition_to_one_hot)
    num_series_descriptions = 3  # Set this to the actual number of series descriptions
    model = SpineCNN(img_width=256, img_height=256, num_conditions=num_conditions,
                     num_series_descriptions=num_series_descriptions)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_dataloader, criterion, optimizer, 1, num_conditions)


if __name__ == '__main__':
    main()

