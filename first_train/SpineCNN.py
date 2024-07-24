from base_model import *


class SpineCNN(nn.Module):
    def __init__(self, img_width, img_height, num_conditions, num_series_descriptions):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the size of the feature map after the convolution and pooling layers
        out_dim1 = self._calculate_output_dim(img_width, img_height, 3, 1, 1)
        out_dim1 = self._calculate_output_dim(out_dim1[0], out_dim1[1], 2, 2, 0)
        out_dim2 = self._calculate_output_dim(out_dim1[0], out_dim1[1], 3, 1, 1)
        out_dim2 = self._calculate_output_dim(out_dim2[0], out_dim2[1], 2, 2, 0)

        # Define the number of features after flattening the feature map
        self.num_features = out_dim2[0] * out_dim2[1] * 64

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.num_features, 128)
        self.condition_fc = nn.Linear(num_conditions, 64)
        self.series_fc = nn.Linear(128 + 64, num_series_descriptions)

    def _calculate_output_dim(self, width, height, kernel_size, stride, padding):
        new_width = ((width - kernel_size + 2 * padding) // stride) + 1
        new_height = ((height - kernel_size + 2 * padding) // stride) + 1
        return new_width, new_height

    def forward(self, x, condition):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_features)
        x = F.relu(self.fc1(x))

        condition = F.relu(self.condition_fc(condition))

        combined = torch.cat((x, condition), dim=1)
        series_output = self.series_fc(combined)

        return series_output