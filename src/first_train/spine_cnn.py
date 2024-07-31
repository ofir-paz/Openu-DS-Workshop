"""
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from base_model import BaseModel


class SpineCNN(BaseModel):
    num_conditions = 25

    def __init__(self, image_height: int, image_width: int, depth=32):
        super(SpineCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=depth, kernel_size=(3, 3, 3), padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv3d(in_channels=depth, out_channels=depth*2, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(in_channels=depth*2, out_channels=depth, kernel_size=(3, 3, 3), padding=1)
        
        # Calculate the size of the feature map after the conv and pooling layers
        def conv3d_output_size(size, kernel_size=3, padding=1, stride=1):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        def pool3d_output_size(size, kernel_size=2, stride=2):
            return (size - kernel_size) // stride + 1
        
        conv_output_depth = pool3d_output_size(conv3d_output_size(pool3d_output_size(conv3d_output_size(pool3d_output_size(depth)))))
        conv_output_height = pool3d_output_size(conv3d_output_size(pool3d_output_size(conv3d_output_size(pool3d_output_size(image_height)))))
        conv_output_width = pool3d_output_size(conv3d_output_size(pool3d_output_size(conv3d_output_size(pool3d_output_size(image_width)))))
        
        self.fc1 = nn.Linear(conv_output_depth * conv_output_height * conv_output_width * depth, self.num_conditions * 3)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(self.num_conditions * 3)

    def forward(self, x: Tensor):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = x.view(-1, self.num_conditions, 3)
        return x
    