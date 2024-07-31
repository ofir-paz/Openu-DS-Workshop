import os
import random

import pandas as pd
import pydicom
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from config import (
    DATA_PATH,
    TRAIN_PATH,
    TEST_PATH
)

class SpineDataset(Dataset):
    def __init__(self, image_height: int, image_width:int, depth = 32, transform=None,):
        
        self.transform = transform
        self.depth = depth
        
        self.image_height = image_height
        self.image_width = image_width
        self.df = pd.read_csv(TRAIN_PATH)
        self.condition_columns = [
            'spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4',
            'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2',
            'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4',
            'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1',
            'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3',
            'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5',
            'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2',
            'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4',
            'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1',
            'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3',
            'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5',
            'right_subarticular_stenosis_l5_s1'
        ]
        self.condition_to_one_hot = self.create_condition_one_hot_mapping()

    def create_condition_one_hot_mapping(self):
        unique_conditions = sorted(self.df[self.condition_columns].stack().unique())
        condition_to_one_hot = {cond: i for i, cond in enumerate(unique_conditions)}
        return condition_to_one_hot

    def pad_or_crop_image(self, image):
        # Ensure the image has 3 dimensions (depth, height, width)
        if len(image.shape) != 3:
            raise ValueError(f"Expected image with 3 dimensions, but got shape {image.shape}")

        # Pad or crop depth
        if image.shape[0] < self.depth:
            padding = (self.depth - image.shape[0]) // 2
            pad_width = ((padding, self.depth - image.shape[0] - padding), (0, 0), (0, 0))
            image = np.pad(image, pad_width, mode='constant')
        elif image.shape[0] > self.depth:
            start = (image.shape[0] - self.depth) // 2
            image = image[start:start + self.depth, :, :]

        # Pad or crop height
        if image.shape[1] < self.image_height:
            padding = (self.image_height - image.shape[1]) // 2
            pad_width = ((0, 0), (padding, self.image_height - image.shape[1] - padding), (0, 0))
            image = np.pad(image, pad_width, mode='constant')
        elif image.shape[1] > self.image_height:
            start = (image.shape[1] - self.image_height) // 2
            image = image[:, start:start + self.image_height, :]

        # Pad or crop width
        if image.shape[2] < self.image_width:
            padding = (self.image_width - image.shape[2]) // 2
            pad_width = ((0, 0), (0, 0), (padding, self.image_width - image.shape[2] - padding))
            image = np.pad(image, pad_width, mode='constant')
        elif image.shape[2] > self.image_width:
            start = (image.shape[2] - self.image_width) // 2
            image = image[:, :, start:start + self.image_width]

        return image
    
    def __len__(self):
        return len(self.df)

    def get_all_dicom_files(self, directory):
        dicom_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        return dicom_files


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        study_id = row['study_id']
        study_path = DATA_PATH + "\\train_images\\" + str(study_id)

        dicom_files = self.get_all_dicom_files(study_path)
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {study_path}")

        images = []
        for dicom_file in dicom_files:
            dicom_image = pydicom.dcmread(dicom_file)
            image = dicom_image.pixel_array
            image = image.astype('float32')

            if self.transform:
                image = self.transform(image)

            images.append(image)

        # Convert list of images to a numpy array and then to tensor
        images = np.array(images).squeeze()
        images = self.pad_or_crop_image(images)
        images = torch.tensor(images)
        images = images.unsqueeze(0) # Only 1 channel mri is grayscale
        
         # Create a one-hot encoded tensor for conditions
        condition_tensor = torch.zeros(len(self.condition_columns), 3, dtype=torch.float32)
        severity_mapping = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
        
        for i, col in enumerate(self.condition_columns):
            condition = row[col]
            if pd.isna(condition):
                condition_tensor[i, :] = -1  # Use -1 or another value to indicate missing data
            else:
                condition_tensor[i, severity_mapping[condition]] = 1
        
        return images, condition_tensor
