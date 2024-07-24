import os
import random

import pandas as pd
import pydicom
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from src.config import (
    DATA_PATH,
    TRAIN_PATH,
    TEST_PATH
)

def create_condition_one_hot_mapping(df):
    conditions = df['condition'].unique()
    condition_to_one_hot = {condition: [1 if i == idx else 0 for i in range(len(conditions))]
                            for idx, condition in enumerate(conditions)}
    return condition_to_one_hot

class SpineDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.train = train

        if self.train:
            self.series_desc_df = pd.read_csv(DATA_PATH / "train_series_descriptions.csv")
            self.coords_df = pd.read_csv(DATA_PATH / "train_label_coordinates.csv")
            self.merged_df = pd.merge(self.coords_df, self.series_desc_df, on=['study_id', 'series_id'])
            self.condition_to_one_hot = create_condition_one_hot_mapping(self.merged_df)
        else:
            self.desc_df = pd.read_csv(DATA_PATH / "test_series_descriptions.csv")
            self.merged_df = self.desc_df
            self.condition_to_one_hot = None

    def __len__(self):
        return len(self.merged_df)

    def __getitem__(self, idx):
        row = self.merged_df.iloc[idx]

        study_id = row['study_id']
        instance_number = row['instance_number']


        if self.train:
            series_id = row['series_id']
            img_path = TRAIN_PATH / str(study_id) / str(series_id) / rf"{instance_number}.dcm"

            condition = row['condition']
            series_description = row['series_description']
            condition_one_hot = torch.tensor(self.condition_to_one_hot[condition], dtype=torch.float32)
        else:
            series_path = TEST_PATH / str(study_id)
            dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            if not dicom_files:
                raise FileNotFoundError(f"No DICOM files found in {series_path}")

            random_dicom_file = random.choice(dicom_files)
            img_path = series_path / random_dicom_file
            series_description = row['series_description']
            condition_one_hot = None  # No condition input for the test set

        dicom_image = pydicom.dcmread(img_path)
        image = dicom_image.pixel_array
        image = image.astype('float32')

        if self.transform:
            image = self.transform(image)

        # Ensure the image has 4 dimensions: [batch_size, channels, height, width]
        image = torch.tensor(image)

        return image, condition_one_hot, series_description
