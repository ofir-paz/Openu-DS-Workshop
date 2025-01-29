import os
import pydicom
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class DICOMDataset:
    def __init__(self, base_dir, csv_path):
        self.base_dir = base_dir
        self.csv_data = pd.read_csv(csv_path, engine='python')  # Load CSV file
        self.series_mapping = self._create_series_mapping()
        self.data_list = self._gather_dicom_files()
    
    def _create_series_mapping(self):
        """Creates a mapping from series_id to condition from the CSV file."""
        mapping = {}
        for _, row in self.csv_data.iterrows():
            series_id = str(row['series_id'])
            condition = row['condition']
            mapping[series_id] = condition
        return mapping
    
    def _gather_dicom_files(self):
        """Recursively finds DICOM files matching the series_id in the CSV file."""
        data_list = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".dcm"):
                    series_id = file.split(".")[0]  # Extract series_id from filename
                    if series_id in self.series_mapping:
                        condition = self.series_mapping[series_id]
                        dicom_path = os.path.join(root, file)
                        data_list.append((condition, dicom_path))
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        condition, dicom_file = self.data_list[index]
        if not os.path.exists(dicom_file):
            return condition, None  # Skip missing files
        dicom_data = pydicom.dcmread(dicom_file)
        pixel_array = dicom_data.pixel_array.astype(np.float32)
        return condition, pixel_array

def get_dataloader(base_dir, csv_path, batch_size=4):
    dataset = DICOMDataset(base_dir, csv_path)
    
    for start_idx in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start_idx, min(start_idx + batch_size, len(dataset)))]
        batch = [(condition, img) for condition, img in batch if img is not None]  # Remove missing images
        if batch:
            yield batch
