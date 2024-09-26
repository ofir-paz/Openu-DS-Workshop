import os
from typing import Union, Optional, List, Dict
from pathlib import Path
from math import floor, ceil

import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from src.config import (
    DATA_PATH,
    TRAIN_IMAGES_PATH,
    TEST_IMAGES_PATH,
    TRAIN_CSV_PATH,
)


def add_channel_dim(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1)


class LumbarSpineDataset(Dataset):
    conditions_i2s: List[str] = [
        "spinal_canal_stenosis_l1_l2", "spinal_canal_stenosis_l2_l3", "spinal_canal_stenosis_l3_l4",
        "spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1", "left_neural_foraminal_narrowing_l1_l2",
        "left_neural_foraminal_narrowing_l2_l3", "left_neural_foraminal_narrowing_l3_l4",
        "left_neural_foraminal_narrowing_l4_l5", "left_neural_foraminal_narrowing_l5_s1",
        "right_neural_foraminal_narrowing_l1_l2", "right_neural_foraminal_narrowing_l2_l3",
        "right_neural_foraminal_narrowing_l3_l4", "right_neural_foraminal_narrowing_l4_l5",
        "right_neural_foraminal_narrowing_l5_s1", "left_subarticular_stenosis_l1_l2",
        "left_subarticular_stenosis_l2_l3", "left_subarticular_stenosis_l3_l4",
        "left_subarticular_stenosis_l4_l5", "left_subarticular_stenosis_l5_s1",
        "right_subarticular_stenosis_l1_l2", "right_subarticular_stenosis_l2_l3",
        "right_subarticular_stenosis_l3_l4", "right_subarticular_stenosis_l4_l5",
        "right_subarticular_stenosis_l5_s1"
    ]  # Acts as enumeration for the conditions.
    severity_i2s: List[str] = ["Normal/Mild", "Moderate", "Severe"]  # Acts as enumeration for the severity.

    conditions_s2i: Dict[str, int] = {condition: i for i, condition in enumerate(conditions_i2s)}
    severity_s2i: Dict[Union[str, float], int] = {severity: i for i, severity in enumerate(severity_i2s)}
    nan_class = severity_s2i["Normal/Mild"]  # Will be 0.

    default_preprocess = transforms.Compose([
        # Accepts a numpy array of dimensions (D, H, W).
        torch.from_numpy,
        transforms.ConvertImageDtype(torch.float32),
        # Add a channel dimension, (D, 1, H, W).
        add_channel_dim,
    ])

    def __init__(
        self,
        train: bool, 
        *,
        preprocess: Optional[transforms.Compose] = None,
        augs: Optional[transforms.Compose] = None
    ) -> None:
        self.train = train
        self.preprocess: transforms.Compose = preprocess or self.default_preprocess
        self.augs = augs
        self._apply_augs = augs is not None
        self.series_description = pd.read_csv(
            DATA_PATH / f"{'train' if train else 'test'}_series_descriptions.csv"
        )
        self.images_path: Path = TRAIN_IMAGES_PATH if train else TEST_IMAGES_PATH
        if train:
            self.train_csv = pd.read_csv(TRAIN_CSV_PATH, index_col="study_id")
        else:
            raise NotImplementedError("Test dataset is not implemented yet.")
    
    def __len__(self) -> int:
        return self.series_description.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """returns x (D, 1, H, W), y (75)"""
        desc_row = self.series_description.iloc[idx]
        series_path = self.images_path / str(desc_row["study_id"]) / str(desc_row["series_id"])        
        dicom_files = load_dicom_series(series_path)

        x: torch.Tensor = self.preprocess(dicom_files)  # type: ignore - torch.from_numpy
        if self.train:
            if self._apply_augs and self.augs is not None:
                x = self.augs(x)
            condition_row = self.train_csv.loc[desc_row["study_id"]]
            y = torch.tensor(
                [self.severity_s2i.get(condition_row[col], self.nan_class) for col in self.conditions_i2s],
                dtype=torch.int64
            )
        else:
            y = None
        
        return x, y
    
    def split(self, val_size: float = 0.2) -> tuple["LumbarSpineDataset", "LumbarSpineDataset"]:
        return random_split(self, [1 - val_size, val_size])  # type: ignore


def load_dicom_series(directory: Union[str, Path]) -> np.ndarray:
    """
    Load a DICOM series from a directory into a 3D numpy array.

    Args:
        directory (Union[str, Path]): The directory containing the DICOM series.

    Returns:
        np.ndarray: The 3D numpy array containing the DICOM series.
    """
    dicom_files = []
    for root, _, files in os.walk(directory):
        # TODO: Raise an exception if no files are found.
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(pydicom.dcmread(os.path.join(root, file)))
    dicom_files.sort(key=lambda x: int(x.InstanceNumber))
    
    # Stack all slices into a 3D numpy array.
    image_3d = [apply_voi_lut(dcmf.pixel_array, dcmf) for dcmf in dicom_files]
    
    # In rare cases, sometimes the pixel arrays are not of the same shape.
    if (shapes := np.unique([np.array(img.shape) for img in image_3d], axis=0)).shape[0] > 1:
        # Calculate the square padding for each image.
        max_h, max_w = np.max(shapes, axis=0)
        padding_dict = {
            (h, w): (
                (floor((max_h - h) / 2), ceil((max_h - h) / 2)),
                (floor((max_w - w) / 2), ceil((max_w - w) / 2))
            )
            for (h, w) in shapes
        }
        stacked_images = np.stack([
            np.pad(img, padding_dict[tuple(img.shape)], mode="constant", constant_values=0)
            for img in image_3d
        ])
    else:
        stacked_images = np.stack(image_3d)
    
    return stacked_images
