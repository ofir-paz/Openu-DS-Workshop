"""
loaders.py

This file contains functions for loading data.

@Author: Ofir Paz
@Version: 14.08.2024
"""


# ================================== Imports ================================= #
import os
from pathlib import Path
from typing import Union
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
# ============================== End Of Imports ============================== #


# ============================= Loaders Functions ============================ #
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
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(pydicom.dcmread(os.path.join(root, file)))
    dicom_files.sort(key=lambda x: int(x.InstanceNumber))
    
    # Stack all slices into a 3D numpy array
    image_3d = np.stack([apply_voi_lut(df.pixel_array, df) for df in dicom_files])
    return image_3d
# ========================== End Of Loaders Functions ========================= #
