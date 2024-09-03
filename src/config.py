"""
config.py

This file contains configuration settings for the project, such as paths and constants.
It also sets the root directory of the module to the project's root directory.

@Author: Ofir Paz
@Version: 14.08.2024
"""


# ================================== Imports ================================= #
import os
import sys
from pathlib import Path
# ============================== End Of Imports ============================== #


# ============================ Configuration Paths =========================== #
PROJECT_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = PROJECT_PATH / "src"
DATA_PATH = PROJECT_PATH / "data"
TRAIN_IMAGES_PATH = DATA_PATH / "train_images"
TEST_IMAGES_PATH = DATA_PATH / "test_images"
TRAIN_SERIES_DESCRIPTIONS_CSV_PATH = DATA_PATH / "train_series_descriptions.csv"
TRAIN_LABEL_COORDINATES_CSV_PATH = DATA_PATH / "train_label_coordinates.csv"
TRAIN_CSV_PATH = DATA_PATH / "train.csv"
TEST_CSV_PATH = DATA_PATH / "test_series_descriptions.csv"
SUBMISSION_CSV_PATH = DATA_PATH / "sample_submission.csv"
# ======================== End Of Configuration Paths ======================== #


# ================================= Constants ================================ #
EXAMPLE_ID: str = "4003253"
EXAMPLE_STIR_ID: str = "702807833"
EXAMPLE_SAGITTAL_T1_ID: str = "1054713880"
EXAMPLE_AXIAL_T2_ID: str = "2448190387"
EXAMPLE_SAGITTAL_T2_STIR_ID: str = "702807833"
THREEDIM_MRI_SHAPE = (30, 300, 300)

# ============================= End Of Constants ============================= #

# Set the root directory of the project.
sys.path.append(str(PROJECT_PATH))