"""
config.py

This module contains configuration settings for the application.
"""


# ================================== Imports ================================= #
import os
from pathlib import Path
# ============================== End Of Imports ============================== #


# ============================ Configuration Paths =========================== #
PROJECT_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = PROJECT_PATH / "src"
DATA_PATH = PROJECT_PATH / "data"
TRAIN_IMAGES_PATH = DATA_PATH / "train_images"
TEST_IMAGES_PATH = DATA_PATH / "test_images"
TRAIN_CSV_PATH = DATA_PATH / "train.csv"
TEST_CSV_PATH = DATA_PATH / "test.csv"
# ======================== End Of Configuration Paths ======================== #


# ================================= Constants ================================ #
EXAMPLE_ID: str = "4003253"
EXAMPLE_SAGITTAL_T1_ID: str = "1054713880"
EXAMPLE_AXIAL_T2_ID: str = "2448190387"
EXAMPLE_STIR_ID: str = "702807833"
# ============================= End Of Constants ============================= #
