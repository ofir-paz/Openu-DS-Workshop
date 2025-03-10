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


# ============================== Constant Paths ============================== #
PROJECT_PATH = Path(os.path.dirname(__file__)).parent
SRC_PATH = PROJECT_PATH / "src"
DATA_PATH = PROJECT_PATH / "data"
TRAIN_IMAGES_PATH = DATA_PATH / "train_images"
TEST_IMAGES_PATH = DATA_PATH / "test_images"
TRAIN_SERIES_DESCRIPTIONS_CSV_PATH = DATA_PATH / "train_series_descriptions.csv"
TRAIN_LABEL_COORDINATES_CSV_PATH = DATA_PATH / "train_label_coordinates.csv"
TRAIN_CSV_PATH = DATA_PATH / "train.csv"
TEST_SERIES_DESCRIPTIONS_CSV_PATH = DATA_PATH / "test_series_descriptions.csv"
SAMPLE_SUBMISSION_CSV_PATH = DATA_PATH / "sample_submission.csv"
# =========================== End Of Constant Paths ========================== #


# ============================ Configuration Paths =========================== #
MODELS_PATH = PROJECT_PATH / "saved_models"
SUBMISSION_PATH = PROJECT_PATH / "submissions"
# ======================== End Of Configuration Paths ======================== #


# =============================== Encoding Maps ============================== #
SERIES_S2I = {"Sagittal T1": 0, "Sagittal T2/STIR": 0, "Axial T2": 1}
SERIES_I2S = {0: "Sagittal", 1: "Axial"}
# =========================== End Of Encoding Maps =========================== #


# ================================= Constants ================================ #
EXAMPLE_ID: str = "4003253"
EXAMPLE_STIR_ID: str = "702807833"
EXAMPLE_SAGITTAL_T1_ID: str = "1054713880"
EXAMPLE_AXIAL_T2_ID: str = "2448190387"
EXAMPLE_SAGITTAL_T2_STIR_ID: str = "702807833"
THREEDIM_MRI_SHAPE = (30, 200, 200)

SCS_ID: str = "4646740"
SCS_Sagittal = TRAIN_IMAGES_PATH / SCS_ID / "3666319702" / "10.dcm"
SCS_AXIAL = TRAIN_IMAGES_PATH / SCS_ID / "3201256954" / "15.dcm"

NFA_ID: str = "7143189"
NFA_Sagittal = TRAIN_IMAGES_PATH / NFA_ID / "3219733239" / "12.dcm"

SS_ID: str = "8785691"
SS_AXIAL = TRAIN_IMAGES_PATH / SS_ID / "2406919186" / "3.dcm"

TEST_PATIENT_ID: str = "44036939"
TEST_PATIENT_SAGITTAL_T2: str = "3844393089" 
BEST_MODEL_PATH = MODELS_PATH / "multi_model_submission_e=8.pt"
BEST_SUBMISSION_PATH = SUBMISSION_PATH / "submission.csv"
LOGITS_OUTPUT_PATH = BEST_MODEL_PATH.with_suffix(".pt_logits.pkl")
# ============================= End Of Constants ============================= #

# Set the root directory of the project.
if str(PROJECT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_PATH))
