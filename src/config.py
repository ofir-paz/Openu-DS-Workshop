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
TRAIN_PATH = DATA_PATH / "train"
TEST_PATH = DATA_PATH / "test"
# ======================== End Of Configuration Paths ======================== #