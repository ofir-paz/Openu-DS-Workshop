"""
config.py

This module contains configuration settings for the application.
"""


# ================================== Imports ================================= #
import os
from pathlib import Path
# ============================== End Of Imports ============================== #


# ============================ Configuration Paths =========================== #
PROJECT_PATH = str(Path(os.path.dirname(__file__)).parent)
SRC_PATH = PROJECT_PATH + "\\src"
DATA_PATH = PROJECT_PATH + "\\data"
TRAIN_PATH = DATA_PATH + "\\train.csv"
TEST_PATH = DATA_PATH + "\\test.csv"
# ======================== End Of Configuration Paths ======================== #