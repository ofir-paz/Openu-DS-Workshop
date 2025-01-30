from src.dataset import image_dataloader 
from src.dataset.image_dataloader import load_batches
import src.config as cfg

k = 10  # Number of batches to load at once
num_workers = 4  # Number of workers for parallel processing

def process_batches(batch_data):
    # Custom operation on all batches
    print(f"Processed {len(batch_data)} batches")
    if batch_data:
        first_condition = next(iter(batch_data[0]))  # Get first condition
        first_image = batch_data[0][first_condition][0][0]  # Get first image
        print("Sample pixel data:")
        print(first_image)

load_batches(cfg.TRAIN_IMAGES_PATH, cfg.TRAIN_LABEL_COORDINATES_CSV_PATH, k, process_batches, num_workers)