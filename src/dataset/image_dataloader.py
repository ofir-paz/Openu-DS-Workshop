import os
import pandas as pd
import numpy as np
import pydicom
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def load_batches(base_path, excel_file_path, k, process_batches_func, num_workers=4):
    # Load the CSV data
    df = pd.read_csv(excel_file_path, dtype={'series_id': str})  # Ensure series_id is treated as string

    # Function to load images for each study_id folder (batch)
    def load_images_from_study(study_path, df):
        batch_data = {}
        for series_id in os.listdir(study_path):
            series_path = os.path.join(study_path, series_id)
            if os.path.isdir(series_path):
                series_id_str = str(series_id)  # Ensure consistent format
                series_df = df[df['series_id'] == series_id_str]
                if series_df.empty:
                    continue  # Skip this folder
                
                condition = series_df.iloc[0]['condition']  # Copy condition from matching series_id
                images = []
                for file in os.listdir(series_path):
                    if file.endswith(".dcm"):  # Ensure we're working with DICOM files
                        dicom_path = os.path.join(series_path, file)
                        try:
                            dicom_data = pydicom.dcmread(dicom_path)
                            img_data = dicom_data.pixel_array  # Convert DICOM pixel data to NumPy array
                            images.append(img_data)
                        except Exception as e:
                            print(f"Error reading {dicom_path}: {e}")
                
                if len(images) > 0:
                    if condition not in batch_data:
                        batch_data[condition] = []
                    batch_data[condition].append(np.array(images))
        return batch_data

    # Load all study_id folders
    study_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    batch_start_idx = 0
    all_batches_data = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while batch_start_idx < len(study_folders):
            batch_end_idx = min(batch_start_idx + k, len(study_folders))
            selected_study_folders = study_folders[batch_start_idx:batch_end_idx]
            
            futures = {executor.submit(load_images_from_study, study_folder, df): study_folder for study_folder in selected_study_folders}
            
            # Track progress with tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading batches"):
                batch_data = future.result()
                if batch_data:  # Only store non-empty batches
                    all_batches_data.append(batch_data)

            # After loading all batches, process them
            process_batches_func(all_batches_data)

            # Prepare for the next batch
            batch_start_idx = batch_end_idx
            all_batches_data = []  # Clear batch data for the next set of batches
