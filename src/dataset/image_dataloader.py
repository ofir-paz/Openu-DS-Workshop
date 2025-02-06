import os
import pandas as pd
import numpy as np
import pydicom
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def load_batches(base_path, excel_file_path, k, process_batches_func, num_workers=4, percent=1.0):
    """
    Loads DICOM files in batches and processes them using a separate thread.
    
    Args:
        base_path (str): Path to the DICOM dataset directory.
        excel_file_path (str): Path to the CSV file containing series IDs and conditions.
        k (int): Number of batches to load at a time.
        process_batches_func (function): Function to process each batch.
        num_workers (int): Number of worker threads for loading.
        percent (float): Fraction of total files to load, range (0,1].
    """

    if not (0 < percent <= 1):
        raise ValueError("Percent must be in the range (0,1].")

    # Load the CSV data
    df = pd.read_csv(excel_file_path, dtype={'series_id': str})  # Ensure series_id is treated as string

    # Count total number of DICOM files
    all_dicom_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(base_path)
        for file in files if file.endswith(".dcm")
    ]
    
    total_files = len(all_dicom_files)
    num_files_to_load = int(total_files * percent)  # Limit files based on percentage
    selected_files = set(np.random.choice(all_dicom_files, num_files_to_load, replace=False))  # Random subset
    
    # Create a tqdm progress bar for file processing
    file_progress = tqdm(total=num_files_to_load, desc="Processing DICOM Files", unit="file")

    # Create a queue for batch processing
    batch_queue = queue.Queue()
    stop_event = threading.Event()

    # Function to load images from each study_id folder (batch)
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
                        if dicom_path not in selected_files:
                            continue  # Skip files not in the selected subset

                        try:
                            dicom_data = pydicom.dcmread(dicom_path)
                            img_data = dicom_data.pixel_array  # Convert DICOM pixel data to NumPy array
                            images.append(img_data)
                        except Exception as e:
                            print(f"Error reading {dicom_path}: {e}")

                        file_progress.update(1)  # Update the tqdm progress bar

                if images:  # Ensure we have at least one valid image
                    if condition not in batch_data:
                        batch_data[condition] = []
                    batch_data[condition].extend(images)

        return batch_data

    # Function to process batches in a separate thread
    def batch_processing_thread():
        while not stop_event.is_set() or not batch_queue.empty():
            try:
                batch_data = batch_queue.get(timeout=1)  # Wait for new batch data
                if batch_data:
                    process_batches_func(batch_data)
                batch_queue.task_done()
            except queue.Empty:
                continue  # Keep checking

    # Start the batch processing thread
    processing_thread = threading.Thread(target=batch_processing_thread, daemon=True)
    processing_thread.start()

    # Load all study_id folders
    study_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    batch_start_idx = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while batch_start_idx < len(study_folders):
            batch_end_idx = min(batch_start_idx + k, len(study_folders))
            selected_study_folders = study_folders[batch_start_idx:batch_end_idx]
            
            futures = {executor.submit(load_images_from_study, study_folder, df): study_folder for study_folder in selected_study_folders}
            
            # Wait for all futures to complete
            batch_data_list = []
            for future in as_completed(futures):
                batch_data = future.result()
                if batch_data:  # Only store non-empty batches
                    batch_data_list.append(batch_data)

            if batch_data_list:
                batch_queue.put(batch_data_list)  # Send batch data to the processing thread

            # Prepare for the next batch
            batch_start_idx = batch_end_idx

    # Stop the processing thread once all batches are enqueued
    batch_queue.join()  # Ensure all batches are processed
    stop_event.set()
    processing_thread.join()  # Wait for the processing thread to finish

    file_progress.close()
