import os
import pandas as pd
import numpy as np
import pydicom
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Callable


def load_batches(
    base_path: str,
    excel_file_path: str,
    k: int,
    process_batches_func: Callable,
    num_workers: int = 4,
    percent: float = 1.0
) -> None:
    """
    Loads DICOM files in batches and processes them using a separate thread.
    """

    if not (0 < percent <= 1):
        raise ValueError("Percent must be in the range (0,1].")

    df = pd.read_csv(excel_file_path, dtype={'series_id': str})

    all_dicom_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(base_path)
        for file in files if file.endswith(".dcm")
    ]
    
    total_files = len(all_dicom_files)
    num_files_to_load = int(total_files * percent)
    selected_files = set(np.random.choice(all_dicom_files, num_files_to_load, replace=False))
    
    file_progress = tqdm(total=num_files_to_load, desc="Processing DICOM Files", unit="file")

    batch_queue = queue.Queue()
    stop_event = threading.Event()

    def load_images_from_study(study_path: str, df: pd.DataFrame):
        batch_data = {}
        for series_id in os.listdir(study_path):
            series_path = os.path.join(study_path, series_id)
            if os.path.isdir(series_path):
                series_id_str = str(series_id)
                series_df = df[df['series_id'] == series_id_str]
                if series_df.empty:
                    continue
            
                condition = series_df.iloc[0]['condition']
                images = []
                for file in os.listdir(series_path):
                    if file.endswith(".dcm"):
                        dicom_path = os.path.join(series_path, file)
                        if dicom_path not in selected_files:
                            continue

                        try:
                            dicom_data = pydicom.dcmread(dicom_path)
                            img_data = dicom_data.pixel_array
                            images.append(img_data)
                        except Exception as e:
                            print(f"Error reading {dicom_path}: {e}")

                        file_progress.update(1)

                if images:
                    if condition not in batch_data:
                        batch_data[condition] = []
                    batch_data[condition].extend(images)

        return batch_data

    def batch_processing_thread() -> None:
        while not stop_event.is_set() or not batch_queue.empty():
            try:
                batch_data = batch_queue.get(timeout=1)
                if batch_data:
                    process_batches_func(batch_data)
                batch_queue.task_done()
            except queue.Empty:
                continue

    processing_thread = threading.Thread(target=batch_processing_thread, daemon=True)
    processing_thread.start()

    study_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    batch_start_idx = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while batch_start_idx < len(study_folders):
            batch_end_idx = min(batch_start_idx + k, len(study_folders))
            selected_study_folders = study_folders[batch_start_idx:batch_end_idx]
            
            futures = {executor.submit(load_images_from_study, study_folder, df): study_folder for study_folder in selected_study_folders}
            
            batch_data_list = []
            for future in as_completed(futures):
                batch_data = future.result()
                if batch_data:
                    batch_data_list.append(batch_data)

            if batch_data_list:
                batch_queue.put(batch_data_list)

            batch_start_idx = batch_end_idx

    batch_queue.join()
    stop_event.set()
    processing_thread.join()

    file_progress.close()
