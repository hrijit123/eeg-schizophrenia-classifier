import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import Config

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG trials."""
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (N, 1) for BCEWithLogitsLoss
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


def average_rows(matrix, n):
    """
    Averages every n rows in a 2D matrix.
    Shape: (time_steps, channels) -> (time_steps // n, channels)
    """
    shape = matrix.shape
    assert len(shape) == 2, "Matrix must be 2D"
    assert shape[0] % n == 0, f"Number of rows ({shape[0]}) must be divisible by {n}"
    
    reshaped = matrix.reshape(shape[0] // n, n, shape[1])
    return reshaped.mean(axis=1)


def find_file(directory, pattern):
    """Finds files matching pattern in a directory (case insensitive)."""
    search_pattern = os.path.join(directory, "**", pattern)
    files = glob.glob(search_pattern, recursive=True)
    return files[0] if files else None


def load_raw_data():
    """
    Loads raw Kaggle button-tone-sz dataset, performs downsampling and normalization,
    and returns X, Y arrays.
    """
    Config.setup_directories()
    
    # 1. Load demographics to map subject to group (0 = Healthy, 1 = Schizophrenia)
    demographic_path = find_file(Config.DATA_DIR, "demographic.csv")
    if not demographic_path:
        raise FileNotFoundError(
            f"demographic.csv not found in {Config.DATA_DIR}. "
            "Please download the dataset from Kaggle and place it in the data folder."
        )
    
    demographic = pd.read_csv(demographic_path)
    # The original dataset has columns with leading spaces: " group"
    group_col = [c for c in demographic.columns if "group" in c.lower()][0]
    subject_col = [c for c in demographic.columns if "subject" in c.lower()][0]
    
    diagnosis_dict = dict(zip(demographic[subject_col], demographic[group_col]))
    
    # 2. Get Electrode columns
    labels_path = find_file(Config.DATA_DIR, "columnLabels.csv")
    if not labels_path:
        raise FileNotFoundError(f"columnLabels.csv not found in {Config.DATA_DIR}.")
    
    column_list = pd.read_csv(labels_path).columns
    # Electrodes are all columns starting from index 4
    electrodes_list = list(column_list[4:])
    num_electrodes = len(electrodes_list)
    print(f"Detected {num_electrodes} electrodes: {electrodes_list[:5]}... {electrodes_list[-5:]}")
    
    # 3. Process each subject
    # Allocate maximum potential size: 81 subjects * ~100 trials = 8100 trials max
    # Feature size per trial = (9216 // N_AVERAGED) * num_electrodes
    features_per_trial = (Config.SAMPLE_RATE_PER_TRIAL // Config.N_AVERAGED) * num_electrodes
    X_raw = np.zeros((Config.TOTAL_SUBJECTS * 100, features_per_trial), dtype=np.float32)
    Y_raw = np.zeros(len(X_raw), dtype=np.float32)
    
    x_counter = 0
    print("Processing subjects and trials...")
    
    # Search for files like '1.csv' or nested '1.csv/1.csv'
    for person_number in tqdm(range(1, Config.TOTAL_SUBJECTS + 1)):
        # Try to locate the csv file for this subject
        csv_file_name = f"{person_number}.csv"
        csv_path = find_file(Config.DATA_DIR, csv_file_name)
        
        if not csv_path:
            # Skip if file is missing (helps with testing/partial dataset)
            print(f"Warning: CSV file for subject {person_number} not found. Skipping...")
            continue
            
        df = pd.read_csv(csv_path, header=None, names=column_list)
        trials_list = set(df.trial)
        
        for trial_number in trials_list:
            trial_df = df[df.trial == trial_number]
            # Verify if trial has the standard length
            if len(trial_df) == Config.SAMPLE_RATE_PER_TRIAL:
                current_sample_matrix = trial_df[electrodes_list].values
                # Downsample (average rows)
                averaged = average_rows(current_sample_matrix, n=Config.N_AVERAGED)
                # Flatten to 1D feature vector
                averaged_flat = averaged.reshape(-1)
                
                X_raw[x_counter] = averaged_flat.astype(np.float32)
                Y_raw[x_counter] = diagnosis_dict[person_number]
                x_counter += 1
                
    print(f"Total valid trials processed: {x_counter}")
    if x_counter == 0:
        raise ValueError("No valid trials were processed. Please check your data directory.")
        
    X = X_raw[:x_counter]
    Y = Y_raw[:x_counter]
    
    # 4. Normalization (max norm per electrode column across the whole dataset)
    print("Normalizing dataset...")
    # Reshape to (N * downsampled_timesteps, num_electrodes) to normalize per channel
    timesteps_downsampled = Config.SAMPLE_RATE_PER_TRIAL // Config.N_AVERAGED
    X_reshaped = X.reshape(-1, num_electrodes)
    # Normalize each column (electrode channel) to have max norm 1
    X_norm = normalize(X_reshaped, axis=0, norm='max')
    # Reshape back to the flat feature format
    X_norm = X_norm.reshape(X.shape)
    
    return X_norm, Y


def get_data_loaders(force_preprocess=False):
    """
    Preprocesses data if not cached, otherwise loads from cache.
    Returns PyTorch DataLoaders for training and testing.
    """
    Config.setup_directories()
    
    if force_preprocess or not os.path.exists(Config.PREPROCESSED_FILE):
        print("Preprocessed cache not found or force_preprocess=True. Processing raw data...")
        X, Y = load_raw_data()
        np.savez_compressed(Config.PREPROCESSED_FILE, X=X, Y=Y)
        print(f"Saved preprocessed data to {Config.PREPROCESSED_FILE}")
    else:
        print(f"Loading preprocessed data from cache: {Config.PREPROCESSED_FILE}")
        data = np.load(Config.PREPROCESSED_FILE)
        X, Y = data['X'], data['Y']
        print(f"Loaded dataset of shape: X={X.shape}, Y={Y.shape}")
        
    # Split train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=Config.TEST_SIZE, shuffle=True, random_state=Config.RANDOM_STATE
    )
    
    # Create Datasets
    train_dataset = EEGDataset(X_train, Y_train)
    test_dataset = EEGDataset(X_test, Y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader
