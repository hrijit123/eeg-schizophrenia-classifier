import os
import torch

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Path to Kaggle raw datasets (where user should place them)
    RAW_DATA_DIR = os.path.join(DATA_DIR, "button-tone-sz")
    
    # Caching path for preprocessed NumPy files
    PREPROCESSED_FILE = os.path.join(DATA_DIR, "preprocessed_eeg.npz")
    
    # Checkpoints
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    # --- Preprocessing ---
    N_AVERAGED = 16
    TOTAL_SUBJECTS = 81
    SAMPLE_RATE_PER_TRIAL = 9216
    
    # Standard EEG electrode labels for this dataset (from columnLabels.csv)
    # The first 4 columns are usually: index, subject, group, trial. The rest are electrodes.
    ELECTRODES_COUNT = 70  # Standard count in the button-tone-sz dataset
    
    # --- Model Training ---
    BATCH_SIZE = 256
    EPOCHS = 100  # Default to 100 for a faster standard run, user can adjust
    LEARNING_RATE = 1e-4
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
