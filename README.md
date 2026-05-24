# EEG Schizophrenia Classifier in PyTorch

A professional, modular, and optimized deep learning pipeline in PyTorch to classify Schizophrenia vs. Healthy Controls using EEG (Electroencephalography) recordings from the clinical **Button-Tone SZ** dataset. 

This project transforms a monolithic Kaggle notebook into a production-grade, Git-ready Python codebase with optimized preprocessing, caching, multiple convolutional neural network (CNN) architectures, and advanced evaluation reporting.

---

## 🧠 Project Background & EEG Analysis

Schizophrenia is a complex psychiatric disorder characterized by disruptions in thought processes, perceptions, and emotional responsiveness. Electroencephalography (EEG) recordings capture the electrical activity of the brain, offering a non-invasive window into neural dynamics. 

During the clinical trial, subjects participated in a button-press tone-listening task (measuring Event-Related Potentials, or ERPs). This codebase downsamples and extracts spatial-temporal signal signatures from 70 electrode channels to classify each EEG trial:
- **0 - Healthy Controls**
- **1 - Schizophrenia Patient**

---

## 🚀 Key Features

* **PyTorch Refactoring:** Rewritten entirely in PyTorch, replacing the original Keras code.
* **Optimized Preprocessing & Caching:** Downsamples EEG signals (averaging time points to reduce shape from 9216 to 576 per trial) and normalizes electrodes across trials. It then caches the results to compressed NumPy (`.npz`) format, allowing subsequent training runs to start **instantly**.
* **Three Model Architectures:**
  1. **`original` (CNN-2D):** Translates the original Keras 2D Conv network, treating the EEG trial as a spatial-temporal image grid `(1, 70 electrodes, 576 timesteps)`.
  2. **`eeg1d` (CNN-1D):** A 1D CNN that treats electrodes as input channels and convolves over the time dimension, which is more native to time-series sequences.
  3. **`eegnet` (EEGNet):** A compact convolutional network architecture designed specifically for EEG-based Brain-Computer Interfaces (BCIs) using depthwise separable convolutions.
* **Advanced Metrics:** Computes loss, accuracy, confusion matrix, precision, recall (sensitivity), specificity, and F1-score.
* **Command Line Interface (CLI):** Seamless entry point `main.py` to run different pipeline steps.
* **Mock Testing:** Includes a helper script to generate synthetic EEG data so you can test the entire pipeline without downloading the full dataset first.

---

## 📁 Repository Structure

```
eeg-schizophrenia-classifier/
├── data/                      # Data directory (contents ignored by git)
│   ├── demographic.csv        # Class mappings (Healthy vs SZ)
│   ├── columnLabels.csv       # Header columns & electrode labels
│   └── README.md              # Instructions to download Kaggle dataset
├── src/                       # Core python package
│   ├── __init__.py
│   ├── config.py              # Central configurations and hyperparameters
│   ├── dataset.py             # Preprocessing, normalization, PyTorch DataLoaders
│   ├── model.py               # PyTorch architectures (CNN-2D, CNN-1D, EEGNet)
│   ├── train.py               # Training epochs, checkpoint saves, logging
│   └── evaluate.py            # Evaluation reports, sensitivity, specificity, and CM
├── reports/                   # Saved performance curves and Heatmaps
│   ├── training_history_*.png
│   └── confusion_matrix_*.png
├── checkpoints/               # Trained PyTorch weight checkpoints (ignored by git)
├── main.py                    # Main CLI entrypoint
├── requirements.txt           # Python library requirements
├── .gitignore                 # Predefined git ignores
└── README.md                  # Project documentation (this file)
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <your-github-repo-url>
   cd eeg-schizophrenia-classifier
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8+ installed. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the Dataset:**
   - Follow the instructions in [data/README.md](file:///c:/Users/hriji/OneDrive/Documents/Desktop/New%20folder/data/README.md) to download the dataset from Kaggle and place it in the `data/` folder.

---

## 🧪 Quick Start & Verification (Mock Test)

You can verify that the code compiles and runs end-to-end on your machine by generating a small, synthetic mock dataset:

1. **Generate Mock EEG Files:**
   ```bash
   python scratch/generate_mock_data.py
   ```

2. **Run Pipeline (Preprocess, Train, and Evaluate):**
   ```bash
   python main.py --mode all --model original --epochs 3 --lr 0.001 --force-preprocess
   ```

3. **Test Other Architectures:**
   - Train a 1D CNN:
     ```bash
     python main.py --mode train --model eeg1d --epochs 5
     ```
   - Train EEGNet:
     ```bash
     python main.py --mode train --model eegnet --epochs 5
     ```

---

## ⚙️ CLI Usage Reference

The CLI tool `main.py` orchestrates the training and evaluation:

```bash
python main.py [flags]
```

### Flags
* `--mode`: The pipeline phase to run. Options are:
  - `preprocess`: Read subject CSVs, downsample, normalize, and cache as `.npz` in `data/`.
  - `train`: Train the specified model.
  - `evaluate`: Load the best saved weights and output classification metrics on the test split.
  - `all` (default): Run all three steps sequentially.
* `--model`: Network architecture. Options are `original` (CNN-2D), `eeg1d`, or `eegnet` (default: `original`).
* `--epochs`: Number of training epochs (overrides config).
* `--lr`: Training learning rate (overrides config).
* `--force-preprocess`: Set this flag to force recalculation of cached data.

---

## 📊 Results and Visualizations

After running in `train` or `all` modes:
- **Training Curves:** Loss and accuracy history plots are saved in `reports/training_history_<model_name>.png`.
- **Confusion Matrix:** Heatmaps depicting classification performance are saved in `reports/confusion_matrix_<model_name>.png`.
