# Dataset Downloader Instructions

To run this project with the actual clinical dataset, you must download the EEG Schizophrenia button-press dataset from Kaggle.

## Steps to Setup the Data

1. **Download the Dataset:**
   - Go to [Kaggle: EEG Schizophrenia Dataset (Button-Tone SZ)](https://www.kaggle.com/datasets/vibolmo/button-tone-sz)
   - Download the zip file containing all subjects and CSVs.

2. **Extract Files into the `data/` Directory:**
   Extract the archive contents directly into this `data/` folder so it has the following layout:
   ```
   eeg-schizophrenia-classifier/
   └── data/
       ├── demographic.csv
       ├── columnLabels.csv
       ├── 1.csv
       ├── 2.csv
       ...
       ├── 81.csv
       └── README.md (This file)
   ```

   *Note: In some zip extracts, the subject files may be structured inside subdirectories (e.g. `1.csv/1.csv`). The code in `src/dataset.py` is robust and will search recursively for all `.csv` files matching the subject number, so both flat and nested directory structures are supported.*

## Dataset Details
- **Subjects:** 81 participants (consisting of both healthy control subjects and subjects diagnosed with Schizophrenia).
- **Format:** Each participant has a CSV file containing their EEG time-series recording across 70 electrode channels.
- **Trial Length:** Every standard trial contains exactly 9216 rows (sampled timepoints).
- **Target:** Classify each trial's signal representation into Schizophrenia (1) vs. Healthy Controls (0) using the target mapping in `demographic.csv`.
