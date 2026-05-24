import os
import pandas as pd
import numpy as np

# Configurable mock parameters
NUM_SUBJECTS = 5
NUM_TRIALS_PER_SUBJECT = 2
SAMPLE_RATE = 9216
NUM_ELECTRODES = 70

# Relative to workspace root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("Generating mock dataset for testing...")

# 1. Generate demographic.csv
demographic_data = {
    "subject": list(range(1, NUM_SUBJECTS + 1)),
    " group": [0, 1, 0, 1, 0]  # Healthy (0) vs Schizophrenia (1)
}
df_demo = pd.DataFrame(demographic_data)
df_demo.to_csv(os.path.join(DATA_DIR, "demographic.csv"), index=False)
print("Saved demographic.csv")

# 2. Generate columnLabels.csv
electrode_names = [f"channel_{i}" for i in range(NUM_ELECTRODES)]
column_labels = ["index", "subject", " group", "trial"] + electrode_names
df_cols = pd.DataFrame(columns=column_labels)
df_cols.to_csv(os.path.join(DATA_DIR, "columnLabels.csv"), index=False)
print("Saved columnLabels.csv")

# 3. Generate subject CSVs (1.csv to 5.csv)
for subject in range(1, NUM_SUBJECTS + 1):
    rows_list = []
    # group label for this subject
    group_val = df_demo[df_demo.subject == subject][" group"].values[0]
    
    global_row_index = 0
    for trial in range(1, NUM_TRIALS_PER_SUBJECT + 1):
        # Generate random signal for each electrode channel
        # We make it have shape (9216, 70)
        signal = np.random.randn(SAMPLE_RATE, NUM_ELECTRODES).astype(np.float32)
        
        # Add a tiny signal shift if group_val == 1 (schizophrenia) so the model can learn something small
        if group_val == 1:
            signal += 0.1
            
        for i in range(SAMPLE_RATE):
            row = [global_row_index, subject, group_val, trial] + list(signal[i])
            rows_list.append(row)
            global_row_index += 1
            
    df_subj = pd.DataFrame(rows_list, columns=column_labels)
    # Save directly as subject_id.csv in the data directory
    df_subj.to_csv(os.path.join(DATA_DIR, f"{subject}.csv"), index=False, header=False)
    print(f"Saved mock subject data: {subject}.csv")

print("Mock dataset generation completed successfully!")
