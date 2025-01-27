import os
import pandas as pd

# Base directory for your training runs
base_dir = "runs/detect"

# List of directories to include (train0, train1, ..., train13)
train_dirs = [os.path.join(base_dir, f"train{i}") for i in range(7,14)]

# List to store dataframes
dfs = []

# Iterate through each directory
for train_dir in train_dirs:
    results_path = os.path.join(train_dir, "results.csv")
    if os.path.exists(results_path):
        # Read the CSV
        df = pd.read_csv(results_path)
        
        # Add a column to track the training directory
        df["train_run"] = os.path.basename(train_dir)
        
        # Append to the list
        dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save to a new CSV file
combined_csv_path = os.path.join(base_dir, "combined_results.csv")
combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined results saved to: {combined_csv_path}")
