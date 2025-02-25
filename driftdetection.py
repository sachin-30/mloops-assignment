import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import json
# Load old and new datasets
old_data = pd.read_csv(r"D:\MLOPS ASS=1\train.csv, index=False")
new_data = pd.read_csv(r"D:\MLOPS ASS=1\test.csv, index=False")
drift_alert = False
drift_results = {}

for column in old_data.columns:
    if old_data[column].dtype == "object":
        # Categorical data: Chi-Square Test
        old_counts = old_data[column].value_counts(normalize=True)
        new_counts = new_data[column].value_counts(normalize=True)
        min_index = old_counts.index.intersection(new_counts.index)
        if len(min_index) > 1:
            chi2, p_val, _, _ = chi2_contingency([old_counts[min_index], new_counts[min_index]])
            drift_detected = p_val < 0.05
        else:
            drift_detected = True  # Assume drift if missing categories
    else:
        # Numerical data: KS Test
        ks_stat, p_val = ks_2samp(old_data[column], new_data[column])
        drift_detected = p_val < 0.05
    drift_results[column] = {"p_value": p_val, "drift_detected": drift_detected}
    if drift_detected:
        drift_alert = True
print("ks_stat",ks_stat,"\np_val", p_val)
# Save results to log file
with open("drift_alert.log", "w") as f:
    f.write("Data drift detected!\n" if drift_alert else "No data drift detected.\n")
print("✅ Data drift check complete.")
