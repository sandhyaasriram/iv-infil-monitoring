"""
Inspector for the PLOS ONE Clinical IV Infiltration Dataset
Paper: "Detection of intravenous infiltration using impedance parameters in patients in a long-term care hospital"
Authors: Jeong et al. (PLOS ONE 2019, DOI: 10.1371/journal.pone.0213585)
Supplementary File: S1 File (pone.0213585.s001.xlsx)
"""
import os
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_sheet(sheet_name):
    csv_path = os.path.join(DATA_DIR, f"{sheet_name}.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("=" * 60)
    print(f"CLINICAL PARAMETER: {sheet_name} (Impedance Modality)")
    print("=" * 60)

    # Patients are columns 1 through 18
    patient_ids = [int(float(x)) for x in df.iloc[0, 1:19].values]
    print(f"Total Hospital Patients: {len(patient_ids)}")
    print(f"Patient IDs: {patient_ids}")

    # Frequencies are in rows 1 to 7
    freqs = df.iloc[1:8, 0].values
    print(f"Frequencies Measured: {list(freqs)}")

    print("\nPairwise Before vs After Infiltration Comparison:")
    print(f"{'Freq':<10} | {'Before Mean':<12} | {'After Mean':<12} | {'Mean Change':<12} | {'% Drop':<10} | {'p-value':<10}")
    print("-" * 75)

    for i, freq in enumerate(freqs):
        row_before = i + 1
        row_after = i + 10 # after section starts at row 9 + 1

        vals_before = df.iloc[row_before, 1:19].astype(float).values
        vals_after = df.iloc[row_after, 1:19].astype(float).values

        diff = vals_after - vals_before
        pct_drop = ((vals_after - vals_before) / np.abs(vals_before)) * 100.0

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(vals_after, vals_before)

        print(f"{freq:<10} | {np.mean(vals_before):<12.2f} | {np.mean(vals_after):<12.2f} | {np.mean(diff):<12.2f} | {np.mean(pct_drop):<9.1f}% | {p_val:<10.4e}")

if __name__ == '__main__':
    for sheet in ['Z', 'R', 'Xc', 'Cm']:
        analyze_sheet(sheet)
        print()
