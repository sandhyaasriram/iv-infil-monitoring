import os
import glob
import pickle
import numpy as np
import pandas as pd

def inspect_wesad(wesad_dir):
    print('=' * 55)
    print('            1. REAL DATASET: WESAD AUDIT            ')
    print('=' * 55)
    subjects = sorted(glob.glob(os.path.join(wesad_dir, 'S*')))
    print('Total Subject Folders: ' + str(len(subjects)))
    rows = []
    for s_path in subjects:
        s_id = os.path.basename(s_path)
        pkl_path = os.path.join(s_path, s_id + '.pkl')
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as fp:
            d = pickle.load(fp, encoding='latin1')
        wrist = d['signal']['wrist']
        labels = d['label']
        rows.append({
            'Subject': s_id,
            'Duration_min': round(len(labels) / 700 / 60, 1),
            'Wrist_TEMP_4Hz': len(wrist['TEMP']),
            'Wrist_BVP_64Hz': len(wrist['BVP']),
            'Wrist_EDA_4Hz': len(wrist['EDA']),
            'Labels': list(np.unique(labels))
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

def inspect_synthetic(csv_path):
    print('\n' + '=' * 55)
    print('      2. SYNTHETIC IV INFILTRATION DATASET AUDIT      ')
    print('=' * 55)
    if not os.path.exists(csv_path):
        print('Not found: ' + csv_path)
        return
    df = pd.read_csv(csv_path)
    print('Total Timestep Rows: ' + str(len(df)))
    print('Total Trials: ' + str(df['trial_id'].nunique()))
    print('Columns: ' + str(df.columns.tolist()))
    print('\nLabel Distribution:')
    for lab, cnt in df['label'].value_counts().sort_index().items():
        print('  Class ' + str(lab) + ': ' + str(cnt) + ' samples (' + str(round(cnt/len(df)*100, 1)) + '%)')

    print('\nRobustness Conditions:')
    conds = df.groupby('condition')['trial_id'].nunique()
    for c, num in conds.items():
        print('  ' + str(c) + ': ' + str(num) + ' trials')

if __name__ == '__main__':
    w_dir = os.path.abspath('../WESAD')
    if not os.path.exists(w_dir):
        w_dir = os.path.abspath('c:/mer/iot projecttt/WESAD')
    inspect_wesad(w_dir)
    inspect_synthetic('synthetic_iv_infiltration_dataset.csv')
