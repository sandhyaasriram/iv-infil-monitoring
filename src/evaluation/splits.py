# Evaluation Split Logic
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

WESAD_TRAIN_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13']
WESAD_VAL_SUBJECTS = ['S14', 'S15']
WESAD_TEST_SUBJECTS = ['S16', 'S17']

def split_iv_trials(df, test_size=0.25, val_size=0.15, random_state=42):
    trial_metadata = df.groupby('trial_id')[['severity', 'condition']].first().reset_index()
    unique_trials = trial_metadata['trial_id'].values
    strat_key = trial_metadata['severity']

    train_val_trials, test_trials = train_test_split(
        unique_trials,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key
    )

    if val_size > 0:
        val_ratio_of_train = val_size / (1.0 - test_size)
        train_val_meta = trial_metadata[trial_metadata['trial_id'].isin(train_val_trials)]
        train_trials, val_trials = train_test_split(
            train_val_trials,
            test_size=val_ratio_of_train,
            random_state=random_state,
            stratify=train_val_meta['severity']
        )
    else:
        train_trials = train_val_trials
        val_trials = np.array([])

    return {
        'train_df': df[df['trial_id'].isin(train_trials)].copy(),
        'val_df': df[df['trial_id'].isin(val_trials)].copy() if len(val_trials) > 0 else pd.DataFrame(),
        'test_df': df[df['trial_id'].isin(test_trials)].copy(),
        'train_trials': train_trials,
        'val_trials': val_trials,
        'test_trials': test_trials,
    }
