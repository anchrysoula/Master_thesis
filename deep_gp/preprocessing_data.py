import pandas as pd
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE

# Load and merge data

def load_data(features_path='../features.csv', targets_path='../targets.csv'):
    """
    Load features and targets data from CSV files.

    Parameters
    ----------
    features_path : str, default='../features.csv'
        Path to the features CSV file.
    targets_path : str, default='../targets.csv'
        Path to the targets CSV file.

    Returns
    -------
    data : pandas.DataFrame
        Dataframe containing the merged features and targets data.
    """
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)

    data = features.merge(
        targets[['study_id', 'case_ISUP']],
        on='study_id'
    )
    return data



# Undersample class 0 using nearest neighbors

def undersample_class0(data):    
    """
    Undersample class 0 by finding the nearest neighbors to class 0 samples
    for each sample in classes 1–5.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the features and targets data.

    Returns
    -------
    df_new : pandas.DataFrame
        Dataframe containing the undersampled class 0 data.
    """
    X = data.drop(['study_id', 'patient_id','case_ISUP'], axis=1)
    y = data['case_ISUP']

    df = X.copy()
    df["case_ISUP"] = y

    idx_class0 = df.index[df["case_ISUP"] == 0]
    
    # Storage for selected class-0 samples 
    selected_class0 = []

    # For each class 1–5, find nearest class-0 sample

    for cls in range(1, 6):
        idx_cls = df.index[df["case_ISUP"] == cls]
        
        # Fit KNN on class 0 samples 
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(df.loc[idx_class0, X.columns])
        
        # For each sample in class cls, find nearest class-0 sample
        _, indices = knn.kneighbors(df.loc[idx_cls, X.columns])
        # Convert neighbor indices to original dataframe indices 
        nearest_idx0 = idx_class0[indices.flatten()]
        # Add to list of selected class-0 samples
        selected_class0.extend(nearest_idx0)

    # keep only unique samples in class 0
    selected_class0 = list(set(selected_class0))
    
    # new reduced dataset
    df_new = pd.concat([
        df[df["case_ISUP"] != 0], # keep all minority classes 
        df.loc[selected_class0] # keep only selected class-0 samples 
    ])

    return df_new

# Apply SMOTE to classes 3, 4, 5

def apply_smote(df_new):
    """
    Apply SMOTE to classes 3, 4, 5 in the given dataframe.

    Parameters
    ----------
    df_new : pandas.DataFrame
        Dataframe containing the features and targets data.

    Returns
    -------
    df_res : pandas.DataFrame
        Dataframe containing the resampled data using SMOTE.
    """
    X = df_new.drop(columns=["case_ISUP"])
    y = df_new["case_ISUP"]

    sm = SMOTE(
        sampling_strategy={3: 150, 4: 150, 5: 150},
        k_neighbors=3,
        random_state=42
    )

    X_res, y_res = sm.fit_resample(X, y)

    df_res = pd.concat([
        pd.DataFrame(X_res, columns=X.columns),
        pd.Series(y_res, name="case_ISUP")
    ], axis=1)

    return df_res
