import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load and merge data


def load_data(features_path='../../features.csv', targets_path='../../targets.csv'): 
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
    X = data.drop(['study_id', 'patient_id', 'case_ISUP'], axis=1)
    y = data['case_ISUP']

    df = X.copy()
    df["case_ISUP"] = y

    idx_class0 = df.index[df["case_ISUP"] == 0]

    selected_class0 = []

    # For each class 1–5, find nearest class-0 samples
    for cls in range(1, 6):
        idx_cls = df.index[df["case_ISUP"] == cls]

        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(df.loc[idx_class0, X.columns])

        _, indices = knn.kneighbors(df.loc[idx_cls, X.columns])
        nearest_idx0 = idx_class0[indices.flatten()]

        selected_class0.extend(nearest_idx0)

    selected_class0 = list(set(selected_class0))

    df_new = pd.concat([
        df[df["case_ISUP"] != 0],   # keep all minority classes
        df.loc[selected_class0]     # keep selected class-0 samples
    ])

    return df_new



# Compute SMOTE strategy for ALL classes

def compute_smote_strategy_all_classes(df_new):
    """
    Compute a SMOTE strategy for all classes in the given dataframe.

    Parameters
    ----------
    df_new : pandas.DataFrame
        Dataframe containing the features and targets data.

    Returns
    -------
    strategy : dict
        A dictionary containing the class labels as keys and the
        corresponding oversampling counts as values.

    Notes
    -----
    The strategy is computed by finding the maximum number of samples
    across all classes and oversampling each class that has fewer
    samples than the max.
    """
    y = df_new["case_ISUP"]

    class_counts = y.value_counts()
    max_count = class_counts.max()

    # Oversample every class that has fewer samples than the max
    strategy = {
        cls: max_count
        for cls in class_counts.index
        if class_counts[cls] < max_count
    }

    return strategy
