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

def undersample_class0(X, y, n_neighbors=3):
    """
    Undersample class 0 by finding the nearest neighbors to class 0 samples
    for each sample in classes 1–5.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix (no ID or label columns).
    y : pandas.Series
        ISUP grade labels (0–5).
    n_neighbors : int, default=3
        Number of nearest class-0 neighbors to select per minority sample.

    Returns
    -------
    X_new : pandas.DataFrame
        Resampled feature matrix.
    y_new : pandas.Series
        Resampled labels, aligned with X_new.
    """
    idx_class0 = y.index[y == 0]

    if len(idx_class0) == 0:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    # Fit KNN once on class-0 samples (capped to available samples)
    k = min(n_neighbors, len(idx_class0))
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X.loc[idx_class0])

    selected_class0 = []

    # For each class 1–5, find nearest class-0 samples
    for cls in range(1, 6):
        idx_cls = y.index[y == cls]

        if len(idx_cls) == 0:
            continue

        _, indices = knn.kneighbors(X.loc[idx_cls])
        nearest_idx0 = idx_class0[indices.flatten()].tolist()

        selected_class0.extend(nearest_idx0)

    selected_class0 = list(set(selected_class0))

    keep_idx = y.index[y != 0].tolist() + selected_class0
    X_new = X.loc[keep_idx].reset_index(drop=True)
    y_new = y.loc[keep_idx].reset_index(drop=True)

    return X_new, y_new



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
