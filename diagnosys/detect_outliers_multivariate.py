import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    features: list,
    contamination: float = 0.01,
    n_estimators: int = 300,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Detect multivariate outliers using Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame containing the data.
    features : list
        List of numeric columns to use for outlier detection.
    contamination : float, optional
        Estimated proportion of outliers in the dataset (default = 0.01).
    n_estimators : int, optional
        Number of trees in the forest (default = 200).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two additional columns:
        - 'outlier_score': anomaly score (higher = more anomalous)
        - 'outlier_flag': 1 for normal points, -1 for outliers
    """
    
    # Validate feature columns
    if not all(col in df.columns for col in features):
        missing = [col for col in features if col not in df.columns]
        raise ValueError(f"The following columns are missing in the DataFrame: {missing}")

    X = df[features].copy()

    # Train Isolation Forest model
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state
    )
    iso.fit(X)

    # Compute anomaly scores and flags
    df_result = df.copy()
    df_result['outlier_score'] = -iso.decision_function(X)  # Higher = more anomalous

    # Map: -1 (outlier) → 1, 1 (normal) → 0
    flags = iso.predict(X)
    df_result['outlier_flag'] = (flags == -1).astype(int)

    return (
        df_result
        .sort_values("outlier_score", ascending=False)
        .reset_index(drop=True)
    )

