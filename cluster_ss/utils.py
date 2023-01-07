import numpy as np
import pandas as pd

def convert_types(X):
    """
    Simple data cleaning and checkup function.

    Params:
        - X: Dataset on numpy or Pandas dataframe.

    Returns:
        - X2: Pandas dataframe with X input data.
    """
    if isinstance(X, list) or isinstance(X, dict):
        raise TypeError("The X parameter need to be a pandas DataFrame")

    if isinstance(X, np.ndarray):
        X2 = pd.DataFrame(X.copy())

    else:
        X2 = X.copy()

    X2 = X2.dropna().reset_index(drop=True)

    if X2.empty:
        raise ValueError("Empty Pandas DataFrame")
    
    try:
        for k in X2.columns:
            X2[k] = X2[k].astype('float64')
        
        return X2

    except:
        raise ValueError(f"Error could not convert feature '{k}' for float64 data type")