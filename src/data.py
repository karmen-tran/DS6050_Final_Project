import pandas as pd
from sklearn.model_selection import train_test_split

def align_target(X, y):
    """Align features and target into Series/DataFrame"""
    if hasattr(y, 'columns'):
        y_series = y[y.columns[0]]
    else:
        y_series = y.iloc[:, 0] if hasattr(y, 'iloc') else y
    X = X.reset_index(drop=True)
    y_series = y_series.reset_index(drop=True)
    return X, y_series

def stratified_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split dataset into train, validation, and test sets"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, stratify=y_train_val, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
