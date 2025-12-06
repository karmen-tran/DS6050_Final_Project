import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

# -----------------------------
# ROC-AUC
# -----------------------------
def safe_roc_auc(y_true, y_proba, n_classes):
    """
    Safely compute ROC-AUC for binary or multiclass classification.
    y_proba: For binary, array of shape (n_samples, 2)
             For multiclass, array of shape (n_samples, n_classes)
    """
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:,1])
    else:
        try:
            return roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except ValueError:
            # fallback: if not all classes are present in y_true
            unique_test = np.unique(y_true)
            if len(unique_test) < n_classes:
                return roc_auc_score(y_true, y_proba[:, unique_test], multi_class='ovr', average='macro')
            else:
                return np.nan

# -----------------------------
# F1 Score
# -----------------------------
def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def f1_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

# -----------------------------
# Balanced Accuracy
# -----------------------------
def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

# -----------------------------
# Recall / Precision for binary or multiclass
# -----------------------------
def recall_pos(y_true, y_pred, pos_label=1, average='binary'):
    return recall_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=0)

def precision_pos(y_true, y_pred, pos_label=1, average='binary'):
    return precision_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=0)

# -----------------------------
# Confusion Matrix
# -----------------------------
def compute_confusion_matrix(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)

# -----------------------------
# Imbalance / Class Distribution Metrics
# -----------------------------
def imbalance_ratio(y):
    """
    Returns the ratio of majority class to minority class (collapsed to 2 classes)
    """
    counts = np.bincount(y) if np.issubdtype(y.dtype, np.integer) else y.value_counts().values
    majority = max(counts)
    minority = sum(counts) - majority
    if minority == 0:
        return np.inf
    return majority / minority

def class_distribution(y):
    """Returns dict {class_label: percentage}"""
    if hasattr(y, "value_counts"):
        counts = y.value_counts()
        percentages = y.value_counts(normalize=True) * 100
        return {cls: percentages[cls] for cls in counts.index}
    else:
        labels, counts = np.unique(y, return_counts=True)
        percentages = counts / counts.sum() * 100
        return {label: pct for label, pct in zip(labels, percentages)}
