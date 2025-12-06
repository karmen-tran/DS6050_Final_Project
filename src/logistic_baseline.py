from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
import numpy as np

def safe_roc_auc(y_true, y_proba, n_classes):
    """Safely compute ROC-AUC for binary or multiclass"""
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:,1])
    else:
        try:
            return roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except ValueError:
            return np.nan

def train_logistic(X_train, y_train, X_test, y_test, class_weight='balanced'):
    """Train class-weighted Logistic Regression and return metrics"""
    n_classes = len(np.unique(y_train))
    
    if n_classes == 2:
        clf = LogisticRegression(max_iter=2000, class_weight=class_weight, solver='lbfgs', random_state=42)
    else:
        from sklearn.multiclass import OneVsRestClassifier
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight=class_weight, solver='lbfgs', random_state=42)
        )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if n_classes == 2 else None

    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = safe_roc_auc(y_test, y_proba, n_classes) if y_proba is not None else None

    return clf, {'f1_macro': f1_macro, 'bal_acc': bal_acc, 'roc_auc': roc_auc}
