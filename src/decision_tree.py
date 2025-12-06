from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, roc_auc_score
import numpy as np

def train_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_proba = clf.predict_proba(X_test)[:,1]
        pos_label = max(unique_labels)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        recall = recall_score(y_test, y_pred, average='macro')
        roc_auc = None

    return clf, {'recall': recall, 'roc_auc': roc_auc}
