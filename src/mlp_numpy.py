import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

def train_mlp(X_train, y_train, X_test, y_test, hidden_layer_sizes=(64,), activation='tanh', max_iter=500):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                        max_iter=max_iter, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if len(np.unique(y_train)) == 2 else None

    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba[:,1]) if y_proba is not None else None

    saliency = np.abs(clf.coefs_[0]).mean(axis=1)
    
    return clf, {'f1_macro': f1_macro, 'bal_acc': bal_acc, 'roc_auc': roc_auc}, saliency
