import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, cohen_kappa_score, recall_score, precision_score, confusion_matrix, f1_score, roc_auc_score

def calculate_binary_metrics(y_true, y_pred, y_pred_probs):
    """
    Calculate accuracy, balanced accuracy, sensitivity, specificity, precision and F1-score for binary classification.
    """
    # calculate auc
    auc = roc_auc_score(y_true, y_pred_probs)
    # calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, average='binary')
    # specificity = recall_score(y_true, y_pred, pos_label=0)
    precision = precision_score(y_true, y_pred, average='binary')
    f1_sc = f1_score(y_true, y_pred, average='binary')
    # confussion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    test_NPV=tn/(tn+fn)
    test_specificity=tn/(tn+fp)
    # test_ppv = tp / (tp + fp)
    # return results
    results = {"roc_auc":auc, "accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "sensitivity": sensitivity, "precision": precision, "f1_score": f1_sc,
               "specificity": test_specificity, "NPV": test_NPV}
    return results