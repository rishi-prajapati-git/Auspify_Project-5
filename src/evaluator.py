"""
Detailed Evaluation & Diagnostic Metrics Module.
Computes per-class classification reports, normalized confusion matrices, and model explainability metrics.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier_performance(
    pipeline: Pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[Dict[str, Any], np.ndarray, list]:
    """
    Generate granular classification metrics and confusion matrix.
    """
    y_pred = pipeline.predict(X_test)
    labels = sorted(list(y_test.unique()))
    
    # 1. Classification report dict
    report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    return report_dict, cm, labels
