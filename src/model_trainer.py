"""
Model Training & Comparison Engine.
Trains and compares 4 classification algorithms:
1. Multinomial Logistic Regression
2. Linear Support Vector Classifier (LinearSVC)
3. Multinomial Naive Bayes
4. Random Forest Classifier
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_preparer import build_preprocessor


def get_candidate_classifiers() -> Dict[str, Any]:
    """
    Returns candidate models with tuned hyperparameters.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.5),
        "Linear SVM": LinearSVC(random_state=42, max_iter=2000, C=0.8),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    }


def train_and_compare_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], str]:
    """
    Fit each pipeline on training data, evaluate on holdout test set, and identify the top performer.
    """
    classifiers = get_candidate_classifiers()
    results = []
    fitted_pipelines = {}
    
    print("\n[INFO] Training and evaluating candidate classification models...")
    for name, clf in classifiers.items():
        print(f"  -> Fitting {name}...")
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        results.append({
            "Algorithm": name,
            "Accuracy": round(float(acc), 4),
            "Precision (Weighted)": round(float(prec), 4),
            "Recall (Weighted)": round(float(rec), 4),
            "F1-Score (Weighted)": round(float(f1), 4)
        })
        fitted_pipelines[name] = pipe
        print(f"     Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
    comparison_df = pd.DataFrame(results).sort_values(by="F1-Score (Weighted)", ascending=False).reset_index(drop=True)
    best_algo_name = comparison_df.iloc[0]["Algorithm"]
    print(f"\n[WINNER] Best Performing Algorithm: '{best_algo_name}' (Accuracy: {comparison_df.iloc[0]['Accuracy']})")
    
    return comparison_df, fitted_pipelines, best_algo_name
