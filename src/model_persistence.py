"""
Model Persistence & Artifact Management Module.
Serializes best pipeline model (preprocessor + classifier) and evaluation JSON reports to disk.
"""

from pathlib import Path
import json
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd


def save_best_model(pipeline: Pipeline, algo_name: str, models_dir: str = "models"):
    out_path = Path(models_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    model_file = out_path / "best_classifier_pipeline.joblib"
    joblib.dump(pipeline, model_file, compress=3)
    print(f"[INFO] Serialized best model pipeline ({algo_name}) saved to: {model_file.resolve()}")


def save_evaluation_report(
    comparison_df: pd.DataFrame, 
    best_algo_name: str, 
    report_dict: dict, 
    report_path: str = "reports/classification_evaluation_summary.json"
):
    rep_file = Path(report_path)
    rep_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "project": "Auspify Technologies Data Science Internship - Task 5",
        "task_name": "Machine Learning Classification Model",
        "objective": "Classify Netflix titles into demographic audience tiers based on metadata",
        "target_classes": list(report_dict.keys()),
        "algorithm_comparison": comparison_df.to_dict(orient="records"),
        "best_algorithm": best_algo_name,
        "best_model_metrics": {
            "accuracy": round(float(report_dict["accuracy"]), 4),
            "macro_avg_f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
            "weighted_avg_f1": round(float(report_dict["weighted avg"]["f1-score"]), 4)
        },
        "per_class_performance": {
            k: {
                "precision": round(float(v["precision"]), 4),
                "recall": round(float(v["recall"]), 4),
                "f1_score": round(float(v["f1-score"]), 4),
                "support": int(v["support"])
            }
            for k, v in report_dict.items() if isinstance(v, dict) and "f1-score" in v
        }
    }
    
    with open(rep_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"[SUCCESS] Classification evaluation report generated at: {rep_file.resolve()}")
