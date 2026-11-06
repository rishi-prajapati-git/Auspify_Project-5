"""
End-to-End Machine Learning Classification Pipeline Runner.
Orchestrates data preparation, multi-algorithm training, performance evaluation, visualization,
and artifact persistence.
"""

from pathlib import Path
import pandas as pd

from data_preparer import load_and_filter_data, prepare_train_test_split
from model_trainer import train_and_compare_models
from evaluator import evaluate_classifier_performance
from visualizer import generate_all_classification_figures
from model_persistence import save_best_model, save_evaluation_report


def run_classification_pipeline(
    data_path: str = "data/netflix_cleaned.csv",
    models_dir: str = "models",
    figures_dir: str = "figures",
    report_path: str = "reports/classification_evaluation_summary.json"
):
    print("=" * 70)
    print("AUSPIFY TECHNOLOGIES INTERNSHIP: TASK 5 - ML CLASSIFICATION MODEL")
    print("=" * 70)
    
    # 1. Prepare Data
    print("\n[STEP 1/5] Loading and preparing demographic dataset...")
    df = load_and_filter_data(data_path)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # 2. Train and Benchmark Classifiers
    print("\n[STEP 2/5] Training candidate classification models...")
    comparison_df, pipelines, best_algo_name = train_and_compare_models(
        X_train, X_test, y_train, y_test
    )
    print("\nAlgorithm Benchmark Comparison Summary:")
    print(comparison_df.to_string(index=False))
    
    # 3. Granular Evaluation on Winner
    print(f"\n[STEP 3/5] Evaluating fine-grained metrics for winner: '{best_algo_name}'...")
    best_pipe = pipelines[best_algo_name]
    report_dict, cm, labels = evaluate_classifier_performance(best_pipe, X_test, y_test)
    
    print("\nDetailed Per-Class Performance:")
    for lbl in labels:
        score_data = report_dict.get(lbl, {})
        print(f"  Class: {lbl:<12} | Precision: {score_data.get('precision', 0):.4f} | Recall: {score_data.get('recall', 0):.4f} | F1: {score_data.get('f1-score', 0):.4f}")
        
    # 4. Generate Visualizations
    print("\n[STEP 4/5] Rendering publication-quality evaluation figures...")
    generate_all_classification_figures(comparison_df, cm, labels, best_algo_name, figures_dir=figures_dir)
    
    # 5. Persist Best Model and Audit Report
    print("\n[STEP 5/5] Persisting trained pipeline and executive report...")
    save_best_model(best_pipe, best_algo_name, models_dir=models_dir)
    save_evaluation_report(comparison_df, best_algo_name, report_dict, report_path=report_path)
    
    print("\n" + "=" * 70)
    print("TASK 5 ML CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_classification_pipeline()
