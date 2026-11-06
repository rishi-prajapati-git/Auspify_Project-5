"""
Automated Testing & Model Sanity Verification Suite for Task 5.
Validates model loading, pipeline inference on unseen test samples, accuracy bounds, and output formatting.
"""

from pathlib import Path
import joblib
import pandas as pd
from data_preparer import load_and_filter_data, prepare_train_test_split


def test_classification_model_integrity():
    print("[TEST] Running automated ML classification test suite...")
    
    # 1. Load serialized model pipeline
    model_path = Path("models/best_classifier_pipeline.joblib")
    assert model_path.exists(), f"Model artifact not found at {model_path}"
    pipeline = joblib.load(model_path)
    print(f"[TEST] PASSED: Serialized model loaded ({type(pipeline.named_steps['classifier']).__name__}).")
    
    # 2. Test inference on synthetic unseen samples
    sample_data = pd.DataFrame([
        {
            "listed_in": "Children & Family Movies, Comedies",
            "title": "Adventures of Little Teddy",
            "type": "Movie",
            "primary_country": "United States",
            "release_year": 2020
        },
        {
            "listed_in": "Crime TV Shows, TV Dramas, TV Thrillers",
            "title": "Dark Syndicate",
            "type": "TV Show",
            "primary_country": "United States",
            "release_year": 2019
        }
    ])
    
    predictions = pipeline.predict(sample_data)
    assert len(predictions) == 2, f"Expected 2 predictions, got {len(predictions)}"
    print(f"[TEST] PASSED: Inference sanity check -> Sample 1: '{predictions[0]}', Sample 2: '{predictions[1]}'")
    
    # 3. Verify accuracy on holdout test set
    df = load_and_filter_data("data/netflix_cleaned.csv")
    _, X_test, _, y_test = prepare_train_test_split(df)
    test_acc = pipeline.score(X_test, y_test)
    assert test_acc > 0.60, f"Model accuracy degraded below acceptable baseline: {test_acc}"
    print(f"[TEST] PASSED: Holdout test accuracy validated ({test_acc:.4f} > 0.60).")
    
    print("\nALL TASK 5 ML CLASSIFICATION TESTS PASSED! READY FOR SUBMISSION.")


if __name__ == "__main__":
    test_classification_model_integrity()
