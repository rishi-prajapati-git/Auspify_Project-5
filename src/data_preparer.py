"""
Feature Engineering & Dataset Preparation Module for ML Classification.
Prepares features (Genres, Format, Country, Release Year, Title Tokens) to classify Netflix content
into Target Audience Demographic Tiers (Adults, Teens, Kids, Older Kids).
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# Valid target audience categories with sufficient sample sizes
VALID_AUDIENCES = ["Adults", "Teens", "Older Kids", "Kids"]


def load_and_filter_data(filepath: str = "data/netflix_cleaned.csv") -> pd.DataFrame:
    """
    Load dataset and filter to titles with well-defined target audience labels.
    """
    df = pd.read_csv(filepath)
    filtered = df[df["target_audience"].isin(VALID_AUDIENCES)].copy()
    print(f"[INFO] Dataset loaded. Retained {len(filtered)} titles belonging to: {VALID_AUDIENCES}")
    print("Class Distribution:")
    print(filtered["target_audience"].value_counts().to_string())
    return filtered


def build_preprocessor() -> ColumnTransformer:
    """
    Build a multi-modal ColumnTransformer:
    - Text: TF-IDF on 'listed_in' (genres)
    - Text: TF-IDF on 'title'
    - Categorical: One-Hot Encoding on 'type' and 'primary_country'
    - Numeric: StandardScaler on 'release_year'
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("genre_tfidf", TfidfVectorizer(max_features=1500, stop_words="english", ngram_range=(1, 2)), "listed_in"),
            ("title_tfidf", TfidfVectorizer(max_features=1000, stop_words="english"), "title"),
            ("cat_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["type", "primary_country"]),
            ("num_scaler", StandardScaler(), ["release_year"]),
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split to preserve demographic proportions.
    """
    feature_cols = ["listed_in", "title", "type", "primary_country", "release_year"]
    X = df[feature_cols].copy()
    y = df["target_audience"].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test
