<div align="center">

# 🤖 Netflix Content Classification ML Model
### Auspify Technologies Data Science Internship — Task 5 (Advanced)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

**Author:** Rishi Prajapati  
**Repository:** [rishi-prajapati-git/Auspify_Project-5](https://github.com/rishi-prajapati-git/Auspify_Project-5)

---

</div>

## 📌 Task Overview

In Task 5 of the Auspify Technologies Data Science Internship, we engineer and benchmark multi-class **Machine Learning Classification Models** to automatically categorize Netflix titles into their target demographic ratings (`Adults`, `Teens`, `Older Kids`, `Kids`) based on content genres, title semantics, production country, format, and release era.

---

## 🎯 Task Workflow (from Auspify Curriculum)

| Step | Milestone | Implementation & Methodology |
| :---: | :--- | :--- |
| **1** | **Feature Selection & Preparation** | Built a multi-modal feature transformer (TF-IDF on genres/titles, One-Hot Encoding on country/format, and MinMax scaling on release year). |
| **2** | **Stratified Train-Test Split** | Split 8,488 titles into 80% training (6,790) and 20% holdout testing (1,698) preserving demographic class ratios. |
| **3** | **Train Classification Models** | Trained and tuned 4 algorithms: Logistic Regression, Linear SVM, Multinomial Naive Bayes, and Random Forest. |
| **4** | **Evaluate Model Performance** | Evaluated models on holdout test set using Accuracy, Precision, Recall, Weighted F1-Score, and Normalized Confusion Matrices. |
| **5** | **Compare Model Accuracy & Persist** | Identified Logistic Regression as the top-performing model ($63.84\%$ accuracy, $0.6310$ weighted F1) and serialized the entire pipeline. |

---

## 🗂️ Project Directory Structure

```plaintext
Task-5_Machine_Learning_Classification/
├── data/
│   └── netflix_cleaned.csv                    # Preprocessed Netflix dataset
├── figures/                                   # High-resolution (300 DPI) visual diagnostics
│   ├── 01_algorithm_benchmark_comparison.png  # Accuracy and F1 benchmark across 4 algorithms
│   └── 02_best_model_confusion_matrix.png     # Normalized confusion matrix heatmap
├── models/                                    # Serialized production pipeline
│   └── best_classifier_pipeline.joblib        # Fitted preprocessor + Logistic Regression model
├── notebooks/
│   └── ml_classification_walkthrough.ipynb    # Interactive step-by-step Jupyter Notebook
├── reports/
│   └── classification_evaluation_summary.json # Machine-readable evaluation report
├── src/
│   ├── data_preparer.py                       # Multi-modal ColumnTransformer & stratified splitter
│   ├── model_trainer.py                       # Multi-model training and benchmarking engine
│   ├── evaluator.py                           # Classification reporting & confusion matrix extractor
│   ├── visualizer.py                          # Publication-grade chart generation engine
│   ├── model_persistence.py                   # Model serialization & JSON report exporter
│   ├── test_classifier.py                     # Automated unit and sanity test suite
│   └── main.py                                # End-to-end execution runner
├── .gitignore, requirements.txt, TODO.md
└── README.md                                  # Professional portfolio presentation
```

---

## 🏆 Model Benchmark Comparison

| Algorithm | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Winner)** | **63.84%** | **0.6328** | **0.6384** | **0.6310** |
| **Multinomial Naive Bayes** | 63.25% | 0.6276 | 0.6325 | 0.6266 |
| **Linear SVM** | 63.25% | 0.6266 | 0.6325 | 0.6254 |
| **Random Forest** | 63.55% | 0.6379 | 0.6355 | 0.6159 |

---

## 🎯 Winning Model Per-Class Performance

- **Adults (`TV-MA`, `R`):** Precision: `0.6848` | Recall: `0.7993` | F1: `0.7376`
- **Kids (`TV-Y`, `G`):** Precision: `0.7222` | Recall: `0.5652` | F1: `0.6341`
- **Older Kids (`TV-PG`, `PG`):** Precision: `0.6507` | Recall: `0.5017` | F1: `0.5665`
- **Teens (`TV-14`, `PG-13`):** Precision: `0.5324` | Recall: `0.4811` | F1: `0.5055`

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/rishi-prajapati-git/Auspify_Project-5.git
cd Auspify_Project-5
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Automated Training Pipeline
```bash
python src/main.py
```

### 4. Run Model Quality Assurance Tests
```bash
python src/test_classifier.py
```

---

## 📬 Auspify Internship Details

- **Company:** Auspify Technologies
- **Program:** Data Science 4-Week Practical Learning Internship
- **Task:** Task 5 — Machine Learning Classification Model
- **Hashtags:** `#Auspify` `#AuspifyTechnologies` `#AuspifyInternship` `#AuspifyProjects`
