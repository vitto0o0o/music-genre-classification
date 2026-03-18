# Music genre classification: Machine Learning on audio data

Multi-class music genre classifier built on the GTZAN dataset using a OVR binary classification strategy.

## What this project does

Classifies 10 music genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) from audio features extracted from 3-second segments. Instead of treating this as a single multi-class problem, it decomposes it into 10 independent binary classifiers, one per genre, and selects the genre with the highest confidence at prediction time.

## Methods compared

| Model | Strategy |
|---|---|
| Logistic Regression | Linear baseline |
| K-Nearest Neighbours | Instance-based |
| Random Forest | Ensemble method |

## Technical pipeline

- Exploratory data analysis & feature correlation
- StandardScaler normalisation inside sklearn Pipelines
- Stratified 80/20 train/test split (no data leakage)
- GridSearchCV hyperparameter tuning
- 5-fold stratified cross-validation
- PCA for dimensionality reduction
- Evaluation: macro-F1, per-class F1, confusion matrix, permutation importance

## Stack

Python · scikit-learn · pandas · NumPy · matplotlib

## Dataset

[GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) — `features_3_sec.csv`
