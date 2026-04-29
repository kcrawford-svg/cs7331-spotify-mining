# Spotify Genre Classification

Machine learning scripts for classifying Spotify track genres using audio features.

## Files

### `eda.ipynb`

Exploratory data analysis on the raw Spotify dataset. Covers feature group
identification, class balance verification, popularity distribution, audio
feature distributions and skewness analysis, correlation heatmap, mood
quadrant visualization using Russell (1980), and IQR outlier detection.
All findings feed directly into preprocessing decisions.

### `preprocessing.ipynb`

Cleans and prepares the dataset for all three modeling objectives. Steps include
duplicate removal, outlier capping, log transform on skewed features, feature
scaling, popularity binarization, and stratified train/val/test splitting.
Generates all files in data/processed/ required by modeling notebooks and scripts.


### `Spotify_mining.ipynb`

Trains and evaluates a classifier on binary popularity labels to predict whether a 
track will be popular or not.


### `mood_cluster.ipynb`

Mood-Based Clustering and Classification - Applies k-means
clustering in valence/energy space consistent with Russell's (1980)
circumplex model. Trains and evaluates three supervised classifiers on
derived mood labels. Analyzes mood distribution across genres with
chi-square significance testing

### `ensemble_methods.py`

Trains and evaluates two ensemble classifiers for multi-class genre prediction:

- **Random Forest** — 100 decision trees, parallelized (`n_jobs=-1`)
- **AdaBoost** — 100 boosting estimators

Both models are evaluated on a validation set and a held-out test set, printing accuracy, precision, recall, F1-score, a full classification report, and a confusion matrix for each.

### `linear_model.py`

Trains and evaluates a **LinearSVC** (linear kernel Support Vector Machine) for genre classification. Runs up to 2000 iterations to aid convergence on the multi-class problem. Reports the same metrics as `ensemble_methods.py` on both validation and test splits.

## Features

Both scripts use the same 15 audio features from the Spotify dataset:

| Feature | Description |
|---|---|
| `popularity` | Track popularity score (0–100) |
| `duration_ms` | Track length in milliseconds |
| `explicit` | Whether the track has explicit lyrics (0/1) |
| `danceability` | How suitable the track is for dancing (0.0–1.0) |
| `energy` | Perceptual intensity and activity (0.0–1.0) |
| `key` | Estimated musical key (0–11) |
| `loudness` | Overall loudness in decibels |
| `mode` | Major (1) or minor (0) modality |
| `speechiness` | Presence of spoken words (0.0–1.0) |
| `acousticness` | Confidence the track is acoustic (0.0–1.0) |
| `instrumentalness` | Predicts whether a track has no vocals (0.0–1.0) |
| `liveness` | Presence of a live audience (0.0–1.0) |
| `valence` | Musical positiveness (0.0–1.0) |
| `tempo` | Estimated beats per minute |
| `time_signature` | Estimated number of beats per bar |

## Dataset

Expected files under `Datasets/`:

```
Datasets/
  spotifytracks.csv       # Full track data with audio features
  y_genre_train.csv       # Training split labels
  y_genre_val.csv         # Validation split labels
  y_genre_test.csv        # Test split labels
```

## Processed Data Files

| File | Description | Use For |
|---|---|---|
| spotify_clean.csv | Deduplicated, outliers capped, raw scale | Tree-based models |
| spotify_log_transformed.csv | Log transform applied, pre-scaling reference | Reference |
| spotify_standard_scaled.csv | Log transformed + StandardScaler | SVM, k-means |
| spotify_minmax_scaled.csv | Log transformed + MinMaxScaler | Neural network |
| X_train/val/test.csv | Feature splits (70/10/20, stratified) | All models |
| y_genre_train/val/test.csv | Genre labels | Objective 1 |
| y_popular_class_train/val/test.csv | Binary popularity labels (threshold=35) | Objective 2 |
| y_popular_reg_train/val/test.csv | Raw popularity scores | Objective 2 regression |

## Model Files

| File | Description |
|---|---|
| standard_scaler.pkl | Fitted StandardScaler |
| min_max_scaler.pkl | Fitted MinMaxScaler |


Each label CSV maps a track index to its `track_genre` column.

## Requirements

```
pandas
scikit-learn
```

Install with:

```bash
pip install pandas scikit-learn
```

## Usage

```bash
python ensemble_methods.py
python linear_model.py
```
