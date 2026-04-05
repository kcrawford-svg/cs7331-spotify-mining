# Preprocessing Guide — preprocessing.ipynb

Run preprocessing notebook before any modeling if you need to.
All the stuff below is so every will know what I did
and what the files generated are and do.
---

## Files Generated

### Clean Dataset (tree-based models)
data/processed/spotify_clean.csv
- Deduplicated, outliers capped, popularity binarized
- Could be used for Random Forest, AdaBoost, Gradient Boosting etc
- They didnt need scaling or log transforms

### Log Transformed + Standard Scaled (distance-based models)
data/processed/spotify_standard_scaled.csv
- Log transform applied to speechiness, liveness, instrumentalness, duration_ms
- StandardScaler applied to all continuous features (mean=0, std=1)
- Could be used for SVM, k-means clustering in Objective 3

### Log Transformed + MinMax Scaled (neural networks)
data/processed/spotify_minmax_scaled.csv
- Same log transform as above
- MinMaxScaler applied to all the continuous features with range 0-1
- Could be used for Neural Networks in Objective 1

### Scalers
outputs/models/scaler_standard.pkl  -> StandardScaler
outputs/models/scaler_minmax.pkl    -> MinMaxScaler

example script if you haven't used joblib
Load with:
  import joblib
  scaler = joblib.load('../outputs/models/scaler_standard.pkl')
  X_scaled = scaler.transform(X)

---

## Train / Val / Test Splits
Stratified 70/10/20 split on track_genre
All objectives use these same splits you dont have redefine them

data/processed/X_train.csv -> features, training set
data/processed/X_val.csv -> features, validation set
data/processed/X_test.csv -> features, test set

---

## Target Labels by Objective

### Objective 1 Genre Classification
data/processed/y_genre_train.csv
data/processed/y_genre_val.csv
data/processed/y_genre_test.csv

### Objective 2 Popularity Prediction
Regression target (raw score 0-100):
  data/processed/y_pop_reg_train.csv
  data/processed/y_pop_reg_val.csv
  data/processed/y_pop_reg_test.csv

Binary classification target (popular=1, not popular=0, threshold=35):
  data/processed/y_pop_cls_train.csv
  data/processed/y_pop_cls_val.csv
  data/processed/y_pop_cls_test.csv

### Objective 3 Mood Clustering
No pre-defined labels k-means can possibly derive them 
Load full dataset:
  data/processed/spotify_clean.csv -> raw scale for reference
  data/processed/spotify_standard_scaled.csv -> use for k-means

mood_prelim column in spotify_clean.csv is a threshold-based
baseline reference only whoever is working on this one has 
to get the final mood labels

---

## Quick Reference by Model

| Model | Features | Targets |
|---|---|---|
| Random Forest (Obj 1) | X_train/val/test.csv | y_genre_*.csv |
| AdaBoost (Obj 1) | X_train/val/test.csv | y_genre_*.csv |
| SVM (Obj 1) | spotify_standard_scaled.csv | y_genre_*.csv |
| Neural Network (Obj 1) | spotify_minmax_scaled.csv | y_genre_*.csv |
| Gradient Boosting (Obj 2) | X_train/val/test.csv | y_pop_reg_*.csv |
| Linear Regression (Obj 2) | X_train/val/test.csv | y_pop_reg_*.csv |
| Random Forest (Obj 2) | X_train/val/test.csv | y_pop_cls_*.csv |
| SVM (Obj 2) | spotify_standard_scaled.csv | y_pop_cls_*.csv |
| K-means (Obj 3) | spotify_standard_scaled.csv | derived by k-means |
| Classifier (Obj 3) | X_train/val/test.csv | derived mood labels |

---

## Just so we dont break training
- Please dont load from data/raw/ in modeling notebooks
- Please dont redefine the train/validation/test splits
- Use random_state=42 for any randomness
- All splits are stratified on track_genre