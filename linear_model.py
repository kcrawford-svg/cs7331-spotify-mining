import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

FEATURES = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
]

tracks = pd.read_csv('Datasets/spotifytracks.csv', index_col=0)
tracks['explicit'] = tracks['explicit'].astype(int)

y_train = pd.read_csv('Datasets/y_genre_train.csv', index_col=0)
y_val = pd.read_csv('Datasets/y_genre_val.csv', index_col=0)
y_test = pd.read_csv('Datasets/y_genre_test.csv', index_col=0)

X_train = tracks.loc[y_train.index, FEATURES]
X_val = tracks.loc[y_val.index, FEATURES]
X_test = tracks.loc[y_test.index, FEATURES]
y_train_labels = y_train['track_genre']
y_val_labels = y_val['track_genre']
y_test_labels = y_test['track_genre']


def evaluate(name, y_true, y_pred):
    print(f"=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()


# SVM with linear kernel
svm = LinearSVC(random_state=42, max_iter=2000)
svm.fit(X_train, y_train_labels)
svm_val_preds = svm.predict(X_val)
svm_test_preds = svm.predict(X_test)

print("--- SVM (Linear Kernel): Validation ---")
evaluate("SVM Linear (Validation)", y_val_labels, svm_val_preds)
print("--- SVM (Linear Kernel): Test ---")
evaluate("SVM Linear (Test)", y_test_labels, svm_test_preds)
