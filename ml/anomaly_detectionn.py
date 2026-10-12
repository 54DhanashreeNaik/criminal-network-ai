import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


DATA_DIR = Path("cleaned_data")

INPUT_FILE = DATA_DIR / "activities_clean.csv"
OUTPUT_FILE = DATA_DIR / "anomaly_results.csv"


print("=" * 70)
print("CRIMINAL NETWORK AI - ANOMALY DETECTION")
print("=" * 70)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nRecords loaded: {len(df)}")


# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

features = [
    "call_count",
    "unique_contacts",
    "locations_visited",
    "transaction_count",
    "activity_score"
]

X = df[features].copy()

y_true = df["synthetic_anomaly"].astype(int)


# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

X = X.fillna(0)


# --------------------------------------------------
# SCALE FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# ISOLATION FOREST
# --------------------------------------------------

print("\nTraining Isolation Forest...")

anomaly_rate = y_true.mean()

model = IsolationForest(
    n_estimators=300,
    contamination=anomaly_rate,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

predictions = model.predict(X_scaled)

# Isolation Forest:
#   1  = normal
#  -1  = anomaly

df["model_prediction"] = predictions

df["predicted_anomaly"] = (
    df["model_prediction"] == -1
).astype(int)


# --------------------------------------------------
# ANOMALY SCORE
# --------------------------------------------------

# Lower decision_function values indicate
# more anomalous observations.

decision_scores = model.decision_function(X_scaled)

df["anomaly_score"] = -decision_scores


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

y_pred = df["predicted_anomaly"]

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 70)
print("ANOMALY DETECTION RESULTS")
print("=" * 70)

print(f"\nActual anomalies     : {y_true.sum()}")
print(f"Predicted anomalies  : {y_pred.sum()}")

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)


# --------------------------------------------------
# TOP ANOMALIES
# --------------------------------------------------

print("\nTop 20 detected anomalies:")

top_anomalies = df.sort_values(
    "anomaly_score",
    ascending=False
).head(20)

print(
    top_anomalies[
        [
            "person_id",
            "date",
            "call_count",
            "unique_contacts",
            "locations_visited",
            "transaction_count",
            "activity_score",
            "synthetic_anomaly",
            "predicted_anomaly",
            "anomaly_score"
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE.resolve())

print("\n" + "=" * 70)
print("ANOMALY DETECTION COMPLETED")
print("=" * 70)