import pandas as pd
from pathlib import Path

DATA_DIR = Path("cleaned_data")

INPUT_FILE = DATA_DIR / "case_network_index.csv"
OUTPUT_FILE = DATA_DIR / "case_priority.csv"

print("=" * 70)
print("CRIMINAL NETWORK AI - CASE PRIORITY ENGINE")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series([0.0] * len(series), index=series.index)

    return (series - minimum) / (maximum - minimum)


df["anomaly_score"] = normalize(
    df["case_anomaly_count"]
)

df["network_score"] = normalize(
    df["max_network_importance"]
)

df["connectivity_score"] = normalize(
    df["graph_connections"]
)

df["entity_score"] = normalize(
    df["entity_count"]
)

# ---------------------------------------------------------
# EXPLAINABLE PRIORITY SCORE
# ---------------------------------------------------------

df["priority_score"] = (
    0.40 * df["anomaly_score"]
    + 0.30 * df["network_score"]
    + 0.20 * df["connectivity_score"]
    + 0.10 * df["entity_score"]
)

# ---------------------------------------------------------
# PRIORITY LEVEL
# ---------------------------------------------------------

def assign_priority(score):

    if score >= 0.70:
        return "HIGH"

    elif score >= 0.40:
        return "MEDIUM"

    else:
        return "LOW"


df["priority"] = df["priority_score"].apply(
    assign_priority
)

# ---------------------------------------------------------
# EXPLANATION
# ---------------------------------------------------------

def generate_explanation(row):

    reasons = []

    if row["case_anomaly_count"] > 0:
        reasons.append(
            f"{int(row['case_anomaly_count'])} anomalous activity record(s)"
        )

    if row["max_network_importance"] >= df[
        "max_network_importance"
    ].quantile(0.75):
        reasons.append(
            "high network importance"
        )

    if row["graph_connections"] >= df[
        "graph_connections"
    ].quantile(0.75):
        reasons.append(
            "high graph connectivity"
        )

    if row["entity_count"] >= df[
        "entity_count"
    ].quantile(0.75):
        reasons.append(
            "large number of linked entities"
        )

    if not reasons:
        reasons.append(
            "limited structural or anomaly evidence"
        )

    return "; ".join(reasons)


df["explanation"] = df.apply(
    generate_explanation,
    axis=1
)

# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

df = df.sort_values(
    "priority_score",
    ascending=False
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCases analysed:", len(df))

print("\nPriority distribution:")
print(
    df["priority"].value_counts()
)

print("\nTop 15 cases:")

print(
    df[
        [
            "case_id",
            "crime_type",
            "status",
            "case_anomaly_count",
            "graph_connections",
            "max_network_importance",
            "priority_score",
            "priority",
            "explanation"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE.resolve())

print("\n" + "=" * 70)
print("CASE PRIORITY ENGINE COMPLETED")
print("=" * 70)