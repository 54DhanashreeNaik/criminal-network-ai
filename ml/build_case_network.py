import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("cleaned_data")

CASES_FILE = DATA_DIR / "cases_clean.csv"
RESOLVED_FILE = DATA_DIR / "resolved_entities.csv"
PROFILES_FILE = DATA_DIR / "investigation_profiles.csv"
ANOMALY_FILE = DATA_DIR / "anomaly_results.csv"
GRAPH_FILE = DATA_DIR / "network_graph.graphml"

OUTPUT_FILE = DATA_DIR / "case_network_index.csv"

print("=" * 70)
print("CRIMINAL NETWORK AI - CASE NETWORK INDEX")
print("=" * 70)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

cases = pd.read_csv(CASES_FILE)
resolved = pd.read_csv(RESOLVED_FILE)
profiles = pd.read_csv(PROFILES_FILE)
anomalies = pd.read_csv(ANOMALY_FILE)
G = nx.read_graphml(GRAPH_FILE)

# --------------------------------------------------
# CREATE PERSON PROFILE LOOKUP
# --------------------------------------------------

profile_lookup = profiles.set_index("person_id").to_dict(
    orient="index"
)

# --------------------------------------------------
# CREATE ANOMALY LOOKUP
# --------------------------------------------------

person_anomaly_lookup = (
    anomalies.groupby("person_id")["predicted_anomaly"]
    .sum()
    .to_dict()
)

# --------------------------------------------------
# BUILD CASE INDEX
# --------------------------------------------------

results = []

for _, case in cases.iterrows():

    case_id = case["case_id"]

    case_entities = resolved[
        resolved["case_id"] == case_id
    ]

    # Unique canonical entities
    entity_ids = (
        case_entities["canonical_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # --------------------------------------------------
    # ENTITY COUNTS
    # --------------------------------------------------

    entity_count = len(entity_ids)

    person_ids = case_entities[
        case_entities["entity_type"] == "PERSON"
    ]["canonical_id"].dropna().unique()

    phone_ids = case_entities[
        case_entities["entity_type"] == "PHONE"
    ]["canonical_id"].dropna().unique()

    vehicle_ids = case_entities[
        case_entities["entity_type"] == "VEHICLE"
    ]["canonical_id"].dropna().unique()

    location_ids = case_entities[
        case_entities["entity_type"] == "LOCATION"
    ]["canonical_id"].dropna().unique()

    # --------------------------------------------------
    # PERSON INTELLIGENCE
    # --------------------------------------------------

    case_network_scores = []

    case_anomaly_count = 0

    for person_id in person_ids:

        if person_id in profile_lookup:

            score = profile_lookup[
                person_id
            ]["network_importance_score"]

            case_network_scores.append(
                float(score)
            )

        case_anomaly_count += int(
            person_anomaly_lookup.get(
                person_id,
                0
            )
        )

    # --------------------------------------------------
    # MAX NETWORK IMPORTANCE
    # --------------------------------------------------

    if case_network_scores:

        max_network_importance = max(
            case_network_scores
        )

        avg_network_importance = sum(
            case_network_scores
        ) / len(case_network_scores)

    else:

        max_network_importance = 0.0
        avg_network_importance = 0.0

    # --------------------------------------------------
    # GRAPH CONNECTION COUNT
    # --------------------------------------------------

    graph_connections = 0

    for entity_id in entity_ids:

        if entity_id in G:

            graph_connections += G.degree(
                entity_id
            )

    # --------------------------------------------------
    # CREATE RESULT
    # --------------------------------------------------

    results.append({

        "case_id": case_id,

        "crime_type": case["crime_type"],

        "status": case["status"],

        "location_id": case["location_id"],

        "date": case["date"],

        "entity_count": entity_count,

        "person_count": len(person_ids),

        "phone_count": len(phone_ids),

        "vehicle_count": len(vehicle_ids),

        "location_count": len(location_ids),

        "graph_connections":
            graph_connections,

        "max_network_importance":
            max_network_importance,

        "avg_network_importance":
            avg_network_importance,

        "case_anomaly_count":
            case_anomaly_count
    })

# --------------------------------------------------
# SAVE
# --------------------------------------------------

index_df = pd.DataFrame(results)

index_df = index_df.sort_values(
    [
        "case_anomaly_count",
        "max_network_importance",
        "graph_connections"
    ],
    ascending=False
)

index_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\nCases indexed:", len(index_df))

print("\nTop cases by network/anomaly evidence:")

print(
    index_df.head(15).to_string(
        index=False
    )
)

print("\nSaved to:")
print(OUTPUT_FILE.resolve())

print("\n" + "=" * 70)
print("CASE NETWORK INDEX COMPLETED")
print("=" * 70)