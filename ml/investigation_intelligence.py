import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("cleaned_data")

GRAPH_FILE = DATA_DIR / "network_graph.graphml"
NETWORK_FILE = DATA_DIR / "network_analysis.csv"
ANOMALY_FILE = DATA_DIR / "anomaly_results.csv"

OUTPUT_FILE = DATA_DIR / "investigation_profiles.csv"


print("=" * 70)
print("CRIMINAL NETWORK AI - INVESTIGATION INTELLIGENCE")
print("=" * 70)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

G = nx.read_graphml(GRAPH_FILE)

network_df = pd.read_csv(NETWORK_FILE)

anomaly_df = pd.read_csv(ANOMALY_FILE)


# --------------------------------------------------
# ONLY PERSONS
# --------------------------------------------------

persons = network_df[
    network_df["node_type"] == "PERSON"
].copy()

print(f"\nPersons analyzed: {len(persons)}")


# --------------------------------------------------
# CREATE INVESTIGATION PROFILES
# --------------------------------------------------

profiles = []


for _, person in persons.iterrows():

    person_id = person["node_id"]

    # ----------------------------------------------
    # BASIC NETWORK INFORMATION
    # ----------------------------------------------

    network_importance = float(
        person["network_importance_score"]
    )

    degree = int(person["degree"])

    betweenness = float(
        person["betweenness_centrality"]
    )

    pagerank = float(
        person["pagerank"]
    )

    community_id = int(
        person["community_id"]
    )


    # ----------------------------------------------
    # GET NEIGHBORS
    # ----------------------------------------------

    neighbors = list(
        G.neighbors(person_id)
    )


    # ----------------------------------------------
    # COUNT CONNECTED ENTITY TYPES
    # ----------------------------------------------

    connected_people = 0
    connected_phones = 0
    connected_vehicles = 0
    connected_locations = 0
    connected_organizations = 0

    for neighbor in neighbors:

        node_data = G.nodes[neighbor]

        node_type = node_data.get(
            "node_type",
            "UNKNOWN"
        )

        if node_type == "PERSON":
            connected_people += 1

        elif node_type == "PHONE":
            connected_phones += 1

        elif node_type == "VEHICLE":
            connected_vehicles += 1

        elif node_type == "LOCATION":
            connected_locations += 1

        elif node_type == "ORGANIZATION":
            connected_organizations += 1


    # ----------------------------------------------
    # ANOMALY HISTORY
    # ----------------------------------------------

    person_activity = anomaly_df[
        anomaly_df["person_id"] == person_id
    ]

    anomaly_records = person_activity[
        person_activity["predicted_anomaly"] == 1
    ]

    anomaly_count = len(anomaly_records)

    total_activity_records = len(
        person_activity
    )


    # ----------------------------------------------
    # MAX ANOMALY SCORE
    # ----------------------------------------------

    if anomaly_count > 0:

        max_anomaly_score = float(
            anomaly_records["anomaly_score"].max()
        )

    else:

        max_anomaly_score = 0.0


    # ----------------------------------------------
    # ANOMALY RATE
    # ----------------------------------------------

    if total_activity_records > 0:

        anomaly_rate = (
            anomaly_count /
            total_activity_records
        )

    else:

        anomaly_rate = 0.0


    # ----------------------------------------------
    # INVESTIGATION INDICATOR
    # ----------------------------------------------

    investigation_indicator = (
        0.45 * network_importance
        + 0.35 * anomaly_rate
        + 0.20 * max_anomaly_score
    )


    # ----------------------------------------------
    # CREATE PROFILE
    # ----------------------------------------------

    profiles.append({

        "person_id": person_id,

        "name": person["label"],

        "degree": degree,

        "degree_centrality":
            person["degree_centrality"],

        "betweenness_centrality":
            betweenness,

        "pagerank":
            pagerank,

        "network_importance_score":
            network_importance,

        "community_id":
            community_id,

        "connected_people":
            connected_people,

        "connected_phones":
            connected_phones,

        "connected_vehicles":
            connected_vehicles,

        "connected_locations":
            connected_locations,

        "connected_organizations":
            connected_organizations,

        "total_activity_records":
            total_activity_records,

        "anomaly_count":
            anomaly_count,

        "anomaly_rate":
            anomaly_rate,

        "max_anomaly_score":
            max_anomaly_score,

        "investigation_indicator":
            investigation_indicator
    })


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

profiles_df = pd.DataFrame(
    profiles
)


# --------------------------------------------------
# SORT
# --------------------------------------------------

profiles_df = profiles_df.sort_values(
    "investigation_indicator",
    ascending=False
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

profiles_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP INVESTIGATION PROFILES")
print("=" * 70)

print(
    profiles_df[
        [
            "person_id",
            "name",
            "degree",
            "network_importance_score",
            "community_id",
            "connected_people",
            "connected_phones",
            "connected_vehicles",
            "connected_locations",
            "anomaly_count",
            "anomaly_rate",
            "max_anomaly_score",
            "investigation_indicator"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

print("\nProfiles saved to:")
print(OUTPUT_FILE.resolve())

print("\n" + "=" * 70)
print("INVESTIGATION INTELLIGENCE COMPLETED")
print("=" * 70)