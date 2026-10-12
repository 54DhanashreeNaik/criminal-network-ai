import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("cleaned_data")

CASES_FILE = DATA_DIR / "cases_clean.csv"
RESOLVED_FILE = DATA_DIR / "resolved_entities.csv"
GRAPH_FILE = DATA_DIR / "network_graph.graphml"

OUTPUT_FILE = DATA_DIR / "case_evidence_graph.graphml"

print("=" * 70)
print("CRIMINAL NETWORK AI - CASE EVIDENCE GRAPH")
print("=" * 70)

# LOAD DATA
cases = pd.read_csv(CASES_FILE)
resolved = pd.read_csv(RESOLVED_FILE)

# LOAD EXISTING NETWORK
G = nx.read_graphml(GRAPH_FILE)

print("\nExisting graph:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ADD CASE NODES
for _, case in cases.iterrows():

    case_id = str(case["case_id"])

    G.add_node(
        case_id,
        entity_type="CASE",
        crime_type=str(case["crime_type"]),
        status=str(case["status"]),
        location_id=str(case["location_id"]),
        date=str(case["date"])
    )

# ADD CASE -> ENTITY EDGES
added_edges = 0

for _, row in resolved.iterrows():

    case_id = str(row["case_id"])
    canonical_id = str(row["canonical_id"])
    entity_type = str(row["entity_type"])

    if (
        case_id in G.nodes
        and canonical_id in G.nodes
    ):

        G.add_edge(
            case_id,
            canonical_id,
            relationship_type="CASE_EVIDENCE",
            entity_type=entity_type
        )

        added_edges += 1

# ADD CASE -> LOCATION EDGE
for _, case in cases.iterrows():

    case_id = str(case["case_id"])
    location_id = str(case["location_id"])

    if (
        case_id in G.nodes
        and location_id in G.nodes
    ):

        G.add_edge(
            case_id,
            location_id,
            relationship_type="CASE_LOCATION"
        )

        added_edges += 1

# SAVE
nx.write_graphml(
    G,
    OUTPUT_FILE
)

print("\nCase nodes added:", len(cases))
print("Case evidence edges added:", added_edges)

print("\nFinal graph:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("\nSaved to:")
print(OUTPUT_FILE.resolve())

print("\n" + "=" * 70)
print("CASE EVIDENCE GRAPH COMPLETED")
print("=" * 70)