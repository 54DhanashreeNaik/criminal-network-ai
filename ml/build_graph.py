import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("cleaned_data")
OUTPUT_DIR = Path("cleaned_data")

print("=" * 70)
print("CRIMINAL NETWORK AI - KNOWLEDGE GRAPH")
print("=" * 70)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

relationships = pd.read_csv(
    DATA_DIR / "relationships_clean.csv"
)

persons = pd.read_csv(
    DATA_DIR / "persons_clean.csv"
)

phones = pd.read_csv(
    DATA_DIR / "phones_clean.csv"
)

vehicles = pd.read_csv(
    DATA_DIR / "vehicles_clean.csv"
)

locations = pd.read_csv(
    DATA_DIR / "locations_clean.csv"
)

organizations = pd.read_csv(
    DATA_DIR / "organizations_clean.csv"
)

# --------------------------------------------------
# CREATE GRAPH
# --------------------------------------------------

G = nx.MultiDiGraph()

# --------------------------------------------------
# ADD PERSON NODES
# --------------------------------------------------

for _, row in persons.iterrows():

    G.add_node(
        row["person_id"],
        node_type="PERSON",
        label=row["name"],
        occupation=row["occupation"],
        age=int(row["age"])
    )

# --------------------------------------------------
# ADD PHONE NODES
# --------------------------------------------------

for _, row in phones.iterrows():

    phone_id = row["phone_id"]

    G.add_node(
        phone_id,
        node_type="PHONE",
        label=str(row["phone_number"])
    )

    # Phone ownership
    G.add_edge(
        row["owner_person_id"],
        phone_id,
        relationship="OWNS"
    )

# --------------------------------------------------
# ADD VEHICLE NODES
# --------------------------------------------------

for _, row in vehicles.iterrows():

    vehicle_id = row["vehicle_id"]

    G.add_node(
        vehicle_id,
        node_type="VEHICLE",
        label=row["registration"],
        vehicle_type=row["vehicle_type"]
    )

    # Vehicle ownership
    G.add_edge(
        row["owner_person_id"],
        vehicle_id,
        relationship="OWNS"
    )

# --------------------------------------------------
# ADD LOCATION NODES
# --------------------------------------------------

for _, row in locations.iterrows():

    location_id = row["location_id"]

    G.add_node(
        location_id,
        node_type="LOCATION",
        label=row["location_name"],
        city=row["city"],
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"])
    )

# --------------------------------------------------
# ADD ORGANIZATION NODES
# --------------------------------------------------

for _, row in organizations.iterrows():

    organization_id = row["organization_id"]

    G.add_node(
        organization_id,
        node_type="ORGANIZATION",
        label=row["organization_name"],
        organization_type=row["organization_type"]
    )

# --------------------------------------------------
# ADD EXISTING RELATIONSHIPS
# --------------------------------------------------

valid_edges = 0
invalid_edges = 0

for _, row in relationships.iterrows():

    source = row["source_id"]
    target = row["target_id"]

    if source in G.nodes and target in G.nodes:

        G.add_edge(
            source,
            target,
            relationship=row["relationship_type"],
            timestamp=str(row["timestamp"])
        )

        valid_edges += 1

    else:

        invalid_edges += 1

# --------------------------------------------------
# ADD HOME LOCATION RELATIONSHIPS
# --------------------------------------------------

for _, row in persons.iterrows():

    person_id = row["person_id"]
    location_id = row["home_location_id"]

    if location_id in G.nodes:

        G.add_edge(
            person_id,
            location_id,
            relationship="LIVES_AT"
        )

# --------------------------------------------------
# GRAPH STATISTICS
# --------------------------------------------------

print("\nGraph created successfully.")

print(f"Nodes : {G.number_of_nodes()}")
print(f"Edges : {G.number_of_edges()}")

print("\nNodes by type:")

node_types = {}

for node, data in G.nodes(data=True):

    node_type = data.get("node_type", "UNKNOWN")

    node_types[node_type] = (
        node_types.get(node_type, 0) + 1
    )

for node_type, count in node_types.items():

    print(f"{node_type:<15} {count}")

print("\nRelationship edges from source dataset:")
print(f"Valid edges   : {valid_edges}")
print(f"Invalid edges : {invalid_edges}")

# --------------------------------------------------
# SAVE GRAPH
# --------------------------------------------------

graph_file = OUTPUT_DIR / "network_graph.graphml"

nx.write_graphml(G, graph_file)

print("\nGraph saved to:")
print(graph_file.resolve())

print("\n" + "=" * 70)
print("KNOWLEDGE GRAPH COMPLETED")
print("=" * 70)