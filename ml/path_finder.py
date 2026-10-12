import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("cleaned_data")

GRAPH_FILE = DATA_DIR / "network_graph.graphml"

print("=" * 70)
print("CRIMINAL NETWORK AI - INVESTIGATION PATH FINDER")
print("=" * 70)


# --------------------------------------------------
# LOAD GRAPH
# --------------------------------------------------

G = nx.read_graphml(GRAPH_FILE)

print(f"\nGraph loaded")
print(f"Nodes : {G.number_of_nodes()}")
print(f"Edges : {G.number_of_edges()}")


# --------------------------------------------------
# CONVERT TO UNDIRECTED GRAPH
# --------------------------------------------------

UG = nx.Graph(G)


# --------------------------------------------------
# PATH FINDER
# --------------------------------------------------

def find_investigation_path(source_id, target_id):

    print("\n" + "=" * 70)
    print(f"SEARCHING PATH: {source_id} → {target_id}")
    print("=" * 70)

    if source_id not in UG:

        print(f"\nSource node '{source_id}' not found.")

        return

    if target_id not in UG:

        print(f"\nTarget node '{target_id}' not found.")

        return

    try:

        path = nx.shortest_path(
            UG,
            source=source_id,
            target=target_id
        )

    except nx.NetworkXNoPath:

        print("\nNo connection found between these entities.")

        return


    print(f"\nPath length: {len(path) - 1} connections")

    print("\nInvestigation Path:\n")

    for i, node in enumerate(path):

        data = UG.nodes[node]

        node_type = data.get(
            "node_type",
            "UNKNOWN"
        )

        label = data.get(
            "label",
            node
        )

        print(
            f"{i + 1}. [{node_type}] "
            f"{label} ({node})"
        )

        if i < len(path) - 1:

            next_node = path[i + 1]

            edge_data = UG.get_edge_data(
                node,
                next_node
            )

            relationship = "CONNECTED"

            if edge_data:

                # GraphML conversion may contain
                # relationship information in the edge data.

                if isinstance(edge_data, dict):

                    if "relationship" in edge_data:

                        relationship = edge_data[
                            "relationship"
                        ]

                    else:

                        first_edge = next(
                            iter(edge_data.values()),
                            {}
                        )

                        relationship = first_edge.get(
                            "relationship",
                            "CONNECTED"
                        )

            print(
                f"       │\n"
                f"       └── {relationship}\n"
            )


# --------------------------------------------------
# EXAMPLE SEARCH
# --------------------------------------------------

# --------------------------------------------------
# INTERACTIVE SEARCH
# --------------------------------------------------

print("\nAvailable example entities:")
print("P063 - Rohan Khan")
print("P042 - Neeraj Verma")

source_id = input(
    "\nEnter source entity ID: "
).strip()

target_id = input(
    "Enter target entity ID: "
).strip()

find_investigation_path(
    source_id,
    target_id
)


print("\n" + "=" * 70)
print("PATH SEARCH COMPLETED")
print("=" * 70)