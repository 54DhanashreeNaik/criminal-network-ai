import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("cleaned_data")

GRAPH_FILE = DATA_DIR / "network_graph.graphml"
OUTPUT_FILE = DATA_DIR / "network_analysis.csv"

print("=" * 70)
print("CRIMINAL NETWORK AI - NETWORK ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# LOAD GRAPH
# --------------------------------------------------

G = nx.read_graphml(GRAPH_FILE)

print(f"\nGraph loaded")
print(f"Nodes : {G.number_of_nodes()}")
print(f"Edges : {G.number_of_edges()}")

# --------------------------------------------------
# CREATE SIMPLE GRAPH FOR CENTRALITY
# --------------------------------------------------

# MultiDiGraph can contain multiple edges between
# the same nodes, so convert it to an undirected graph
# for structural analysis.

UG = nx.Graph()

for node, data in G.nodes(data=True):
    UG.add_node(node, **data)

for source, target in G.edges():

    if source != target:
        UG.add_edge(source, target)

# --------------------------------------------------
# DEGREE CENTRALITY
# --------------------------------------------------

print("\nCalculating degree centrality...")

degree_centrality = nx.degree_centrality(UG)

# --------------------------------------------------
# BETWEENNESS CENTRALITY
# --------------------------------------------------

print("Calculating betweenness centrality...")

betweenness_centrality = nx.betweenness_centrality(
    UG,
    normalized=True
)

# --------------------------------------------------
# PAGERANK
# --------------------------------------------------

print("Calculating PageRank...")

pagerank = nx.pagerank(UG)

# --------------------------------------------------
# COMMUNITY DETECTION
# --------------------------------------------------

print("Detecting communities...")

communities = nx.community.greedy_modularity_communities(UG)

community_map = {}

for community_id, community in enumerate(communities):

    for node in community:

        community_map[node] = community_id

# --------------------------------------------------
# BUILD ANALYSIS TABLE
# --------------------------------------------------

results = []

for node, data in UG.nodes(data=True):

    results.append({
        "node_id": node,
        "node_type": data.get("node_type", "UNKNOWN"),
        "label": data.get("label", node),

        "degree": UG.degree(node),

        "degree_centrality": degree_centrality.get(
            node, 0
        ),

        "betweenness_centrality": betweenness_centrality.get(
            node, 0
        ),

        "pagerank": pagerank.get(
            node, 0
        ),

        "community_id": community_map.get(
            node, -1
        )
    })

analysis_df = pd.DataFrame(results)

# --------------------------------------------------
# NETWORK IMPORTANCE SCORE
# --------------------------------------------------

analysis_df["network_importance_score"] = (
    0.40 * analysis_df["degree_centrality"]
    + 0.35 * analysis_df["betweenness_centrality"]
    + 0.25 * analysis_df["pagerank"]
)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

analysis_df = analysis_df.sort_values(
    "network_importance_score",
    ascending=False
)

analysis_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("NETWORK ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nTotal nodes      : {len(analysis_df)}")
print(f"Total communities: {analysis_df['community_id'].nunique()}")

print("\nTop 10 structurally important nodes:")

print(
    analysis_df[
        [
            "node_id",
            "node_type",
            "label",
            "degree",
            "degree_centrality",
            "betweenness_centrality",
            "pagerank",
            "community_id",
            "network_importance_score"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# --------------------------------------------------
# TOP PEOPLE ONLY
# --------------------------------------------------

top_people = analysis_df[
    analysis_df["node_type"] == "PERSON"
].head(10)

print("\nTop 10 PERSON nodes:")

print(
    top_people[
        [
            "node_id",
            "label",
            "degree",
            "degree_centrality",
            "betweenness_centrality",
            "pagerank",
            "community_id",
            "network_importance_score"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:")
print(OUTPUT_FILE.resolve())

print("\n" + "=" * 70)