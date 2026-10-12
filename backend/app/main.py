from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="Criminal Network Intelligence API",
    description="Backend API for AI-powered investigation intelligence",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# DATA PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CASE_PRIORITY_FILE = (
    BASE_DIR / "cleaned_data" / "case_priority.csv"
)


# --------------------------------------------------
# LOAD CASE DATA
# --------------------------------------------------

def load_case_data():
    if not CASE_PRIORITY_FILE.exists():
        raise FileNotFoundError(
            f"Case priority file not found: {CASE_PRIORITY_FILE}"
        )

    df = pd.read_csv(CASE_PRIORITY_FILE)

    # Convert missing values to None-compatible values
    df = df.where(pd.notnull(df), None)

    return df


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Criminal Network Intelligence API",
        "version": "1.0.0",
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "data_source": "case_priority.csv",
    }


# --------------------------------------------------
# GET ALL CASES
# --------------------------------------------------

@app.get("/api/cases")
def get_cases():
    df = load_case_data()

    cases = df.to_dict(orient="records")

    return {
        "count": len(cases),
        "cases": cases,
    }


# --------------------------------------------------
# GET SINGLE CASE
# --------------------------------------------------

@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    df = load_case_data()

    case = df[
        df["case_id"].astype(str).str.upper()
        == case_id.upper()
    ]

    if case.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Case {case_id} not found",
        )

    result = case.iloc[0].to_dict()

    return result

# --------------------------------------------------
# NETWORK ANALYSIS
# --------------------------------------------------

NETWORK_ANALYSIS_FILE = (
    BASE_DIR / "cleaned_data" / "network_analysis.csv"
)


def load_network_analysis():
    if not NETWORK_ANALYSIS_FILE.exists():
        raise FileNotFoundError(
            f"Network analysis file not found: {NETWORK_ANALYSIS_FILE}"
        )

    df = pd.read_csv(NETWORK_ANALYSIS_FILE)

    df = df.where(pd.notnull(df), None)

    return df


@app.get("/api/network/summary")
def get_network_summary():
    df = load_network_analysis()

    return {
        "nodes": len(df),
        "edges": 660,
        "communities": int(df["community_id"].nunique()),
        "persons": int(
            (df["node_type"] == "PERSON").sum()
        ),
    }


@app.get("/api/network/entities")
def get_network_entities():
    df = load_network_analysis()

    entities = df.to_dict(orient="records")

    return {
        "count": len(entities),
        "entities": entities,
    }


@app.get("/api/network/entities/{entity_id}")
def get_network_entity(entity_id: str):
    df = load_network_analysis()

    entity = df[
        df["node_id"].astype(str).str.upper()
        == entity_id.upper()
    ]

    if entity.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Entity {entity_id} not found",
        )

    return entity.iloc[0].to_dict()

@app.get("/api/network/entities/{entity_id}/connections")
def get_entity_connections(entity_id: str):
    import networkx as nx

    GRAPH_FILE = BASE_DIR / "cleaned_data" / "network_graph.graphml"

    if not GRAPH_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Network graph file not found"
        )

    graph = nx.read_graphml(GRAPH_FILE)

    entity_id_upper = entity_id.upper()

    # Find the requested entity
    matching_node = None

    for node in graph.nodes:
        if str(node).upper() == entity_id_upper:
            matching_node = node
            break

    if matching_node is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity {entity_id} not found"
        )

    connections = []

    # MultiDiGraph can contain multiple edges
    for source, target, data in graph.edges(
        matching_node,
        data=True
    ):
        target_data = graph.nodes[target]

        connections.append({
            "entity_id": str(target),
            "label": target_data.get(
                "label",
                str(target)
            ),
            "node_type": target_data.get(
                "node_type",
                target_data.get("type", "UNKNOWN")
            ),
            "relationship": data.get(
                "relationship",
                data.get("relation", "CONNECTED")
            )
        })

    # Also check incoming relationships
    for source, target, data in graph.in_edges(
        matching_node,
        data=True
    ):
        source_data = graph.nodes[source]

        connections.append({
            "entity_id": str(source),
            "label": source_data.get(
                "label",
                str(source)
            ),
            "node_type": source_data.get(
                "node_type",
                source_data.get("type", "UNKNOWN")
            ),
            "relationship": data.get(
                "relationship",
                data.get("relation", "CONNECTED")
            )
        })

    # Remove duplicate connections
    unique_connections = {}

    for connection in connections:
        key = (
            connection["entity_id"],
            connection["relationship"]
        )

        unique_connections[key] = connection

    connections = list(unique_connections.values())

    return {
        "entity_id": entity_id_upper,
        "count": len(connections),
        "connections": connections
    }

# --------------------------------------------------
# ALERTS / SUSPICIOUS ACTIVITY
# --------------------------------------------------

INVESTIGATION_PROFILES_FILE = (
    BASE_DIR / "cleaned_data" / "investigation_profiles.csv"
)


def load_investigation_profiles():
    if not INVESTIGATION_PROFILES_FILE.exists():
        raise FileNotFoundError(
            f"Investigation profiles file not found: "
            f"{INVESTIGATION_PROFILES_FILE}"
        )

    df = pd.read_csv(INVESTIGATION_PROFILES_FILE)

    df = df.where(pd.notnull(df), None)

    return df


@app.get("/api/alerts")
def get_alerts():
    df = load_investigation_profiles()

    alerts = []

    for _, row in df.iterrows():

        anomaly_count = int(
            row.get("anomaly_count", 0)
        )

        anomaly_rate = float(
            row.get("anomaly_rate", 0)
        )

        max_anomaly_score = float(
            row.get("max_anomaly_score", 0)
        )

        network_importance = float(
            row.get("network_importance_score", 0)
        )

        investigation_indicator = float(
            row.get("investigation_indicator", 0)
        )

        # --------------------------------------------------
        # SEVERITY
        # --------------------------------------------------

        if anomaly_count >= 2 or max_anomaly_score >= 0.20:
            severity = "HIGH"

        elif anomaly_count >= 1 or max_anomaly_score >= 0.10:
            severity = "MEDIUM"

        else:
            severity = "LOW"

        # --------------------------------------------------
        # REASON
        # --------------------------------------------------

        reasons = []

        if anomaly_count > 0:
            reasons.append(
                f"{anomaly_count} anomalous activity record(s)"
            )

        if max_anomaly_score >= 0.10:
            reasons.append(
                "elevated anomaly score"
            )

        if network_importance >= 0.03:
            reasons.append(
                "high network importance"
            )

        if not reasons:
            reasons.append(
                "structural network activity detected"
            )

        reason = ", ".join(reasons)

        alerts.append({
            "entity_id": str(
                row.get("person_id", "")
            ),

            "entity_name": str(
                row.get("name", "")
            ),

            "severity": severity,

            "anomaly_count": anomaly_count,

            "anomaly_rate": round(
                anomaly_rate,
                4
            ),

            "max_anomaly_score": round(
                max_anomaly_score,
                4
            ),

            "network_importance_score": round(
                network_importance,
                4
            ),

            "investigation_indicator": round(
                investigation_indicator,
                4
            ),

            "reason": reason,
        })

    # --------------------------------------------------
    # SORT ALERTS
    # --------------------------------------------------

    severity_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    alerts.sort(
        key=lambda x: (
            severity_order[x["severity"]],
            -x["max_anomaly_score"],
            -x["network_importance_score"],
        )
    )

    return {
        "count": len(alerts),
        "alerts": alerts,
    }

# --------------------------------------------------
# CASE NETWORK / EVIDENCE GRAPH
# --------------------------------------------------

CASE_EVIDENCE_GRAPH_FILE = (
    BASE_DIR / "cleaned_data" / "case_evidence_graph.graphml"
)


@app.get("/api/cases/{case_id}/network")
def get_case_network(case_id: str):
    import networkx as nx

    if not CASE_EVIDENCE_GRAPH_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Case evidence graph file not found"
        )

    graph = nx.read_graphml(
        CASE_EVIDENCE_GRAPH_FILE
    )

    case_id_upper = case_id.upper()

    # Find case node
    matching_case = None

    for node in graph.nodes:
        if str(node).upper() == case_id_upper:
            matching_case = node
            break

    if matching_case is None:
        raise HTTPException(
            status_code=404,
            detail=f"Case {case_id} not found in evidence graph"
        )

    nodes = []
    edges = []

    # --------------------------------------------------
    # CASE NODE
    # --------------------------------------------------

    case_data = graph.nodes[matching_case]

    nodes.append({
        "id": str(matching_case),
        "label": case_data.get(
            "label",
            str(matching_case)
        ),
        "node_type": "CASE"
    })

    # --------------------------------------------------
    # CONNECTED NODES
    # --------------------------------------------------

    connected_nodes = set()

    for source, target, data in graph.edges(
        matching_case,
        data=True
    ):
        connected_nodes.add(target)

        edges.append({
            "source": str(source),
            "target": str(target),
            "relationship": data.get(
                "relationship",
                data.get("relation", "CONNECTED")
            )
        })

    for source, target, data in graph.in_edges(
        matching_case,
        data=True
    ):
        connected_nodes.add(source)

        edges.append({
            "source": str(source),
            "target": str(target),
            "relationship": data.get(
                "relationship",
                data.get("relation", "CONNECTED")
            )
        })

    # --------------------------------------------------
    # NODE DETAILS
    # --------------------------------------------------

    for node in connected_nodes:

        node_data = graph.nodes[node]

        nodes.append({
            "id": str(node),

            "label": node_data.get(
                "label",
                str(node)
            ),

            "node_type": node_data.get(
                "node_type",
                node_data.get(
                    "type",
                    "UNKNOWN"
                )
            )
        })

    return {
        "case_id": case_id_upper,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges
    }

# --------------------------------------------------
# DASHBOARD / ANALYTICS
# --------------------------------------------------

@app.get("/api/dashboard")
def get_dashboard():
    import networkx as nx

    # ----------------------------------------------
    # CASE DATA
    # ----------------------------------------------

    if not CASE_PRIORITY_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Case priority file not found"
        )

    cases_df = pd.read_csv(
        CASE_PRIORITY_FILE
    )

    total_cases = len(cases_df)

    high_cases = int(
        (
            cases_df["priority"]
            .astype(str)
            .str.upper() == "HIGH"
        ).sum()
    )

    medium_cases = int(
        (
            cases_df["priority"]
            .astype(str)
            .str.upper() == "MEDIUM"
        ).sum()
    )

    low_cases = int(
        (
            cases_df["priority"]
            .astype(str)
            .str.upper() == "LOW"
        ).sum()
    )

    # ----------------------------------------------
    # CRIME TYPE DISTRIBUTION
    # ----------------------------------------------

    crime_distribution = []

    crime_counts = (
        cases_df["crime_type"]
        .value_counts()
    )

    for crime_type, count in crime_counts.items():
        crime_distribution.append({
            "crime_type": str(crime_type),
            "count": int(count)
        })

    # ----------------------------------------------
    # NETWORK
    # ----------------------------------------------

    GRAPH_FILE = (
        BASE_DIR /
        "cleaned_data" /
        "network_graph.graphml"
    )

    total_entities = 0
    total_connections = 0
    entity_distribution = []
    top_entities = []

    if GRAPH_FILE.exists():

        graph = nx.read_graphml(
            GRAPH_FILE
        )

        total_entities = graph.number_of_nodes()
        total_connections = graph.number_of_edges()

        # ------------------------------------------
        # ENTITY TYPES
        # ------------------------------------------

        type_counts = {}

        for node in graph.nodes:

            data = graph.nodes[node]

            node_type = data.get(
                "node_type",
                data.get(
                    "type",
                    "UNKNOWN"
                )
            )

            node_type = str(
                node_type
            ).upper()

            type_counts[node_type] = (
                type_counts.get(
                    node_type,
                    0
                ) + 1
            )

        for node_type, count in sorted(
            type_counts.items(),
            key=lambda x: -x[1]
        ):
            entity_distribution.append({
                "node_type": node_type,
                "count": int(count)
            })

    # ----------------------------------------------
    # NETWORK ANALYSIS
    # ----------------------------------------------

    NETWORK_FILE = (
        BASE_DIR /
        "cleaned_data" /
        "network_analysis.csv"
    )

    if NETWORK_FILE.exists():

        network_df = pd.read_csv(
            NETWORK_FILE
        )

        network_df = network_df.sort_values(
            "network_importance_score",
            ascending=False
        )

        for _, row in network_df.head(10).iterrows():

            top_entities.append({
                "entity_id": str(
                    row.get(
                        "node_id",
                        ""
                    )
                ),

                "name": str(
                    row.get(
                        "label",
                        row.get(
                            "node_id",
                            ""
                        )
                    )
                ),

                "node_type": str(
                    row.get(
                        "node_type",
                        "UNKNOWN"
                    )
                ),

                "importance": round(
                    float(
                        row.get(
                            "network_importance_score",
                            0
                        )
                    ),
                    4
                )
            })

    # ----------------------------------------------
    # ALERTS
    # ----------------------------------------------

    alerts = []

    profiles = load_investigation_profiles()

    for _, row in profiles.iterrows():

        anomaly_count = int(
            row.get(
                "anomaly_count",
                0
            )
        )

        max_anomaly_score = float(
            row.get(
                "max_anomaly_score",
                0
            )
        )

        network_importance = float(
            row.get(
                "network_importance_score",
                0
            )
        )

        if (
            anomaly_count >= 2
            or max_anomaly_score >= 0.20
        ):
            severity = "HIGH"

        elif (
            anomaly_count >= 1
            or max_anomaly_score >= 0.10
        ):
            severity = "MEDIUM"

        else:
            severity = "LOW"

        if severity != "LOW":

            alerts.append({
                "entity_id": str(
                    row.get(
                        "person_id",
                        ""
                    )
                ),

                "entity_name": str(
                    row.get(
                        "name",
                        ""
                    )
                ),

                "severity": severity,

                "anomaly_count":
                    anomaly_count,

                "max_anomaly_score":
                    round(
                        max_anomaly_score,
                        4
                    ),

                "network_importance_score":
                    round(
                        network_importance,
                        4
                    )
            })

    severity_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2
    }

    alerts.sort(
        key=lambda x: (
            severity_order[
                x["severity"]
            ],
            -x["max_anomaly_score"]
        )
    )

    high_alerts = sum(
        1
        for alert in alerts
        if alert["severity"] == "HIGH"
    )

    medium_alerts = sum(
        1
        for alert in alerts
        if alert["severity"] == "MEDIUM"
    )

    # ----------------------------------------------
    # RESPONSE
    # ----------------------------------------------

    return {
        "summary": {
            "total_cases": total_cases,
            "high_priority_cases":
                high_cases,
            "medium_priority_cases":
                medium_cases,
            "low_priority_cases":
                low_cases,

            "total_entities":
                total_entities,

            "total_connections":
                total_connections,

            "high_alerts":
                high_alerts,

            "medium_alerts":
                medium_alerts
        },

        "crime_distribution":
            crime_distribution,

        "entity_distribution":
            entity_distribution,

        "top_entities":
            top_entities,

        "recent_alerts":
            alerts[:10]
    }

# --------------------------------------------------
# ENTITY → RELATED CASES
# --------------------------------------------------

# --------------------------------------------------
# ENTITY → RELATED CASES
# --------------------------------------------------

@app.get("/api/network/entities/{entity_id}/cases")
def get_entity_related_cases(entity_id: str):

    entity_id_upper = entity_id.upper()

    # ----------------------------------------------
    # LOAD PERSON DATA
    # ----------------------------------------------

    PERSONS_FILE = (
        BASE_DIR
        / "cleaned_data"
        / "persons_clean.csv"
    )

    if not PERSONS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Persons data not found"
        )

    persons_df = pd.read_csv(
        PERSONS_FILE
    )

    # ----------------------------------------------
    # FIND ENTITY NAME
    # ----------------------------------------------

    person_match = persons_df[
        persons_df["person_id"]
        .astype(str)
        .str.upper()
        == entity_id_upper
    ]

    if person_match.empty:

        raise HTTPException(
            status_code=404,
            detail=f"Person {entity_id} not found"
        )

    entity_name = str(
        person_match.iloc[0]["name"]
    )

    # ----------------------------------------------
    # LOAD EXTRACTED ENTITIES
    # ----------------------------------------------

    ENTITIES_FILE = (
        BASE_DIR
        / "cleaned_data"
        / "extracted_entities.csv"
    )

    if not ENTITIES_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Extracted entities file not found"
        )

    entities_df = pd.read_csv(
        ENTITIES_FILE
    )

    # ----------------------------------------------
    # FIND CASES WHERE ENTITY NAME APPEARS
    # ----------------------------------------------

    entity_matches = entities_df[
        (
            entities_df["entity_type"]
            .astype(str)
            .str.upper()
            == "PERSON"
        )
        &
        (
            entities_df["entity_text"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            entity_name.strip().lower()
        )
    ]

    related_case_ids = sorted(
        entity_matches["case_id"]
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    # ----------------------------------------------
    # LOAD CASE PRIORITY DATA
    # ----------------------------------------------

    if not CASE_PRIORITY_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Case priority file not found"
        )

    cases_df = pd.read_csv(
        CASE_PRIORITY_FILE
    )

    # ----------------------------------------------
    # BUILD CASE RESPONSE
    # ----------------------------------------------

    related_cases = []

    for case_id in related_case_ids:

        matches = cases_df[
            cases_df["case_id"]
            .astype(str)
            .str.upper()
            ==
            case_id
        ]

        if matches.empty:
            continue

        row = matches.iloc[0]

        related_cases.append({

            "case_id": str(
                row["case_id"]
            ),

            "crime_type": str(
                row["crime_type"]
            ),

            "status": str(
                row["status"]
            ),

            "priority": str(
                row["priority"]
            ),

            "priority_score": round(
                float(
                    row["priority_score"]
                ),
                4
            ),

            "anomaly_count": int(
                row.get(
                    "case_anomaly_count",
                    0
                )
            ),

            "graph_connections": int(
                row.get(
                    "graph_connections",
                    0
                )
            )
        })

    # ----------------------------------------------
    # SORT BY PRIORITY
    # ----------------------------------------------

    related_cases.sort(
        key=lambda x:
        -x["priority_score"]
    )

    # ----------------------------------------------
    # RETURN
    # ----------------------------------------------

    return {
        "entity_id": entity_id_upper,
        "entity_name": entity_name,
        "case_count": len(related_cases),
        "cases": related_cases
    }

    import networkx as nx

    entity_id_upper = entity_id.upper()

    # ----------------------------------------------
    # CASE PRIORITY DATA
    # ----------------------------------------------

    if not CASE_PRIORITY_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Case priority file not found"
        )

    cases_df = pd.read_csv(
        CASE_PRIORITY_FILE
    )

    # ----------------------------------------------
    # CASE EVIDENCE GRAPH
    # ----------------------------------------------

    CASE_EVIDENCE_GRAPH_FILE = (
        BASE_DIR
        / "cleaned_data"
        / "case_evidence_graph.graphml"
    )

    if not CASE_EVIDENCE_GRAPH_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Case evidence graph not found"
        )

    graph = nx.read_graphml(
        CASE_EVIDENCE_GRAPH_FILE
    )

    # ----------------------------------------------
    # VERIFY ENTITY EXISTS
    # ----------------------------------------------

    matching_entity = None

    for node in graph.nodes:

        if str(node).upper() == entity_id_upper:
            matching_entity = node
            break

    if matching_entity is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity {entity_id} not found"
        )

    # ----------------------------------------------
    # FIND CASES WHERE ENTITY IS CASE EVIDENCE
    # ----------------------------------------------

    related_case_ids = set()

    for source, target, data in graph.edges(
        matching_entity,
        data=True
    ):

        relationship_type = str(
            data.get(
                "relationship_type",
                ""
            )
        ).upper()

        if relationship_type == "CASE_EVIDENCE":

            if str(source).upper().startswith("C"):
                related_case_ids.add(
                    str(source).upper()
                )

            if str(target).upper().startswith("C"):
                related_case_ids.add(
                    str(target).upper()
                )

    # ----------------------------------------------
    # CHECK INCOMING EDGES
    # ----------------------------------------------

    for source, target, data in graph.in_edges(
        matching_entity,
        data=True
    ):

        relationship_type = str(
            data.get(
                "relationship_type",
                ""
            )
        ).upper()

        if relationship_type == "CASE_EVIDENCE":

            if str(source).upper().startswith("C"):
                related_case_ids.add(
                    str(source).upper()
                )

            if str(target).upper().startswith("C"):
                related_case_ids.add(
                    str(target).upper()
                )

    # ----------------------------------------------
    # BUILD RESPONSE
    # ----------------------------------------------

    related_cases = []

    for case_id in related_case_ids:

        matches = cases_df[
            cases_df["case_id"]
            .astype(str)
            .str.upper()
            == case_id
        ]

        if matches.empty:
            continue

        row = matches.iloc[0]

        related_cases.append({

            "case_id": str(
                row["case_id"]
            ),

            "crime_type": str(
                row["crime_type"]
            ),

            "status": str(
                row["status"]
            ),

            "priority": str(
                row["priority"]
            ),

            "priority_score": round(
                float(
                    row["priority_score"]
                ),
                4
            ),

            "anomaly_count": int(
                row.get(
                    "case_anomaly_count",
                    0
                )
            ),

            "graph_connections": int(
                row.get(
                    "graph_connections",
                    0
                )
            )
        })

    # ----------------------------------------------
    # SORT HIGHEST PRIORITY FIRST
    # ----------------------------------------------

    related_cases.sort(
        key=lambda x:
        -x["priority_score"]
    )

    # ----------------------------------------------
    # FINAL RESPONSE
    # ----------------------------------------------

    return {
        "entity_id": entity_id_upper,
        "case_count": len(related_cases),
        "cases": related_cases
    }

    entity_id_upper = entity_id.upper()

    # ----------------------------------------------
    # LOAD CASE PRIORITY DATA
    # ----------------------------------------------

    if not CASE_PRIORITY_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Case priority file not found"
        )

    cases_df = pd.read_csv(
        CASE_PRIORITY_FILE
    )

    # ----------------------------------------------
    # LOAD CASE EVIDENCE GRAPH
    # ----------------------------------------------

    import networkx as nx

    CASE_EVIDENCE_GRAPH_FILE = (
        BASE_DIR
        / "cleaned_data"
        / "case_evidence_graph.graphml"
    )

    if not CASE_EVIDENCE_GRAPH_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Case evidence graph not found"
        )

    graph = nx.read_graphml(
        CASE_EVIDENCE_GRAPH_FILE
    )

    # ----------------------------------------------
    # FIND ENTITY NODE
    # ----------------------------------------------

    matching_entity = None

    for node in graph.nodes:

        if str(node).upper() == entity_id_upper:
            matching_entity = node
            break

    if matching_entity is None:

        raise HTTPException(
            status_code=404,
            detail=f"Entity {entity_id} not found"
        )

    # ----------------------------------------------
    # FIND CONNECTED CASES
    # ----------------------------------------------

    related_case_ids = set()

    for source, target, data in graph.edges(
        matching_entity,
        data=True
    ):

        source_str = str(source)
        target_str = str(target)

        if source_str.upper().startswith("C"):
            related_case_ids.add(
                source_str.upper()
            )

        if target_str.upper().startswith("C"):
            related_case_ids.add(
                target_str.upper()
            )

    # Also check incoming edges
    for source, target, data in graph.in_edges(
        matching_entity,
        data=True
    ):

        source_str = str(source)
        target_str = str(target)

        if source_str.upper().startswith("C"):
            related_case_ids.add(
                source_str.upper()
            )

        if target_str.upper().startswith("C"):
            related_case_ids.add(
                target_str.upper()
            )

    # ----------------------------------------------
    # BUILD CASE RESPONSE
    # ----------------------------------------------

    related_cases = []

    for case_id in related_case_ids:

        matches = cases_df[
            cases_df["case_id"]
            .astype(str)
            .str.upper()
            == case_id
        ]

        if matches.empty:
            continue

        row = matches.iloc[0]

        related_cases.append({

            "case_id": str(
                row["case_id"]
            ),

            "crime_type": str(
                row["crime_type"]
            ),

            "status": str(
                row["status"]
            ),

            "priority": str(
                row["priority"]
            ),

            "priority_score": round(
                float(
                    row["priority_score"]
                ),
                4
            ),

            "anomaly_count": int(
                row.get(
                    "case_anomaly_count",
                    0
                )
            ),

            "graph_connections": int(
                row.get(
                    "graph_connections",
                    0
                )
            )
        })

    # ----------------------------------------------
    # SORT BY PRIORITY SCORE
    # ----------------------------------------------

    related_cases.sort(
        key=lambda x:
        -x["priority_score"]
    )

    # ----------------------------------------------
    # RESPONSE
    # ----------------------------------------------

    return {
        "entity_id": entity_id_upper,

        "case_count":
            len(related_cases),

        "cases":
            related_cases
    }