import pandas as pd
from pathlib import Path

DATA_DIR = Path("cleaned_data")

CASES_FILE = DATA_DIR / "cases_clean.csv"
REPORTS_FILE = DATA_DIR / "crime_reports_clean.csv"
ENTITIES_FILE = DATA_DIR / "extracted_entities.csv"
PROFILES_FILE = DATA_DIR / "investigation_profiles.csv"

OUTPUT_FILE = DATA_DIR / "case_intelligence.csv"


print("=" * 70)
print("CRIMINAL NETWORK AI - CASE INTELLIGENCE")
print("=" * 70)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

cases = pd.read_csv(CASES_FILE)

reports = pd.read_csv(REPORTS_FILE)

entities = pd.read_csv(ENTITIES_FILE)

profiles = pd.read_csv(PROFILES_FILE)


print(f"\nCases loaded   : {len(cases)}")
print(f"Reports loaded : {len(reports)}")
print(f"Entities loaded: {len(entities)}")
print(f"Profiles loaded: {len(profiles)}")


# --------------------------------------------------
# ENTITY → PERSON MAPPING
# --------------------------------------------------

person_entities = entities[
    entities["entity_type"] == "PERSON"
].copy()


# --------------------------------------------------
# CASE INTELLIGENCE
# --------------------------------------------------

case_results = []


for _, case in cases.iterrows():

    case_id = case["case_id"]

    # ----------------------------------------------
    # REPORTS FOR THIS CASE
    # ----------------------------------------------

    case_reports = reports[
        reports["case_id"] == case_id
    ]

    report_count = len(case_reports)


    # ----------------------------------------------
    # ENTITIES FOR THIS CASE
    # ----------------------------------------------

    case_entities = entities[
        entities["case_id"] == case_id
    ]


    # ----------------------------------------------
    # UNIQUE ENTITY COUNTS
    # ----------------------------------------------

    person_count = case_entities[
        case_entities["entity_type"] == "PERSON"
    ]["entity_text"].nunique()

    phone_count = case_entities[
        case_entities["entity_type"] == "PHONE"
    ]["entity_text"].nunique()

    vehicle_count = case_entities[
        case_entities["entity_type"] == "VEHICLE"
    ]["entity_text"].nunique()

    location_count = case_entities[
        case_entities["entity_type"] == "LOCATION"
    ]["entity_text"].nunique()

    organization_count = case_entities[
        case_entities["entity_type"] == "ORGANIZATION"
    ]["entity_text"].nunique()


    # ----------------------------------------------
    # PERSON INVESTIGATION SCORES
    # ----------------------------------------------

    case_person_names = case_entities[
        case_entities["entity_type"] == "PERSON"
    ]["entity_text"].unique()


    case_profiles = profiles[
        profiles["name"].isin(case_person_names)
    ]


    # ----------------------------------------------
    # NETWORK IMPORTANCE
    # ----------------------------------------------

    if len(case_profiles) > 0:

        max_network_importance = (
            case_profiles[
                "network_importance_score"
            ].max()
        )

        max_anomaly_count = (
            case_profiles[
                "anomaly_count"
            ].max()
        )

        max_anomaly_score = (
            case_profiles[
                "max_anomaly_score"
            ].max()
        )

    else:

        max_network_importance = 0

        max_anomaly_count = 0

        max_anomaly_score = 0


    # ----------------------------------------------
    # CASE INTELLIGENCE INDICATOR
    # ----------------------------------------------

    intelligence_indicator = (
        0.30 * max_network_importance
        + 0.40 * max_anomaly_score
        + 0.30 * min(
            person_count / 10,
            1
        )
    )


    # ----------------------------------------------
    # STORE RESULT
    # ----------------------------------------------

    case_results.append({

        "case_id": case_id,

        "crime_type": case["crime_type"],

        "location_id": case["location_id"],

        "date": case["date"],

        "status": case["status"],

        "report_count": report_count,

        "person_count": person_count,

        "phone_count": phone_count,

        "vehicle_count": vehicle_count,

        "location_count": location_count,

        "organization_count": organization_count,

        "max_network_importance":
            max_network_importance,

        "max_anomaly_count":
            max_anomaly_count,

        "max_anomaly_score":
            max_anomaly_score,

        "intelligence_indicator":
            intelligence_indicator
    })


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

case_df = pd.DataFrame(case_results)


# --------------------------------------------------
# SORT
# --------------------------------------------------

case_df = case_df.sort_values(
    "intelligence_indicator",
    ascending=False
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

case_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP CASE INTELLIGENCE RESULTS")
print("=" * 70)

print(
    case_df[
        [
            "case_id",
            "crime_type",
            "location_id",
            "status",
            "report_count",
            "person_count",
            "phone_count",
            "vehicle_count",
            "max_network_importance",
            "max_anomaly_count",
            "max_anomaly_score",
            "intelligence_indicator"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print("\nResults saved to:")
print(OUTPUT_FILE.resolve())


print("\n" + "=" * 70)
print("CASE INTELLIGENCE COMPLETED")
print("=" * 70)