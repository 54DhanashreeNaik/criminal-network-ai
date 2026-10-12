import pandas as pd
from pathlib import Path

DATA_DIR = Path("cleaned_data")

ENTITIES_FILE = DATA_DIR / "extracted_entities.csv"
PERSONS_FILE = DATA_DIR / "persons_clean.csv"
PHONES_FILE = DATA_DIR / "phones_clean.csv"
VEHICLES_FILE = DATA_DIR / "vehicles_clean.csv"
LOCATIONS_FILE = DATA_DIR / "locations_clean.csv"
ORGANIZATIONS_FILE = DATA_DIR / "organizations_clean.csv"

OUTPUT_FILE = DATA_DIR / "resolved_entities.csv"

print("=" * 70)
print("CRIMINAL NETWORK AI - ENTITY RESOLUTION")
print("=" * 70)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

entities = pd.read_csv(ENTITIES_FILE)
persons = pd.read_csv(PERSONS_FILE)
phones = pd.read_csv(PHONES_FILE)
vehicles = pd.read_csv(VEHICLES_FILE)
locations = pd.read_csv(LOCATIONS_FILE)
organizations = pd.read_csv(ORGANIZATIONS_FILE)


# --------------------------------------------------
# CREATE LOOKUP TABLES
# --------------------------------------------------

person_lookup = {
    str(row["name"]).strip().lower(): row["person_id"]
    for _, row in persons.iterrows()
}

phone_lookup = {
    str(row["phone_number"]).strip(): row["phone_id"]
    for _, row in phones.iterrows()
}

vehicle_lookup = {
    str(row["registration"]).strip().upper(): row["vehicle_id"]
    for _, row in vehicles.iterrows()
}

location_lookup = {
    str(row["location_name"]).strip().lower(): row["location_id"]
    for _, row in locations.iterrows()
}

organization_lookup = {
    str(row["organization_name"]).strip().lower():
        row["organization_id"]
    for _, row in organizations.iterrows()
}


# --------------------------------------------------
# RESOLVE ENTITIES
# --------------------------------------------------

resolved_rows = []

resolved_count = 0
unresolved_count = 0


for _, row in entities.iterrows():

    text = str(row["entity_text"]).strip()
    entity_type = str(row["entity_type"]).upper()

    canonical_id = None

    # ----------------------------------------------
    # PERSON
    # ----------------------------------------------

    if entity_type == "PERSON":

        canonical_id = person_lookup.get(
            text.lower()
        )

    # ----------------------------------------------
    # PHONE
    # ----------------------------------------------

    elif entity_type == "PHONE":

        canonical_id = phone_lookup.get(
            text
        )

    # ----------------------------------------------
    # VEHICLE
    # ----------------------------------------------

    elif entity_type == "VEHICLE":

        canonical_id = vehicle_lookup.get(
            text.upper()
        )

    # ----------------------------------------------
    # LOCATION
    # ----------------------------------------------

    elif entity_type == "LOCATION":

        canonical_id = location_lookup.get(
            text.lower()
        )

    # ----------------------------------------------
    # ORGANIZATION
    # ----------------------------------------------

    elif entity_type == "ORGANIZATION":

        canonical_id = organization_lookup.get(
            text.lower()
        )


    # ----------------------------------------------
    # RESOLUTION STATUS
    # ----------------------------------------------

    if canonical_id is not None:

        resolution_status = "RESOLVED"
        resolved_count += 1

    else:

        resolution_status = "UNRESOLVED"
        unresolved_count += 1


    resolved_rows.append({

        "report_id": row["report_id"],

        "case_id": row["case_id"],

        "entity_text": text,

        "entity_type": entity_type,

        "source": row["source"],

        "canonical_id": canonical_id,

        "resolution_status": resolution_status
    })


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

resolved_df = pd.DataFrame(
    resolved_rows
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

resolved_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nTotal extracted entities :", len(entities))
print("Resolved entities        :", resolved_count)
print("Unresolved entities      :", unresolved_count)

resolution_rate = (
    resolved_count / len(entities) * 100
    if len(entities) > 0
    else 0
)

print(
    f"Resolution rate          : {resolution_rate:.2f}%"
)


print("\nResolution by entity type:")

summary = (
    resolved_df
    .groupby(
        ["entity_type", "resolution_status"]
    )
    .size()
    .unstack(fill_value=0)
)

print(summary.to_string())


# --------------------------------------------------
# SHOW UNRESOLVED
# --------------------------------------------------

unresolved = resolved_df[
    resolved_df["resolution_status"] == "UNRESOLVED"
]

if len(unresolved) > 0:

    print("\nFirst unresolved entities:")

    print(
        unresolved[
            [
                "entity_text",
                "entity_type"
            ]
        ]
        .drop_duplicates()
        .head(20)
        .to_string(index=False)
    )


# --------------------------------------------------
# SHOW RESOLVED
# --------------------------------------------------

print("\nFirst 20 resolved entities:")

print(
    resolved_df[
        [
            "entity_text",
            "entity_type",
            "canonical_id",
            "resolution_status"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\nSaved to:")
print(OUTPUT_FILE.resolve())

print("\n" + "=" * 70)
print("ENTITY RESOLUTION COMPLETED")
print("=" * 70)