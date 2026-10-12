import pandas as pd
import spacy
import re
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_DIR = Path("cleaned_data")
OUTPUT_DIR = Path("cleaned_data")

# --------------------------------------------------
# LOAD NLP MODEL
# --------------------------------------------------

nlp = spacy.load("en_core_web_sm")

# --------------------------------------------------
# LOAD REFERENCE DATA
# --------------------------------------------------

persons = pd.read_csv(DATA_DIR / "persons_clean.csv")
locations = pd.read_csv(DATA_DIR / "locations_clean.csv")
organizations = pd.read_csv(DATA_DIR / "organizations_clean.csv")
vehicles = pd.read_csv(DATA_DIR / "vehicles_clean.csv")
phones = pd.read_csv(DATA_DIR / "phones_clean.csv")
reports = pd.read_csv(DATA_DIR / "crime_reports_clean.csv")

# --------------------------------------------------
# REFERENCE SETS
# --------------------------------------------------

person_names = set(
    persons["name"].astype(str).str.strip()
)

location_names = set(
    locations["location_name"].astype(str).str.strip()
)

organization_names = set(
    organizations["organization_name"].astype(str).str.strip()
)

vehicle_registrations = set(
    vehicles["registration"].astype(str).str.strip().str.upper()
)

phone_numbers = set(
    phones["phone_number"].astype(str).str.replace(r"\D", "", regex=True)
)

# --------------------------------------------------
# REGEX PATTERNS
# --------------------------------------------------

PHONE_PATTERN = r"\b\d{10}\b"

VEHICLE_PATTERN = r"\b[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}\b"

# --------------------------------------------------
# EXTRACTION FUNCTION
# --------------------------------------------------

def extract_entities(text):

    entities = []

    text = str(text)

    # ----------------------------------------------
    # spaCy NER
    # ----------------------------------------------

    doc = nlp(text)

    for ent in doc.ents:

        if ent.label_ == "PERSON":

            # Only keep names that exist in our dataset
            if ent.text.strip() in person_names:

                entities.append({
                    "entity_text": ent.text.strip(),
                    "entity_type": "PERSON",
                    "source": "spacy"
                })

    # ----------------------------------------------
    # PERSON DATASET MATCHING
    # ----------------------------------------------

    for name in person_names:

        if name.lower() in text.lower():

            entities.append({
                "entity_text": name,
                "entity_type": "PERSON",
                "source": "dataset_match"
            })

    # ----------------------------------------------
    # PHONE EXTRACTION
    # ----------------------------------------------

    phone_matches = re.findall(PHONE_PATTERN, text)

    for phone in phone_matches:

        if phone in phone_numbers:

            entities.append({
                "entity_text": phone,
                "entity_type": "PHONE",
                "source": "regex"
            })

    # ----------------------------------------------
    # VEHICLE EXTRACTION
    # ----------------------------------------------

    vehicle_matches = re.findall(
        VEHICLE_PATTERN,
        text.upper()
    )

    for vehicle in vehicle_matches:

        if vehicle in vehicle_registrations:

            entities.append({
                "entity_text": vehicle,
                "entity_type": "VEHICLE",
                "source": "regex"
            })

    # ----------------------------------------------
    # LOCATION MATCHING
    # ----------------------------------------------

    for location in location_names:

        if location.lower() in text.lower():

            entities.append({
                "entity_text": location,
                "entity_type": "LOCATION",
                "source": "dataset_match"
            })

    # ----------------------------------------------
    # ORGANIZATION MATCHING
    # ----------------------------------------------

    for organization in organization_names:

        if organization.lower() in text.lower():

            entities.append({
                "entity_text": organization,
                "entity_type": "ORGANIZATION",
                "source": "dataset_match"
            })

    # ----------------------------------------------
    # REMOVE DUPLICATES
    # ----------------------------------------------

    unique_entities = {}

    for entity in entities:

        key = (
            entity["entity_text"].lower(),
            entity["entity_type"]
        )

        unique_entities[key] = entity

    return list(unique_entities.values())


# --------------------------------------------------
# PROCESS REPORTS
# --------------------------------------------------

all_entities = []

print("=" * 70)
print("CRIMINAL NETWORK AI - ENTITY EXTRACTION")
print("=" * 70)

for _, row in reports.iterrows():

    extracted = extract_entities(row["text"])

    for entity in extracted:

        all_entities.append({
            "report_id": row["report_id"],
            "case_id": row["case_id"],
            "entity_text": entity["entity_text"],
            "entity_type": entity["entity_type"],
            "source": entity["source"]
        })


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

entities_df = pd.DataFrame(all_entities)

output_file = OUTPUT_DIR / "extracted_entities.csv"

entities_df.to_csv(
    output_file,
    index=False
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print(f"\nReports processed : {len(reports)}")
print(f"Entities extracted: {len(entities_df)}")

print("\nEntities by type:")

print(
    entities_df["entity_type"]
    .value_counts()
    .to_string()
)

print("\nFirst 20 extracted entities:")

print(
    entities_df.head(20).to_string(index=False)
)

print("\nSaved to:")
print(output_file.resolve())

print("\n" + "=" * 70)
print("ENTITY EXTRACTION COMPLETED")
print("=" * 70)