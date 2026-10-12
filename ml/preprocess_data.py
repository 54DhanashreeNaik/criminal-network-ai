import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("cleaned_data")

OUTPUT_DIR.mkdir(exist_ok=True)


def load_csv(filename):
    return pd.read_csv(DATA_DIR / filename)


def save_csv(df, filename):
    df.to_csv(OUTPUT_DIR / filename, index=False)


print("=" * 70)
print("CRIMINAL NETWORK AI - DATA PREPROCESSING")
print("=" * 70)


# --------------------------------------------------
# 1. PERSONS
# --------------------------------------------------

persons = load_csv("persons.csv")

persons = persons.drop_duplicates(subset=["person_id"])

persons["person_id"] = persons["person_id"].astype(str).str.strip()
persons["name"] = persons["name"].astype(str).str.strip()
persons["occupation"] = persons["occupation"].astype(str).str.strip()
persons["home_location_id"] = persons["home_location_id"].astype(str).str.strip()

save_csv(persons, "persons_clean.csv")

print(f"Persons: {len(persons)} records")


# --------------------------------------------------
# 2. PHONES
# --------------------------------------------------

phones = load_csv("phones.csv")

phones = phones.drop_duplicates(subset=["phone_id"])

phones["phone_id"] = phones["phone_id"].astype(str).str.strip()
phones["phone_number"] = phones["phone_number"].astype(str).str.replace(
    r"\D", "", regex=True
)
phones["owner_person_id"] = phones["owner_person_id"].astype(str).str.strip()

save_csv(phones, "phones_clean.csv")

print(f"Phones: {len(phones)} records")


# --------------------------------------------------
# 3. VEHICLES
# --------------------------------------------------

vehicles = load_csv("vehicles.csv")

vehicles = vehicles.drop_duplicates(subset=["vehicle_id"])

vehicles["vehicle_id"] = vehicles["vehicle_id"].astype(str).str.strip()
vehicles["registration"] = (
    vehicles["registration"]
    .astype(str)
    .str.upper()
    .str.replace(" ", "", regex=False)
)
vehicles["vehicle_type"] = vehicles["vehicle_type"].astype(str).str.strip()
vehicles["owner_person_id"] = vehicles["owner_person_id"].astype(str).str.strip()

save_csv(vehicles, "vehicles_clean.csv")

print(f"Vehicles: {len(vehicles)} records")


# --------------------------------------------------
# 4. LOCATIONS
# --------------------------------------------------

locations = load_csv("locations.csv")

locations = locations.drop_duplicates(subset=["location_id"])

locations["location_id"] = locations["location_id"].astype(str).str.strip()
locations["location_name"] = locations["location_name"].astype(str).str.strip()
locations["city"] = locations["city"].astype(str).str.strip()

locations["latitude"] = pd.to_numeric(
    locations["latitude"], errors="coerce"
)

locations["longitude"] = pd.to_numeric(
    locations["longitude"], errors="coerce"
)

save_csv(locations, "locations_clean.csv")

print(f"Locations: {len(locations)} records")


# --------------------------------------------------
# 5. ORGANIZATIONS
# --------------------------------------------------

organizations = load_csv("organizations.csv")

organizations = organizations.drop_duplicates(subset=["organization_id"])

organizations["organization_id"] = (
    organizations["organization_id"].astype(str).str.strip()
)

organizations["organization_name"] = (
    organizations["organization_name"].astype(str).str.strip()
)

organizations["organization_type"] = (
    organizations["organization_type"].astype(str).str.strip()
)

save_csv(organizations, "organizations_clean.csv")

print(f"Organizations: {len(organizations)} records")


# --------------------------------------------------
# 6. CASES
# --------------------------------------------------

cases = load_csv("cases.csv")

cases = cases.drop_duplicates(subset=["case_id"])

cases["case_id"] = cases["case_id"].astype(str).str.strip()
cases["crime_type"] = cases["crime_type"].astype(str).str.strip()
cases["location_id"] = cases["location_id"].astype(str).str.strip()
cases["status"] = cases["status"].astype(str).str.strip()

cases["date"] = pd.to_datetime(
    cases["date"], errors="coerce"
)

save_csv(cases, "cases_clean.csv")

print(f"Cases: {len(cases)} records")


# --------------------------------------------------
# 7. RELATIONSHIPS
# --------------------------------------------------

relationships = load_csv("relationships.csv")

relationships = relationships.drop_duplicates(
    subset=["relationship_id"]
)

relationships["relationship_id"] = (
    relationships["relationship_id"].astype(str).str.strip()
)

relationships["source_type"] = (
    relationships["source_type"].astype(str).str.upper().str.strip()
)

relationships["source_id"] = (
    relationships["source_id"].astype(str).str.strip()
)

relationships["relationship_type"] = (
    relationships["relationship_type"].astype(str).str.upper().str.strip()
)

relationships["target_type"] = (
    relationships["target_type"].astype(str).str.upper().str.strip()
)

relationships["target_id"] = (
    relationships["target_id"].astype(str).str.strip()
)

relationships["timestamp"] = pd.to_datetime(
    relationships["timestamp"], errors="coerce"
)

save_csv(relationships, "relationships_clean.csv")

print(f"Relationships: {len(relationships)} records")


# --------------------------------------------------
# 8. ACTIVITIES
# --------------------------------------------------

activities = load_csv("activities.csv")

activities = activities.drop_duplicates(
    subset=["person_id", "date"]
)

activities["person_id"] = (
    activities["person_id"].astype(str).str.strip()
)

activities["date"] = pd.to_datetime(
    activities["date"], errors="coerce"
)

numeric_columns = [
    "call_count",
    "unique_contacts",
    "locations_visited",
    "transaction_count",
    "synthetic_anomaly"
]

for column in numeric_columns:
    activities[column] = pd.to_numeric(
        activities[column], errors="coerce"
    )

# Total activity score
activities["activity_score"] = (
    activities["call_count"]
    + activities["unique_contacts"]
    + activities["locations_visited"]
    + activities["transaction_count"]
)

save_csv(activities, "activities_clean.csv")

print(f"Activities: {len(activities)} records")


# --------------------------------------------------
# 9. CRIME REPORTS
# --------------------------------------------------

crime_reports = load_csv("crime_reports.csv")

crime_reports = crime_reports.drop_duplicates(
    subset=["report_id"]
)

crime_reports["report_id"] = (
    crime_reports["report_id"].astype(str).str.strip()
)

crime_reports["case_id"] = (
    crime_reports["case_id"].astype(str).str.strip()
)

crime_reports["report_date"] = pd.to_datetime(
    crime_reports["report_date"], errors="coerce"
)

crime_reports["text"] = (
    crime_reports["text"].astype(str).str.strip()
)

save_csv(crime_reports, "crime_reports_clean.csv")

print(f"Crime reports: {len(crime_reports)} records")


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(f"\nCleaned files saved to: {OUTPUT_DIR.resolve()}")

print("\nFiles created:")

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    df = pd.read_csv(file)
    print(f"{file.name:<30} {len(df):>6} rows")

print("\nNo original files were modified.")