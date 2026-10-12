# inspect

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("CRIMINAL NETWORK AI - DATASET INSPECTION")
print("=" * 70)

print(f"\nTotal CSV files found: {len(csv_files)}\n")

for file in csv_files:
    print("\n" + "=" * 70)
    print(f"FILE: {file.name}")
    print("=" * 70)

    df = pd.read_csv(file)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names:")
    print(list(df.columns))

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 3 Records:")
    print(df.head(3).to_string(index=False))