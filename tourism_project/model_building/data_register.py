"""
Data Registration step of the Tourism Package Prediction pipeline.

The dataset lives inside the GitHub repository itself (tourism_project/data/),
so "registration" is a lightweight, file-based validation: the script loads the
CSV, verifies that every expected column is present, and prints a summary that
is visible in the GitHub Actions log. If the schema has drifted, the script
fails loudly and the pipeline stops before any downstream job runs.
"""
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Columns the downstream pipeline (prep.py / train.py / app.py) relies on.
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
TARGET_COL = "ProdTaken"

# ---- Load the raw dataset -------------------------------------------------
df = pd.read_csv(RAW_PATH)

# ---- Validate the schema --------------------------------------------------
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

# ---- Print a registration summary -----------------------------------------
print("Dataset registered successfully.")
print(f"Source file : {RAW_PATH}")
print(f"Rows        : {df.shape[0]}")
print(f"Columns     : {df.shape[1]}")
print(f"Expected columns present: {len(EXPECTED_COLUMNS)}/{len(EXPECTED_COLUMNS)}")
if extra:
    print(f"Extra columns (will be dropped in data prep): {extra}")

print("\nColumn data types:")
print(df.dtypes.to_string())

print("\nMissing values per column:")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0].to_string() if missing_counts.sum() > 0
      else "  No missing values found.")

print(f"\nTarget distribution ({TARGET_COL}):")
print(df[TARGET_COL].value_counts().to_string())
print(f"Positive class rate: {df[TARGET_COL].mean():.2%}")
