"""
Data Preparation step of the Tourism Package Prediction pipeline.

1. Loads the registered dataset directly from the repository data folder.
2. Cleans it: drops identifier / index columns, fixes known categorical typos,
   removes duplicate rows and imputes any missing values.
3. Splits the cleaned data into stratified train and test sets.
4. Saves Xtrain / Xtest / ytrain / ytest as CSV files so the GitHub Actions
   workflow can hand them to the training job as a workflow artifact.

Categorical features are deliberately kept as raw strings here. They are
one-hot encoded inside the model pipeline in train.py, which guarantees the
Streamlit app (which also sends raw strings) uses exactly the same encoding.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"
TARGET_COL = "ProdTaken"

# ---- 1. Load -----------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print(f"Loaded raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# ---- 2. Clean ----------------------------------------------------------------
# Drop columns that carry no predictive signal:
#   - 'Unnamed: 0' is a leftover pandas index written into the CSV
#   - 'CustomerID' is a unique identifier
cols_to_drop = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"Dropped unnecessary columns: {cols_to_drop}")

# Fix the known typo in Gender ('Fe Male' -> 'Female')
if "Gender" in df.columns:
    before = (df["Gender"] == "Fe Male").sum()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    print(f"Fixed {before} 'Fe Male' entries in Gender -> 'Female'")

# Remove exact duplicate records
n_dup = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Removed {n_dup} duplicate rows")

# Impute missing values (median for numeric, mode for categorical).
# The current file has none, but this keeps the pipeline robust to new data.
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]
n_missing = int(df.isnull().sum().sum())
for c in numeric_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())
for c in categorical_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode()[0])
print(f"Imputed {n_missing} missing values")

print(f"Cleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Numeric features     : {[c for c in numeric_cols if c != TARGET_COL]}")
print(f"Categorical features : {categorical_cols}")

# ---- 3. Split ----------------------------------------------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# stratify=y keeps the ~19% positive-class ratio identical across both splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 4. Save -----------------------------------------------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("\nData prepared: train/test splits written to Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
print(f"Train shape: {Xtrain.shape}, positive rate: {ytrain.mean():.2%}")
print(f"Test  shape: {Xtest.shape}, positive rate: {ytest.mean():.2%}")
