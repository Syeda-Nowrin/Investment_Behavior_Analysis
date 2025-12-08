# advanced_clean_finance_trends.py
# ---------------------------------------------------------------
# Purpose:
#   Perform advanced data cleaning and preprocessing on the
#   Finance_Trends dataset. Produces two key outputs:
#
#   1) Finance_Trends_Clean_Exploratory.csv
#        - Human-readable cleaned data for EDA / dashboards.
#
#   2) Finance_Trends_ML_Ready_Advanced.csv
#        - Fully processed, ML-ready dataset:
#            * Encoded categoricals
#            * Scaled numeric features
#            * Discretized key variables
#            * PCA components
#            * Reduced via variance and correlation filters
#
# Main Techniques:
#   - Whitespace and category normalization
#   - Type inference for numeric / percentage columns
#   - Advanced imputation (KNNImputer)
#   - Advanced outlier detection (Local Outlier Factor)
#   - Feature engineering (Age Group, Tenure Band, Risk Band)
#   - One-hot encoding, scaling, PCA, feature selection
# ---------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import re

from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# =======================
# 0. CONFIG / FILE PATHS
# =======================

# Root folder where your dataset lives
DATA_DIR = Path(r"/Users/saritaaaaa/Documents/Investment_Behavior_Analysis")

# Raw input file (UNCLEAN data). Make sure this exists and the extension is .csv
INPUT_FILE = DATA_DIR / "Finance_Trends-RAW.csv"

# Output 1: Clean but human-readable (good for EDA)
OUTPUT_CLEAN_EXPLORATORY = DATA_DIR / "Finance_Trends-Cleaned.csv"

# Output 2: Fully transformed and feature-reduced (ML-ready)
OUTPUT_ML_READY = DATA_DIR / "Finance_Trends_ML.csv"

# Advanced processing parameters
DROP_OUTLIERS = True       # If True, rows flagged as LOF outliers are removed
N_NEIGHBORS_KNN = 5        # K for KNNImputer (numeric missing values)
LOF_CONTAMINATION = 0.05   # Fraction of points considered outliers in LOF
PCA_COMPONENTS = 2         # PCA dimensions to keep in ML file

# ============
# 1. LOAD DATA
# ============

print(f"Loading raw data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# Work on a copy of the original to avoid accidental modification
clean = df.copy()

print(f"Initial shape: {clean.shape[0]} rows x {clean.shape[1]} columns")

# =====================================
# 2. BASIC HYGIENE & TEXT NORMALIZATION
# =====================================

# Trim whitespace from column names: " Age " -> "Age"
clean.columns = clean.columns.str.strip()

# Trim whitespace from all object (text) values
for col in clean.select_dtypes(include="object").columns:
    clean[col] = clean[col].astype(str).str.strip()

# Normalize certain known categorical columns to Title Case
# (Only applied if those columns exist; safe for any dataset)
for col in ["Gender", "Segment", "Region", "Category", "Investment Type"]:
    if col in clean.columns:
        clean[col] = clean[col].str.title()

# =====================================
# 3. TYPE INFERENCE & NUMERIC CONVERSION
# =====================================

def to_numeric_if_possible(series: pd.Series) -> pd.Series:
    """
    Try to convert a Series to numeric.
    If >30% of values are numeric after conversion, keep it numeric,
    otherwise return the original text series.
    """
    converted = pd.to_numeric(series, errors="coerce")
    # "converted.notna().mean()" = proportion of non-NaN after conversion.
    if converted.notna().mean() > 0.3:
        return converted
    return series

# Patterns hinting a column is numeric (by name)
numeric_name_patterns = [
    r"amount", r"value", r"balance", r"score", r"income", r"salary",
    r"age", r"year", r"rate", r"return", r"percentage", r"%"
]

numeric_like_cols = []
for col in clean.columns:
    if any(re.search(pat, col, flags=re.IGNORECASE) for pat in numeric_name_patterns):
        numeric_like_cols.append(col)

# Loop through text columns and decide what to convert
object_cols = clean.select_dtypes(include="object").columns.tolist()

for col in object_cols:
    s = clean[col]

    # Case A: Column name or values suggest percentages (e.g., "Expected Return %")
    if re.search(r"%", col, flags=re.IGNORECASE) or s.astype(str).str.contains("%").any():
        # Remove '%' and commas, then convert to float
        stripped = (
            s.astype(str)
             .str.replace("%", "", regex=False)
             .str.replace(",", "", regex=False)
             .str.strip()
        )
        converted = pd.to_numeric(stripped, errors="coerce")
        if converted.notna().mean() > 0.3:
            clean[col] = converted

    # Case B: Column name looks numeric-like (Amount, Score, Age, etc.)
    elif col in numeric_like_cols:
        clean[col] = to_numeric_if_possible(s)

# Re-identify numeric and categorical columns after conversion
num_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = clean.select_dtypes(include=["object"]).columns.tolist()

# ==========================
# AGE SANITY FIX (ADD THIS)
# ==========================
age_col = None
for candidate in ["age", "Age"]:
    if candidate in clean.columns:
        age_col = candidate
        break

if age_col:
    clean[age_col] = pd.to_numeric(clean[age_col], errors="coerce")

    # Strategy: mark impossible ages as missing
    clean.loc[(clean[age_col] < 18) | (clean[age_col] > 100), age_col] = np.nan

print(f"Numeric columns detected: {len(num_cols)}")
print(f"Categorical columns detected: {len(cat_cols)}")

# ==========================
# 4. ADVANCED MISSING VALUES
# ==========================

# ---- 4.1 Numeric: KNNImputer (uses K nearest neighbors) ----
if num_cols:
    print("Applying KNNImputer for numeric missing values...")
    imputer = KNNImputer(n_neighbors=N_NEIGHBORS_KNN)
    clean[num_cols] = imputer.fit_transform(clean[num_cols])

# ---- 4.2 Categorical: fill with 'Unknown' ----
for col in cat_cols:
    # replace literal string "nan" with None, then fill with "Unknown"
    clean[col] = clean[col].replace({"nan": None}).fillna("Unknown")

# Recompute numeric + categorical after imputation (safety)
num_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = clean.select_dtypes(include=["object"]).columns.tolist()

# ===============================
# 5. ADVANCED OUTLIER DETECTION
# ===============================

if num_cols:
    print("Running Local Outlier Factor (LOF) for multivariate outlier detection...")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=LOF_CONTAMINATION,
        metric="minkowski"
    )

    # LOF returns 1 for inliers and -1 for outliers
    labels = lof.fit_predict(clean[num_cols])
    clean["Is_Outlier_LOF"] = (labels == -1)

    n_outliers = clean["Is_Outlier_LOF"].sum()
    print(f"LOF flagged {n_outliers} rows as potential outliers.")

    if DROP_OUTLIERS:
        clean = clean.loc[clean["Is_Outlier_LOF"] == False].copy()
        print(f"After dropping LOF outliers: {clean.shape[0]} rows.")

# ====================
# 6. FEATURE ENGINEERING
# ====================

def make_age_group(age):
    """
    Bucket age into coarse Age Groups.
    """
    if pd.isna(age):
        return "Unknown"
    a = int(age)
    if a <= 24: return "18-24"
    if a <= 34: return "25-34"
    if a <= 44: return "35-44"
    if a <= 54: return "45-54"
    return "55+"

def make_years_band(years):
    """
    Bucket a numeric "years" value into experience/tenure bands.
    """
    if pd.isna(years):
        return "Unknown"
    y = float(years)
    if y < 2: return "0-1 yrs"
    if y < 5: return "2-4 yrs"
    if y < 10: return "5-9 yrs"
    if y < 15: return "10-14 yrs"
    if y < 20: return "15-19 yrs"
    return "20+ yrs"

# ---- 6.1 Age Group ----
if "Age" in clean.columns:
    clean["Age Group"] = clean["Age"].apply(make_age_group)

# ---- 6.2 Tenure Band from any investment-duration-like column ----
for dur_col in ["Years Invested", "Investment Duration", "Investment Duration (Years)"]:
    if dur_col in clean.columns:
        clean["Tenure Band"] = clean[dur_col].apply(make_years_band)
        break

# ---- 6.3 Risk Band from Expected Return or Risk Score ----
risk_source_col = None
for candidate in ["Expected Return %", "Expected Return", "Risk Score"]:
    if candidate in clean.columns:
        risk_source_col = candidate
        break

if risk_source_col:
    # Use 33% and 66% quantiles to set Low/Medium/High cutpoints
    q1 = clean[risk_source_col].quantile(0.33)
    q2 = clean[risk_source_col].quantile(0.66)

    def risk_band(x):
        if pd.isna(x):
            return "Unknown"
        if x <= q1: return "Low"
        if x <= q2: return "Medium"
        return "High"

    clean["Risk Band"] = clean[risk_source_col].apply(risk_band)

# ===========================
# 7. REMOVE EXACT DUPLICATES
# ===========================

before_dups = clean.shape[0]
clean = clean.drop_duplicates()
after_dups = clean.shape[0]
print(f"Removed {before_dups - after_dups} exact duplicate rows.")

# ===========================================
# 8. SAVE CLEAN, HUMAN-READABLE DATA (EDA)
# ===========================================

OUTPUT_CLEAN_EXPLORATORY.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(OUTPUT_CLEAN_EXPLORATORY, index=False)

print(f"Exploratory clean file saved to: {OUTPUT_CLEAN_EXPLORATORY}")
print(f"Shape (EDA clean): {clean.shape[0]} rows x {clean.shape[1]} columns")

# ========================================================
# 9. ADVANCED ML-ORIENTED PREPROCESSING (ML-READY VERSION)
# ========================================================

# Start from the fully cleaned version
ml = clean.copy()

# Re-identify data types
num_cols = ml.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = ml.select_dtypes(include=["object"]).columns.tolist()

# ---------------------------------------
# 9.1 One-hot encode categorical features
# ---------------------------------------

print("One-hot encoding categorical features for ML...")
ml = pd.get_dummies(ml, columns=cat_cols, drop_first=False)

# After one-hot encoding, re-detect numeric columns
num_cols = ml.select_dtypes(include=[np.number]).columns.tolist()

# ----------------------------------------
# 9.2 Scale numeric features (StandardScaler)
# ----------------------------------------

print("Scaling numeric features (StandardScaler)...")
scaler = StandardScaler()
ml[num_cols] = scaler.fit_transform(ml[num_cols])

# -----------------------------------------
# 9.3 Discretize (bin) some key numeric cols
# -----------------------------------------

# We will pick up to 2 columns to discretize (if they exist)
discretized_candidates = ["Expected Return %", "Expected Return", "Risk Score", "Age"]
discretized_cols = [c for c in discretized_candidates if c in ml.columns][:2]

for col in discretized_cols:
    bin_col = f"{col}_bin"
    # Equal-width binning into 4 bins with labels 0,1,2,3
    ml[bin_col] = pd.cut(ml[col], bins=4, labels=False)

# ----------------------------------
# 9.4 PCA (dimensionality reduction)
# ----------------------------------

num_cols = ml.select_dtypes(include=[np.number]).columns.tolist()

if len(num_cols) >= PCA_COMPONENTS:
    print(f"Applying PCA with {PCA_COMPONENTS} components...")
    pca = PCA(n_components=PCA_COMPONENTS)
    pca_vals = pca.fit_transform(ml[num_cols])
    for i in range(PCA_COMPONENTS):
        ml[f"PCA_{i+1}"] = pca_vals[:, i]

# ----------------------------------------
# 9.5 Remove low-variance features
# ----------------------------------------

print("Applying VarianceThreshold (drop low-variance features)...")
selector = VarianceThreshold(threshold=0.01)
reduced_array = selector.fit_transform(ml)
reduced_cols = ml.columns[selector.get_support()]
ml = pd.DataFrame(reduced_array, columns=reduced_cols)

# ---------------------------------------------------
# 9.6 Remove highly correlated features (collinearity)
# ---------------------------------------------------

print("Dropping highly correlated features (corr > 0.95)...")
corr = ml.corr(numeric_only=True).abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
ml = ml.drop(columns=to_drop_corr, errors="ignore")

print(f"Dropped {len(to_drop_corr)} highly correlated features.")

# ------------------------
# 9.7 SAVE ML-READY OUTPUT
# ------------------------

ml.to_csv(OUTPUT_ML_READY, index=False)

print(f"ML-ready file saved to: {OUTPUT_ML_READY}")
print(f"Shape (ML-ready): {ml.shape[0]} rows x {ml.shape[1]} columns")
print("Advanced cleaning & preprocessing complete ✅")
