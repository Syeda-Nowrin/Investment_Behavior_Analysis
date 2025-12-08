# clean_salary.py
# ---------------------------------------------------------------
# Purpose:
#   Clean the Salary dataset and add helpful derived columns.
#
# Operations:
#   1) Trim spaces in column names and text cells
#   2) Normalize categories (Title Case)
#   3) Convert numeric-looking columns to numeric
#   4) Handle missing values (median for numeric, "Unknown" for text)
#   5) Create Age Group and Experience Band
#   6) Remove exact duplicates
#   7) Save cleaned CSV
#   8) Standardize "Education Level" bucket:
#        - "Master'S" or "Master'S Degree" -> "Master's"
#        - "Bachelor'S" or "Bachelor'S Degree" -> "Bachelor's"
# ---------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------- file paths ----------
DATA_DIR = Path("/Users/saritaaaaa/Documents/Investment_Behavior_Analysis")
INPUT_FILE = DATA_DIR / "Salary_Dataset.csv"
OUTPUT_FILE = DATA_DIR / "Salary_Dataset_Cleaned.csv"

# ---------- load ----------
df = pd.read_csv(INPUT_FILE)

# 1) Trim spaces in column names
df.columns = df.columns.str.strip()

# 2) Trim whitespace inside all text cells & Title Case for key categoricals
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

for c in ["Gender", "Education Level", "Job Title"]:
    if c in df.columns:
        df[c] = df[c].str.title()

# 3) Convert numeric-looking columns to numeric (support minor header variants)
numeric_candidates = ["Age", "Years of Experience", "Years Of Experience", "Salary"]
for c in numeric_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
# unify possible variant
if "Years Of Experience" in df.columns and "Years of Experience" not in df.columns:
    df = df.rename(columns={"Years Of Experience": "Years of Experience"})

# 4) Handle missing values: median for numeric, "Unknown" for text
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].replace({"nan": None}).fillna("Unknown")

# 5) Create Age Group and Experience Band
def make_age_group(age):
    if pd.isna(age): return "Unknown"
    a = int(age)
    if a <= 24: return "18-24"
    if a <= 34: return "25-34"
    if a <= 44: return "35-44"
    if a <= 54: return "45-54"
    return "55+"

def make_experience_band(years):
    if pd.isna(years): return "Unknown"
    y = float(years)
    if y < 2: return "0-1 yrs"
    if y < 5: return "2-4 yrs"
    if y < 10: return "5-9 yrs"
    if y < 15: return "10-14 yrs"
    if y < 20: return "15-19 yrs"
    return "20+ yrs"

if "Age" in df.columns:
    df["Age Group"] = df["Age"].apply(make_age_group)
if "Years of Experience" in df.columns:
    df["Experience Band"] = df["Years of Experience"].apply(make_experience_band)

# 8) Standardize "Education Level" into buckets
#    EXACTLY as requested:
#      - "Master'S" or "Master'S Degree" -> "Master's"
#      - "Bachelor'S" or "Bachelor'S Degree" -> "Bachelor's"
col = "Education Level"
if col in df.columns:
    # normalize punctuation for comparison
    edu_raw = df[col].astype(str)
    edu_norm = (
        edu_raw.str.replace("’", "'", regex=False)       # curly -> straight apostrophe
               .str.replace("`", "'", regex=False)       # backtick -> apostrophe
               .str.replace(r"\s+", " ", regex=True)
               .str.strip()
    )

    # Build boolean masks (case-insensitive, allow optional " Degree")
    master_mask = edu_norm.str.fullmatch(r"(?i)master's(?:\s+degree)?")
    bachelor_mask = edu_norm.str.fullmatch(r"(?i)bachelor's(?:\s+degree)?")

    # Apply buckets
    df.loc[master_mask, col] = "Master's"
    df.loc[bachelor_mask, col] = "Bachelor's"

    # Final polish: if any values still have "'S" uppercase, fix to "'s"
    df[col] = df[col].str.replace("'S", "'s", regex=False)

# 6) Remove exact duplicates
df = df.drop_duplicates()

# 7) Save cleaned CSV
df.to_csv(OUTPUT_FILE, index=False)

# ---------- console feedback ----------
print(f"Cleaned file saved to: {OUTPUT_FILE}")
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("New columns present:",
      [c for c in ["Age Group", "Experience Band"] if c in df.columns])
