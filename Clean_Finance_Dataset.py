# clean_and_rename_finance.py
# ---------------------------------------------------------------
# Purpose:
#   1) Load the survey CSV exported from Excel
#   2) Light cleaning (trim spaces, coerce numeric types, fill nulls)
#   3) Rename columns that start with:
#        "What do you think are the best options for investing your money?"
#      to human-friendly names, for example:
#        "What do you think are the best options for investing your money? (Rank in order of preference) [Gold]"
#        --> "Preference rank for gold investment"
#        (if "Rank in order of preference" phrase is missing, then:)
#        --> "Preference for gold investment"
#
# How to run:
#   python clean_and_rename_finance.py
# ---------------------------------------------------------------

from pathlib import Path
import re
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
# Update this path if your files live elsewhere
DATA_DIR = Path("/Users/saritaaaaa/Documents/Project")

INPUT_FILE = DATA_DIR / "Finance_Dataset.csv"          # your raw CSV
OUTPUT_FILE = DATA_DIR / "Finance_Dataset_Cleaned.csv" # cleaned + renamed

# If your source is actually an Excel workbook, switch to:
#   pip install openpyxl
#   df = pd.read_excel(DATA_DIR / "Finance_Dataset.xlsx")

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(INPUT_FILE)

# -------------------------------
# Light cleaning (safe defaults)
# -------------------------------

# 1) Normalize column names by trimming whitespace
df.columns = df.columns.str.strip()

# 2) Trim whitespace in string columns (avoids " gold " vs "gold")
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

# 3) Coerce common numeric-like fields if they exist
#    Add or remove names as needed for your file
numeric_candidates = [
    "AGE",
    "EXPECTED_RETURN_PCT",
    "AVG_SALARY",
    "AVG_EXPERIENCE",
    "MONITOR_FREQ_PER_MONTH",
]
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4) Simple missing value handling
#    - Numbers: fill with median
#    - Text: fill with "Unknown"
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].replace({"nan": None}).fillna("Unknown")

# -------------------------------
# Column renaming logic
# -------------------------------
# We will look for columns that START with the target prefix,
# optionally followed by "(Rank in order of preference)",
# and then a bracketed instrument like "[Gold]".
#
# Examples that this regex will match:
#  - What do you think are the best options for investing your money? (Rank in order of preference) [Gold]
#  - What do you think are the best options for investing your money? [Mutual Funds]
#
# Groups captured:
#  - rank_phrase: the literal rank phrase if present
#  - option:      the bracketed option text (e.g., Gold, Mutual Funds, etc.)
#
PREFIX = r"^What do you think are the best options for investing your money\?"
RANK_PHRASE = r"\s*\(Rank in order of preference\)"
OPTION_BRACKET = r"\s*\[(?P<option>.+?)\]\s*$"

pattern_with_rank = re.compile(PREFIX + RANK_PHRASE + OPTION_BRACKET, flags=re.IGNORECASE)
pattern_no_rank   = re.compile(PREFIX + OPTION_BRACKET,                   flags=re.IGNORECASE)

rename_map = {}   # old_name -> new_name
renamed_list = [] # for printing a summary

for col in df.columns:
    col_stripped = col.strip()

    # First try matching the "with rank" version
    m_rank = pattern_with_rank.match(col_stripped)
    if m_rank:
        option_raw = m_rank.group("option").strip()
        # We want "Preference rank for {option upper} investment"
        new_name = f"Preference rank for {option_raw.upper()} investment"
        rename_map[col] = new_name
        renamed_list.append((col, new_name))
        continue

    # Then try matching the "no rank" version
    m_no_rank = pattern_no_rank.match(col_stripped)
    if m_no_rank:
        option_raw = m_no_rank.group("option").strip()
        # We want "Preference for {option upper} investment"
        new_name = f"Preference for {option_raw.upper()} investment"
        rename_map[col] = new_name
        renamed_list.append((col, new_name))
        continue

# Apply the renames in one pass
df = df.rename(columns=rename_map)

# -------------------------------
# Optional: derive Age_Group if useful and missing
# -------------------------------
def to_age_group(x):
    try:
        a = float(x)
    except Exception:
        return "Unknown"
    if pd.isna(a): 
        return "Unknown"
    a = int(a)
    if a <= 24: return "18-24"
    if a <= 34: return "25-34"
    if a <= 44: return "35-44"
    if a <= 54: return "45-54"
    return "55+"

if "Age_Group" not in df.columns and "AGE" in df.columns:
    df["Age_Group"] = df["AGE"].apply(to_age_group)

# -------------------------------
# Drop exact duplicate rows (safe)
# -------------------------------
df = df.drop_duplicates()

# -------------------------------
# Save and report
# -------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved cleaned file to:\n  {OUTPUT_FILE}\n")
if renamed_list:
    print("Renamed columns:")
    for old, new in renamed_list:
        print(f"  {old}  -->  {new}")
else:
    print("No columns matched the renaming pattern. Check your exact headers.")
