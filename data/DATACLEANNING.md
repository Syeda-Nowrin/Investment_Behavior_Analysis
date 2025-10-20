# Investment & Salary EDA Dashboard (Streamlit + Plotly)

A clean, interactive dashboard to explore investment behavior alongside salary and education insights.

---

## What’s inside

- **Seven pages** (left navigation): Overview, Investment Behavior, Salary & Education Insights, Income & Investment Relationship, Demographics & Behavioral Patterns, Interactive Data Explorer, Insights & Storytelling.
- Robust to messy data: smart column detection, median imputation for numerics, “Unknown” for text, Expected Return parsing, Rank‑1 logic.
- Visuals: Plotly pie/bar/violin/box/stacked/grouped/scatter+trendline/treemap/heatmaps.

---

## Project Structure (recommended)

```
project-root/
├─ app.py                      # Streamlit app (rename if you prefer)
├─ data/
│  ├─ finance.csv             # was: Finance_Dataset_Cleaned.csv
│  └─ salary.csv              # was: Salary_Dataset_Cleaned.csv
├─ scripts/
│  ├─ clean_salary.py
│  └─ clean_and_rename_finance.py
└─ README.md
```

> If you keep a different layout, just update the `FIN_PATH` and `SAL_PATH` constants in `app.py`.

---

## Requirements

```bash
pip install streamlit pandas numpy plotly
```

(Optional for Excel):
```bash
pip install openpyxl
```

---

## Run

```bash
streamlit run app.py
```

---

## Data files (rename & paths)

Expected files:

- `data/finance.csv`  (previously `Finance_Dataset_Cleaned.csv`)
- `data/salary.csv`   (previously `Salary_Dataset_Cleaned.csv`)

Update `app.py`:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FIN_PATH = DATA_DIR / "finance.csv"
SAL_PATH = DATA_DIR / "salary.csv"
```

Move/rename your files accordingly.

---

## Remove UI Icons

- Remove `page_icon` from `st.set_page_config(...)`.
- Delete the emoji `<span>...</span>` in the sidebar header block.
- Optionally remove decorative emojis from markdown text.

**Before:**
```python
st.set_page_config(page_title="Investment Behavior Analysis", page_icon="🪙", layout="wide")
FIN_PATH = BASE_DIR / "Finance_Dataset_Cleaned.csv"
SAL_PATH = BASE_DIR / "Salary_Dataset_Cleaned.csv"
```

**After:**
```python
st.set_page_config(page_title="Investment Behavior Analysis", layout="wide")
DATA_DIR = BASE_DIR / "data"
FIN_PATH = DATA_DIR / "finance.csv"
SAL_PATH = DATA_DIR / "salary.csv"
```

---

## Cleaning Scripts

Run:
```bash
python scripts/clean_salary.py
python scripts/clean_and_rename_finance.py
```

Copy/rename outputs to `data/salary.csv` and `data/finance.csv`.

---

## Troubleshooting

- File not found → check `data/` folder & filenames.
- Plots missing → confirm `plotly` installed.
- Permissions → ensure terminal has write access.

---

## License

MIT (or your choice).
