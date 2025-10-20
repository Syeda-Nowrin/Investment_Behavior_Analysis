# Investment & Salary EDA Dashboard (Streamlit + Plotly)

An interactive Streamlit app for exploring **investment behavior** and **salary/education** patterns side-by-side.  
It reads two cleaned CSVs — `Finance_Dataset_Cleaned.csv` and `Salary_Dataset_Cleaned.csv` — and provides guided tabs, filters, and beautiful visuals (Plotly) to help you discover insights quickly.

---

## ✨ Features

- **Left-hand navigation** with active highlight (radio-based)  
- **Global filters** (Gender, Age Group, Education, Experience Band) that update all pages  
- **Seven pages**:
  1. **Overview Dashboard** — KPIs, investment pie, salary-by-education bar, combined correlation heatmap, summary stats for both datasets  
  2. **Investment Behavior** — risk appetite bands, tenure preferences, monitoring frequency, return-by-education, insights  
  3. **Salary & Education Insights** — distributions, box/violin plots, correlation, OLS trendlines, insights  
  4. **Income & Investment Relationship** — salary ↔ expected return ↔ experience bubble, preferences by education, combined heatmap, insights  
  5. **Demographics & Behavioral Patterns** — gender/age/education stacks, treemap of age→education→goals, insights  
  6. **Interactive Data Explorer** — quick charts + missingness summary & heatmaps, raw-data viewing & CSV download  
  7. **Insights & Storytelling** — concise takeaways ready for slides/reports  

- **Robust parsing** of messy fields (e.g., “Expected Return %” like `8–12%`, `~10`).  
- **Auto-detection** of “Preference rank for … investment” columns (1 = most preferred).  
- **Clean theming & cards** with CSS for Plotly charts, metrics, and dataframes.  

---

## 📁 Project Structure

```
.
├── app.py
├── Finance_Dataset_Cleaned.csv
├── Salary_Dataset_Cleaned.csv
├── README.md
└── requirements.txt
```

> If your CSVs live elsewhere, update the paths at the top of `app.py`:
>
> ```python
> BASE_DIR = Path(__file__).resolve().parent
> FIN_PATH = BASE_DIR / "Finance_Dataset_Cleaned.csv"
> SAL_PATH = BASE_DIR / "Salary_Dataset_Cleaned.csv"
> ```

---

## 🔧 Requirements

- Python **3.9+** (3.10/3.11 OK)
- See `requirements.txt` for exact versions:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `plotly`

Install all dependencies:
```bash
pip install -r requirements.txt
```

Or minimal:
```bash
pip install streamlit pandas numpy plotly
```

---

## 🚀 Run Locally

From the project folder (where `app.py` lives):

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

**Tip:** If you change CSV locations or names, update `FIN_PATH` and `SAL_PATH` in `app.py`.

---

## 📊 Data Expectations

- **Finance_Dataset_Cleaned.csv**  
  Typical columns (names are auto-detected and case-insensitive):
  - Gender (e.g., `Gender`/`Sex`)
  - Age / Age_Group
  - *Expected Return* (string or numeric, e.g., `10%`, `8–12%`, `~12`)
  - Monitoring frequency (e.g., Weekly/Monthly)
  - Tenure preference
  - One or more **“Preference rank for … investment”** columns (1 = most preferred)
  - Optional goal/objective fields (e.g., “Savings Goal”, “Investment Objectives”)

- **Salary_Dataset_Cleaned.csv**  
  Typical columns:
  - Salary / Income (numeric or string convertible)
  - Education Level
  - Years of Experience (numeric) **or** Experience Band (e.g., `3–5 years`, `20+ yrs`)

> The app is resilient to whitespace and small naming variations. If a field is missing, the related chart will show a helpful message rather than breaking.

---

## 🧭 Navigation & Filters

The **left sidebar** contains:
- **Navigation** (radio) — the selected page is **highlighted**.
- **Global filters** — Gender, Age Group, Education, Experience Band.  
  These filters are applied across pages. Some pages also have extra **page-level filters** inside expanders.

---

## 🧠 Notes & Troubleshooting

- If a chart says “not enough data,” broaden filters or check that your CSV column names contain the expected keywords (e.g., “expected return”, “monitor”, “education”).  
- For the “rank” logic, ensure your columns literally match the pattern:  
  `Preference rank for <something> investment` (any case).  
- If you see duplicate labels in heatmaps, the app automatically cleans/uniquifies them.  
- For performance, `@st.cache_data` is used on CSV loading — clear cache via **Rerun** or **Always rerun** if needed.

---

## 📦 Deploying

- **Streamlit Community Cloud**: Push to GitHub and deploy; make sure both CSVs are in the repo or fetched at runtime.  
- **Docker** (optional quick example):
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY . .
  RUN pip install -r requirements.txt
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
  ```

---

## 📝 License

MIT (or your preferred license).

---

## 🙌 Acknowledgements

Built with ♥ using **Streamlit** and **Plotly**.  
Data cleaning assumptions and label parsing are designed to be robust to “real-world” messy survey data.
