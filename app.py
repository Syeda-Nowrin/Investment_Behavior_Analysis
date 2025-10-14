# app.py
# -------------------------------------------------------------------
# Investment & Salary EDA Dashboard (Plotly-only, robust to messy data)
# -------------------------------------------------------------------
# Fixes included:
#   • Safe slider for Expected Return (skips if non-numeric -> no empty table)
#   • Popular Investment Options: supports Rank=1 OR binary "preference" columns
#   • Goals: robust detection + split on comma/semicolon/pipe
#   • Heatmap: selects meaningful numeric cols, drops zero-variance, cleans labels
#   • Extra filters added: Experience Band, Education Level, Monitor Frequency,
#     Expected Return %, Tenure Preference, Savings Objectives
#
# Requires:
#   pip install streamlit pandas numpy plotly
# -------------------------------------------------------------------

from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page config
# -----------------------------

st.set_page_config(
    page_title="Investment Behavior Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Config: file paths
# -----------------------------
#DATA_DIR = Path("/Users/saritaaaaa/Documents/Investment_Behavior_Analysis")
#FINANCE_FILE = DATA_DIR / "Finance_Dataset_Cleaned.csv"
#SALARY_FILE  = DATA_DIR / "Salary_Dataset_Cleaned.csv"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FINANCE_FILE = DATA_DIR / "Finance_Dataset_Cleaned.csv"
SALARY_FILE  = DATA_DIR / "Salary_Dataset_Cleaned.csv"



# 3) DEBUG (optional; you can delete later)
st.caption(f"CWD: {Path.cwd()}")
st.caption(f"Looking for: {FINANCE_FILE}  ->  exists? {FINANCE_FILE.exists()}")
st.caption(f"Looking for: {SALARY_FILE}   ->  exists? {SALARY_FILE.exists()}")

# 4) Friendly guard (prevents crash if files missing)
if not FINANCE_FILE.exists() or not SALARY_FILE.exists():
    st.error("CSV files not found in `data/`. Check exact names/case and push them to GitHub.")
    st.stop()


# -----------------------------
# Helpers
# -----------------------------
def fcol(cols, key: str):
    """First column whose lowercase name contains `key`."""
    key = key.lower()
    for c in cols:
        if key in c.lower():
            return c
    return None

def multi_match(cols, keys):
    """First column containing any key (case-insensitive)."""
    for k in keys:
        c = fcol(cols, k)
        if c:
            return c
    return None

def to_age_group(x):
    """Map an age number to a standard age bucket."""
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

def split_multi(s: pd.Series) -> pd.Series:
    """
    Split multi-select strings on comma/semicolon/pipe and trim.
    Handles NaNs safely.
    """
    if s.isna().all():
        return s
    return (
        s.astype(str)
         .str.replace(r"\s*[,;|]\s*", "|", regex=True)  # unify separators to |
         .str.split("|")
    )

def tidy_yesno_to_int(series: pd.Series) -> pd.Series:
    """
    Convert yes/no-ish strings to {0,1}. If numeric already, keep numeric.
    """
    out = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(out):
        return out.fillna(0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "yes", "true", "y"]).astype(int)

def pretty_heat_label(name: str) -> str:
    """
    Clean noisy column labels for heatmap axes.
    Examples:
      'fin_Preference rank for MUTUAL FUNDS investment' -> 'Rank: Mutual Funds'
      'sal_Years of Experience' -> 'Years of Experience'
    """
    n = name
    n = n.replace("fin_", "").replace("sal_", "")
    n = re.sub(r"Preference\s+rank\s+for\s+(.*?)\s+investment", r"Rank: \1", n, flags=re.I)
    n = n.replace("_", " ")
    n = n.title()
    return n

def find_return_col(cols):
    """Find the expected-return column robustly."""
    key_parts = ["how much return do you expect", "expected return", "return"]
    for c in cols:
        lc = c.lower().strip()
        if any(k in lc for k in key_parts):
            return c
    return None


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(fin_path: Path, sal_path: Path):
    finance = pd.read_csv(fin_path)
    salary  = pd.read_csv(sal_path)
    finance.columns = finance.columns.str.strip()
    salary.columns  = salary.columns.str.strip()
    return finance, salary

finance, salary = load_data(FINANCE_FILE, SALARY_FILE)

# -----------------------------
# Robust column detection
# -----------------------------
# Finance
col_gender  = multi_match(finance.columns, ["gender", "sex"]) or "Gender"
col_age     = multi_match(finance.columns, ["age"]) or "Age"
col_agegrp  = fcol(finance.columns, "age_group") or "Age_Group"
col_return  = multi_match(finance.columns, ["expected_return_pct", "expected return", "return"])
col_monitor = multi_match(finance.columns, ["how often do you monitor", "monitor"])
col_tenure  = multi_match(finance.columns, ["how long do you prefer to keep your money", "tenure", "keep your money"])
col_goals   = multi_match(finance.columns, ["savings objectives", "purpose behind", "investment objective", "goals"])

# Derive Age_Group if missing
if col_agegrp not in finance.columns and col_age in finance.columns:
    finance[col_agegrp] = finance[col_age].apply(to_age_group)

# Option columns:
rank_cols = [c for c in finance.columns if c.lower().startswith("preference rank for ")]
binary_pref_cols = [c for c in finance.columns if c.lower().startswith("preference for ")]
legacy_like = [c for c in finance.columns if c.upper().startswith("INVEST_OPTION_")]
if not binary_pref_cols and legacy_like:
    binary_pref_cols = legacy_like

# Salary
col_salary   = multi_match(salary.columns, ["salary", "comp"])
col_edu      = multi_match(salary.columns, ["education level", "education"])
col_expyrs   = multi_match(salary.columns, ["years of experience", "experience"])
col_expband  = fcol(salary.columns, "experience band")

# Type coercion (numeric expectations)
for c in [col_age, col_return]:
    if c in finance.columns:
        finance[c] = pd.to_numeric(finance[c], errors="coerce")
for c in [col_salary, col_expyrs]:
    if c in salary.columns:
        salary[c] = pd.to_numeric(salary[c], errors="coerce")

# -----------------------------
# Page config
# -----------------------------

#st.set_page_config(page_title="Overview", layout="wide")
#st.title("💹 Investment Behavior Analysis")
#st.caption("Use the sidebar to filter. Charts update automatically.")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔍 Filters")

# Gender
gender_vals = sorted(finance[col_gender].dropna().unique()) if col_gender in finance.columns else []
sel_gender  = st.sidebar.multiselect("Gender", gender_vals, default=gender_vals) if gender_vals else []

# Age Group
age_vals    = sorted(finance[col_agegrp].dropna().unique()) if col_agegrp in finance.columns else []
sel_agegrp  = st.sidebar.multiselect("Age Group", age_vals, default=age_vals) if age_vals else []

# Education (Salary)
edu_vals    = sorted(salary[col_edu].dropna().unique()) if col_edu in salary.columns else []
sel_edu     = st.sidebar.multiselect("Education Level", edu_vals, default=edu_vals) if edu_vals else []

# Experience Band (Salary)
expband_vals = sorted(salary[col_expband].dropna().unique()) if col_expband in salary.columns else []
sel_expband  = st.sidebar.multiselect("Experience Band", expband_vals, default=expband_vals) if expband_vals else []

# Monitor Frequency (Finance)
mon_vals    = sorted(finance[col_monitor].dropna().unique()) if col_monitor in finance.columns else []
sel_monitor = st.sidebar.multiselect("How often do you monitor your investment?", mon_vals, default=mon_vals) if mon_vals else []

# Expected Return slider (safe)
sel_ret = None
if col_return in finance.columns:
    s_all = pd.to_numeric(finance[col_return], errors="coerce")
    valid = s_all[np.isfinite(s_all)]
    if valid.size >= 2:
        rmin, rmax = float(valid.min()), float(valid.max())
        sel_ret = st.sidebar.slider("Expected Return (%)", rmin, rmax, (rmin, rmax))
    else:
        st.sidebar.info("Expected Return (%) not numeric or insufficient data — filter disabled.")

# Tenure (Finance)
tenure_vals = sorted(finance[col_tenure].dropna().unique()) if col_tenure in finance.columns else []
sel_tenure  = st.sidebar.multiselect("How long do you prefer to keep your money?", tenure_vals, default=tenure_vals) if tenure_vals else []

# Savings Objectives (Finance) — discover from data
sel_goals = None
if col_goals in finance.columns:
    g = split_multi(finance[col_goals])
    tmp = finance.copy()
    tmp[col_goals] = g
    tmp = tmp.explode(col_goals)
    tmp[col_goals] = tmp[col_goals].astype(str).str.strip()
    goal_vals = sorted([x for x in tmp[col_goals].dropna().unique() if x])
    sel_goals = st.sidebar.multiselect("Savings objectives", goal_vals, default=goal_vals) if goal_vals else None

# -----------------------------
# Apply filters (Finance & Salary)
# -----------------------------
f_fin = finance.copy()
if col_gender in f_fin.columns and sel_gender:
    f_fin = f_fin[f_fin[col_gender].isin(sel_gender)]
if col_agegrp in f_fin.columns and sel_agegrp:
    f_fin = f_fin[f_fin[col_agegrp].isin(sel_agegrp)]
if col_monitor in f_fin.columns and sel_monitor:
    f_fin = f_fin[f_fin[col_monitor].isin(sel_monitor)]
if col_tenure in f_fin.columns and sel_tenure:
    f_fin = f_fin[f_fin[col_tenure].isin(sel_tenure)]
if col_return in f_fin.columns and sel_ret is not None:
    s = pd.to_numeric(f_fin[col_return], errors="coerce")
    f_fin = f_fin[(s >= sel_ret[0]) & (s <= sel_ret[1])]
if col_goals in f_fin.columns and sel_goals:
    g = split_multi(f_fin[col_goals])
    mask = np.zeros(len(f_fin), dtype=bool)
    # Row matches if it contains ANY selected goal
    for i, lst in enumerate(g):
        if isinstance(lst, list) and any(goal in [x.strip() for x in lst] for goal in sel_goals):
            mask[i] = True
    f_fin = f_fin[mask]

f_sal = salary.copy()
if col_edu in f_sal.columns and sel_edu:
    f_sal = f_sal[f_sal[col_edu].isin(sel_edu)]
if col_expband in f_sal.columns and sel_expband:
    f_sal = f_sal[f_sal[col_expband].isin(sel_expband)]

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total number of financial records", len(f_fin))
with k2:
    if col_salary in f_sal.columns:
        st.metric("Median salary", f"{pd.to_numeric(f_sal[col_salary], errors='coerce').median():,.0f}")
with k3:
    label = "Most popular investing age group"

    # --- robust finder for an "invests?" column ---
    def find_invest_col(cols):
        keys = [
            "do you invest in investment avenues",
            "invest in investment avenues",
            "do you invest",
            "invest in stock",
            "invest in stock market",
        ]
        for c in cols:
            lc = c.lower().strip()
            if any(k in lc for k in keys):
                return c
        return None

    # --- robust yes/no normalizer (does NOT coerce to numeric first) ---
    def to_yes_no_flag(series):
        s = series.astype(str).str.strip().str.lower()
        yes_vals = {"yes","y","true","t","1","agree","i do","invest","investing"}
        no_vals  = {"no","n","false","f","0","disagree","i do not","dont","don't"}
        out = pd.Series(np.nan, index=series.index, dtype="float")
        out[s.isin(yes_vals)] = 1
        out[s.isin(no_vals)]  = 0
        # Also handle explicit 1/0 strings mixed with text
        out[s.str.fullmatch(r"\s*1\s*")] = 1
        out[s.str.fullmatch(r"\s*0\s*")] = 0
        return out

    if col_agegrp in f_fin.columns:
        df_age = f_fin.copy()

        invest_col = find_invest_col(df_age.columns)
        used_filter = False
        if invest_col:
            flag = to_yes_no_flag(df_age[invest_col])
            yes_mask = flag == 1
            if yes_mask.any():
                df_age = df_age[yes_mask]
                used_filter = True  # we actually filtered to investors

        # If filtering wiped everything, fall back to all respondents
        if df_age.empty or df_age[col_agegrp].dropna().empty:
            st.metric(label, "—", help="No eligible rows under current filters")
        else:
            counts = df_age[col_agegrp].dropna().astype(str).value_counts()
            top_age = counts.idxmax()
            top_cnt = int(counts.iloc[0])
            total   = int(counts.sum())
            pct     = 100.0 * top_cnt / total if total else 0.0
            extra   = " (investors only)" if used_filter else " (all respondents)"
            st.metric(label, top_age) #, help=f"{top_cnt}/{total} • {pct:.1f}%{extra}")
    else:
        st.metric(label, "—", help="Age group column not found")


with k4:
    label = "Most active investing gender"
    invest_col_exact = "Do you invest in Investment Avenues?"

    def to_yes_no_flag(series):
        s = series.astype(str).str.strip().str.lower()
        yes_vals = {"yes","y","true","t","1","agree","i do","invest","investing"}
        no_vals  = {"no","n","false","f","0","disagree","i do not","dont","don't"}
        out = pd.Series(np.nan, index=series.index, dtype="float")
        out[s.isin(yes_vals)] = 1
        out[s.isin(no_vals)]  = 0
        # explicit 1/0 strings
        out[s.str.fullmatch(r"\s*1\s*")] = 1
        out[s.str.fullmatch(r"\s*0\s*")] = 0
        return out

    if (col_gender in f_fin.columns) and (invest_col_exact in f_fin.columns):
        flags = to_yes_no_flag(f_fin[invest_col_exact])
        investors = f_fin[flags == 1]

        if investors.empty or investors[col_gender].dropna().empty:
            st.metric(label, "—")  # no tooltip => no "?"
        else:
            vc = investors[col_gender].astype(str).value_counts()
            top_gender = vc.idxmax()
            top_cnt = int(vc.iloc[0])
            total_cnt = int(vc.sum())
            pct = 100.0 * top_cnt / total_cnt if total_cnt else 0.0
            st.metric(label, top_gender) #, delta=f"{top_cnt}/{total_cnt} • {pct:.1f}%")
    else:
        st.metric(label, "—")

st.divider()

# -----------------------------
# Chart 1: Most Preferred Investment Option (by average interest rank)
# -----------------------------
st.subheader("📊 Most Preferred Investment Option")

rank_cols_map = {
    "Mutual Funds": "Preference rank for MUTUAL FUNDS investment",
    "Equity Market": "Preference rank for EQUITY MARKET investment",
    "Debentures": "Preference rank for DEBENTURES investment",
    "Government Bonds": "Preference rank for GOVERNMENT BONDS investment",
    "Fixed Deposits": "Preference rank for FIXED DEPOSITS investment",
    "Public Provident Fund": "Preference rank for PUBLIC PROVIDENT FUND investment",
    "Gold": "Preference rank for GOLD investment",
}

available = {label: col for label, col in rank_cols_map.items() if col in f_fin.columns}

if len(f_fin) == 0:
    st.info("No finance rows under current filters.")
elif not available:
    st.info("No rank columns found in the current dataset.")
else:
    avg_ranks = {}
    counts = {}
    for label, col in available.items():
        s = pd.to_numeric(f_fin[col], errors="coerce")
        s = s[(s >= 1) & (s <= 10)]  # keep valid 1–10 responses
        if s.notna().any():
            avg_ranks[label] = s.mean()
            counts[label] = int(s.count())

    if not avg_ranks:
        st.info("No numeric rank values available for the current filters.")
    else:
        avg_df = (
            pd.DataFrame({
                "Option": list(avg_ranks.keys()),
                "Average Interest Rank (1=low, 10=high)": list(avg_ranks.values()),
                "Responses": [counts[k] for k in avg_ranks.keys()],
            })
            .sort_values("Average Interest Rank (1=low, 10=high)", ascending=False)  # higher = better
            .reset_index(drop=True)
        )

        # Metric: most preferred (highest average)
        top_row = avg_df.iloc[0]
        st.metric(
            "Most preferred investment option",
            str(top_row["Option"]),
            delta=f"Avg rank: {top_row['Average Interest Rank (1=low, 10=high)']:.2f} • Total surveyee: {int(top_row['Responses'])}"
        )

        # Bar chart: sort by preference (highest first)
        fig = px.bar(
            avg_df,
            x="Option",
            y="Average Interest Rank (1=low, 10=high)",
            title="Average Interest Rank by Investment Option (higher = more preferred)",
        )
        fig.update_layout(xaxis_tickangle=30, yaxis_title="Average Interest Rank (1=low, 10=high)")
        st.plotly_chart(fig, use_container_width=True)

#st.divider()

# -----------------------------
# Chart 2: Average Salary by Education Level
# -----------------------------
st.subheader("💰 Average Salary by Education Level")
if col_salary in f_sal.columns and col_edu in f_sal.columns:
    grp = f_sal.groupby(col_edu, dropna=False)[col_salary].mean().reset_index()
    fig2 = px.bar(grp, x=col_edu, y=col_salary, color=col_salary, color_continuous_scale="Blues",
                  title="Average Salary by Education Level")
    fig2.update_layout(xaxis_title="Education Level", yaxis_title="Average Salary")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Education or Salary column not found in the salary dataset.")

st.divider()

# -----------------------------
# Chart 3: Savings Objectives by Gender (stacked)
# -----------------------------
st.subheader("🎯 Savings Objectives by Gender (stacked)")
if col_goals and col_gender in f_fin.columns and len(f_fin) > 0:
    tmp = f_fin[[col_gender, col_goals]].dropna().copy()
    tmp[col_goals] = split_multi(tmp[col_goals])
    tmp = tmp.explode(col_goals)
    tmp[col_goals] = tmp[col_goals].astype(str).str.strip()
    tmp = tmp[tmp[col_goals].str.len() > 0]
    if not tmp.empty:
        stack_df = tmp.value_counts().reset_index(name="Count")
        stack_df.columns = ["Gender", "Goal", "Count"]
        fig3 = px.bar(stack_df, x="Gender", y="Count", color="Goal", barmode="stack",
                      title="Savings Objectives by Gender")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No parsed goals for the current filters.")
else:
    st.info("Goal column not found or no finance rows under current filters.")

st.divider()

# -----------------------------
# Chart 4: Correlation Heatmap (clear labels)
# -----------------------------
st.subheader("🌡️ Correlation Heatmap (numeric features)")
fn = f_fin.copy().add_prefix("fin_")
sl = f_sal.copy().add_prefix("sal_")

# Numeric only and drop zero-variance columns
num_fn = [c for c in fn.columns if pd.api.types.is_numeric_dtype(fn[c])]
num_sl = [c for c in sl.columns if pd.api.types.is_numeric_dtype(sl[c])]
num_df = pd.concat([fn[num_fn], sl[num_sl]], axis=1)
# Drop columns with no variation (std == 0)
num_df = num_df.loc[:, num_df.std(numeric_only=True) > 0]

preferred_keys = ["age", "expected", "salary", "experience", "monitor", "invests", "rank", "return"]
prioritized = [c for c in num_df.columns if any(k in c.lower() for k in preferred_keys)]
base = num_df[prioritized] if len(prioritized) >= 2 else num_df

if not base.empty and base.shape[1] >= 2:
    corr = base.corr(numeric_only=True)
    # Pretty labels
    corr.index = [pretty_heat_label(i) for i in corr.index]
    corr.columns = [pretty_heat_label(i) for i in corr.columns]
    fig4 = px.imshow(corr, text_auto=".2f", aspect="auto",
                     title="Correlation between numeric features (finance + salary)")
    fig4.update_xaxes(tickangle=45)
    fig4.update_yaxes(tickangle=0, automargin=True)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Not enough informative numeric columns to render a heatmap.")

st.divider()

# -----------------------------
# Chart 5: Summary Statistics
# -----------------------------
st.subheader("🧮 Summary Statistics")
c1, c2 = st.columns(2)
with c1:
    fin_num = [c for c in f_fin.columns if pd.api.types.is_numeric_dtype(f_fin[c])]
    if fin_num:
        st.markdown("**Finance (numeric)**")
        st.dataframe(f_fin[fin_num].describe().T)
    else:
        st.info("No numeric finance columns under current filters.")
with c2:
    sal_num = [c for c in f_sal.columns if pd.api.types.is_numeric_dtype(f_sal[c])]
    if sal_num:
        st.markdown("**Salary (numeric)**")
        st.dataframe(f_sal[sal_num].describe().T)
    else:
        st.info("No numeric salary columns under current filters.")

st.divider()

# -----------------------------
# Chart 6: Expected Return by Age Group (box)
# -----------------------------
st.subheader("📦 Expected Return by Age Group (Finance)")
if col_return in f_fin.columns and col_agegrp in f_fin.columns:
    tmp = f_fin[[col_agegrp, col_return]].dropna()
    tmp[col_return] = pd.to_numeric(tmp[col_return], errors="coerce")
    tmp = tmp.dropna()
    if not tmp.empty:
        fig6 = px.box(tmp, x=col_agegrp, y=col_return, points="outliers",
                      title="Expected Return (%) by Age Group")
        fig6.update_layout(xaxis_title="Age Group", yaxis_title="Expected Return (%)")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("No numeric expected return data for the current filters.")
else:
    st.info("Expected Return or Age Group column not found.")

st.divider()

# -----------------------------
# Chart 7: Salary by Experience Band (histogram)
# -----------------------------
st.subheader("📊 Salary Distribution by Experience Band (Salary)")
if col_salary in f_sal.columns and col_expband in f_sal.columns:
    tmp = f_sal[[col_salary, col_expband]].dropna()
    if not tmp.empty:
        fig7 = px.histogram(tmp, x=col_salary, color=col_expband, nbins=40, barmode="overlay",
                            title="Salary Distribution by Experience Band")
        fig7.update_layout(xaxis_title="Salary", yaxis_title="Count")
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("No salary data for the current filters.")
else:
    st.info("Salary or Experience Band column not found.")

st.divider()

# -----------------------------
# Chart 8: Rank-1 Counts by Gender (grouped bar)
# -----------------------------
st.subheader("🏅 Rank-1 Counts by Gender (Finance)")
if rank_cols and col_gender in f_fin.columns and len(f_fin) > 0:
    rows = []
    for c in rank_cols:
        s = pd.to_numeric(f_fin[c], errors="coerce")
        opt = c.lower().replace("preference rank for ", "").replace(" investment", "").strip().title()
        gcounts = f_fin[s == 1][col_gender].value_counts(dropna=False)
        for g, cnt in gcounts.items():
            rows.append({"Option": opt, "Gender": str(g), "Rank1_Count": int(cnt)})
    rank_df = pd.DataFrame(rows)
    if not rank_df.empty:
        fig8 = px.bar(rank_df, x="Option", y="Rank1_Count", color="Gender", barmode="group",
                      title="Rank-1 (most preferred) counts by Gender")
        fig8.update_layout(xaxis_tickangle=30, xaxis_title="Investment Option", yaxis_title="Rank-1 Count")
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("No rank-1 preferences found for the current filters.")
else:
    st.info("Rank columns or Gender column not found, or no finance rows.")

st.divider()

# -----------------------------
# Downloads
# -----------------------------
st.subheader("⬇️ Download Current Filtered Data")
st.download_button("Download filtered Finance (CSV)", f_fin.to_csv(index=False).encode("utf-8"),
                   "finance_filtered.csv", "text/csv")
st.download_button("Download filtered Salary (CSV)", f_sal.to_csv(index=False).encode("utf-8"),
                   "salary_filtered.csv", "text/csv")
