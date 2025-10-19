# app.py
# -------------------------------------------------------------------
# Investment & Salary EDA Dashboard (Plotly-only, robust to messy data)
# + Seven Streamlit Tabs for guided exploration & storytelling
# -------------------------------------------------------------------
# Fixes included (from your version + extras):
#   • Fallback finder for Expected Return (prevents empty finance table)
#   • Popular Investment Options: auto-detects any "Preference rank for ... investment"
#   • Goals: split on comma/semicolon/pipe/slash
#   • Heatmap: drops zero-variance, caps to top-variance columns, cleans labels
#   • Extra filters: Experience Band, Education Level, Monitor Frequency,
#     Expected Return %, Tenure Preference, Savings Objectives
#   • Rank consistency: 1 = most preferred (Chart 1 + Rank-1 logic)
#   • Duplicate helpers removed; vectorized goal filtering
#
# Requires:
#   pip install streamlit pandas numpy plotly
# -------------------------------------------------------------------

# app.py
# -------------------------------------------------------------------
# Investment & Salary EDA Dashboard — Left Navigation Version
# -------------------------------------------------------------------
# Libraries required:
#   pip install streamlit pandas numpy plotly
# -------------------------------------------------------------------

# =============================================================
# MadeForYou Investment Insights Dashboard
# Full Streamlit App — All Pages + Comments
# =============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import re

from pathlib import Path


# -------------------------------------------------------------
# 1. APP CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(page_title="Investment Behavior Analysis", page_icon= "🪙", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* Turn any st.container(border=True) into a light-purple card */
section[data-testid="stContainer"][class*="st-emotion-cache"][style*="border: 1px"]{
  background: #F3EAFF;                /* very light purple */
  border: 1px solid #D9C7FF;          /* soft purple border */
  border-radius: 16px !important;
  box-shadow: 0 10px 24px rgba(92, 52, 179, .25),  /* outer shadow (3D) */
              inset 0 1px 0 rgba(255,255,255,.8);  /* subtle top highlight */
  padding: 16px;                       /* comfy inner spacing */
}

/* Optional: make plotly charts blend into the card */
div[data-testid="stPlotlyChart"] > div{
  background: transparent !important;
}

/* Optional: tidy section subtitles inside cards */
h3, h4 { margin-top: .25rem; }
</style>
""", unsafe_allow_html=True)



# -------------------------------------------------------------
# 2. LOAD DATASETS
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FIN_PATH = BASE_DIR / "Finance_Dataset_Cleaned.csv"
SAL_PATH = BASE_DIR / "Salary_Dataset_Cleaned.csv"

@st.cache_data
def load_data(fin_path, sal_path):
    """Load finance and salary datasets and clean column names."""
    finance = pd.read_csv(fin_path)
    salary = pd.read_csv(sal_path)
    finance.columns = finance.columns.str.strip()
    salary.columns = salary.columns.str.strip()
    return finance, salary

finance, salary = load_data(FIN_PATH, SAL_PATH)

# -------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------
def fcol(cols, key):
    """Find the first column name containing the given keyword."""
    key = key.lower()
    for c in cols:
        if key in c.lower():
            return c
    return None

def multi_match(cols, keys):
    """Find a column name matching any keyword from a list."""
    for k in keys:
        c = fcol(cols, k)
        if c:
            return c
    return None

def to_age_group(x):
    """Convert numeric age values into defined age groups."""
    try:
        a = float(x)
    except Exception:
        return "Unknown"
    if pd.isna(a): return "Unknown"
    a = int(a)
    if a <= 24: return "18-24"
    if a <= 34: return "25-34"
    if a <= 44: return "35-44"
    if a <= 54: return "45-54"
    return "55+"

def pretty_heat_label(name: str) -> str:
    """Clean noisy column labels for heatmap axes."""
    n = name
    n = n.replace("fin_", "").replace("sal_", "")
    n = re.sub(r"Preference\s*rank\s*for\s+(.*?)\s+investment", r"Rank: \1", n, flags=re.I)
    n = n.replace("_", " ")
    n = n.title()
    return n

def make_unique(labels):
    """Ensure axis labels are unique for heatmap."""
    seen = {}
    out = []
    for lbl in labels:
        if lbl in seen:
            seen[lbl] += 1
            out.append(f"{lbl} ({seen[lbl]})")
        else:
            seen[lbl] = 1
            out.append(lbl)
    return out

# -------------------------------------------------------------
# 4. IDENTIFY IMPORTANT COLUMNS
# -------------------------------------------------------------
col_gender  = multi_match(finance.columns, ["gender", "sex"])
col_age     = multi_match(finance.columns, ["age"])
col_agegrp  = fcol(finance.columns, "age_group")
col_return  = multi_match(finance.columns, ["expected return", "return"])
col_monitor = multi_match(finance.columns, ["monitor"])
col_tenure  = multi_match(finance.columns, ["tenure", "keep your money"])
col_invtype = multi_match(finance.columns, ["investment type", "type of investment"])
col_salary  = multi_match(salary.columns, ["salary", "income"])
col_edu     = multi_match(salary.columns, ["education"])
col_expband = fcol(salary.columns, "experience band")
col_expyrs  = multi_match(salary.columns, ["years of experience", "experience"])

# If Age_Group missing, create it
if col_agegrp not in finance.columns and col_age in finance.columns:
    finance["Age_Group"] = finance[col_age].apply(to_age_group)
    col_agegrp = "Age_Group"

# -------------------------------------------------------------
# 5. SIDEBAR CUSTOM STYLING
# -------------------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #E0E0E0;
            padding-top: 0 !important; 
            padding-right: 1rem;
            padding-left: 1rem;
        }
        .nav-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #5bc0be;
            text-align: center;
            margin-bottom: 0.3rem;
        }
        .nav-subtitle {
            font-size: 0.8rem;
            color: #cfcfcf;
            text-align: center;
            margin-bottom: 1.5rem;
            letter-spacing: 1px;
        }
        .nav-button {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 10px;
            background-color: #1c2541;
            border: 1px solid rgba(255,255,255,0.08);
            color: #e0e0e0;
            font-size: 0.9rem;
            font-weight: 500;
            padding: 0.6rem 0.8rem;
            border-radius: 10px;
            margin-bottom: 0.4rem;
            transition: all 0.2s ease-in-out;
            cursor: pointer;
            width: 100%;
        }
        .nav-button:hover {
            background-color: #3a506b;
            border-color: #5bc0be;
            color: #ffffff;
            transform: translateX(3px);
        }
        .nav-active {
            background-color: #5bc0be !important;
            color: #fff !important;
            border: 1px solid #5bc0be !important;
        }
        .sidebar-divider {
            border-top: 1px solid rgba(255,255,255,0.1);
            margin: 1rem 0;
        }
        .filter-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 0.8rem;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* RADIO BASE — tidy spacing */
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
  gap: 6px !important;
}

/* EACH OPTION CONTAINER */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border: 1px solid rgba(91,192,190,.35);
  border-radius: 10px;
  padding: 8px 10px;
  background: #ffffff;              /* default (matches your theme) */
  color: #e0e0e0;
  transition: background .15s ease, transform .15s ease, border-color .15s ease;
}

/* HOVER */
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #3a506b;
  border-color: #5bc0be;
  transform: translateX(3px);
}

/* ACTIVE / SELECTED STATE — highlight current page */
[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"],
/* Fallback selector (older Streamlit builds) */
[data-testid="stSidebar"] [data-testid="stRadio"] div[aria-checked="true"] label {
  background: #5bc0be !important;   /* your accent color */
  border-color: #5bc0be !important;
  color: #ffffff !important;
  box-shadow: 0 6px 16px rgba(91,192,190,.35);
}

/* Make the emoji + text align nicely */
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
  margin: 0;
  font-weight: 600;
  letter-spacing: .2px;
}

/* Remove the default radio circle (purely visual; click still works) */
[data-testid="stSidebar"] [data-testid="stRadio"] svg {
  display: none !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 6. SIDEBAR NAVIGATION + FILTERS
# -------------------------------------------------------------
#st.sidebar.markdown('<div class="nav-title">MadeForYou', unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <style>
      :root {
        --border-color: rgba(180, 200, 220, 0.25);
      }

      .iba-title {
        font-size: 22px;
        font-weight: 800;
        color: #2C3539 !important;
        letter-spacing: -0.02em;
        line-height: 0.2;
        margin: 0;
        padding: 0;
        text-align: center;
      }

      .iba-subtitle {
        font-size: 13px;
        font-weight: 600;
        color: #2C3539 !important;
        letter-spacing: 0.05em;
        opacity: 1 !important;
        text-align: center;
      }
    </style>

    <div style="padding: 8px 16px 12px 16px; border-bottom: 1px solid var(--border-color); margin-bottom: 8px;">
      <div style="display: flex; flex-direction: column; align-items: center; gap: 0px;">
        <span style="font-size: 48px; line-height: 1; filter: drop-shadow(0 0 12px rgba(0, 217, 255, 0.5));">🪙</span>
        <h1 class="iba-title">Investment Behavior Analysis</h1>
        <div class="iba-subtitle">People.Investment.Patterns</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Navigation items
nav_items = {
    "Overview Dashboard": "",
    "Investment Behavior": "",
    "Salary & Education Insights": "",
    "Income & Investment Relationship": "",
    "Demographics & Behavioral Patterns": "",
    "Interactive Data Explorer": "",
    "Insights & Storytelling": ""
}

# Handle navigation state
if "page" not in st.session_state:
    st.session_state.page = list(nav_items.keys())[0]

# --- Left nav: radio with active highlight ---
pages = list(nav_items.keys())
# keep current page on reruns
if "page" not in st.session_state:
    st.session_state.page = pages[0]

# compute current index so radio highlights the active one
_current_idx = pages.index(st.session_state.page)

choice = st.sidebar.radio(
    label="Navigation",
    options=pages,
    index=_current_idx,
    format_func=lambda k: f"{nav_items[k]}  {k}",
    label_visibility="collapsed",
    key="nav_radio",
)
st.session_state.page = choice  # update current page

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="filter-header">Filter</div>', unsafe_allow_html=True)

# --- Sidebar filters ---
gender_vals = sorted(finance[col_gender].dropna().unique()) if col_gender else []
sel_gender = st.sidebar.multiselect("Gender", gender_vals, default=[])

age_vals = sorted(finance[col_agegrp].dropna().unique()) if col_agegrp else []
sel_age = st.sidebar.multiselect("Age Group", age_vals, default=[])

edu_vals = sorted(salary[col_edu].dropna().unique()) if col_edu else []
sel_edu = st.sidebar.multiselect("Education Level", edu_vals, default=[])

expband_vals = sorted(salary[col_expband].dropna().unique()) if col_expband else []
sel_expband = st.sidebar.multiselect("Experience Band", expband_vals, default=[])

# Apply filters to data
f_fin = finance.copy()
if col_gender and sel_gender: f_fin = f_fin[f_fin[col_gender].isin(sel_gender)]
if col_agegrp and sel_age: f_fin = f_fin[f_fin[col_agegrp].isin(sel_age)]

f_sal = salary.copy()
if col_edu and sel_edu: f_sal = f_sal[f_sal[col_edu].isin(sel_edu)]
if col_expband and sel_expband: f_sal = f_sal[f_sal[col_expband].isin(sel_expband)]

# -------------------------------------------------------------
# 7. MAIN CONTENT AREA
# -------------------------------------------------------------
page = st.session_state.page
st.title(page)

# =============================================================
# PAGE 1: OVERVIEW DASHBOARD
# =============================================================
if page == "Overview Dashboard":
    st.subheader("Key Metrics")

    # -----------------------------
    # KPI: Key Metrics
    # -----------------------------
    total_respondents = len(finance)
    avg_salary = pd.to_numeric(salary[col_salary], errors="coerce").mean() if col_salary else np.nan
    avg_return = pd.to_numeric(finance[col_return], errors="coerce").mean() if col_return else np.nan
    common_edu = salary[col_edu].mode().iat[0] if col_edu and not salary[col_edu].dropna().empty else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Survey Respondents", f"{total_respondents:,}")
    with c2: st.metric("Average Salary", f"{avg_salary:,.0f}" if pd.notna(avg_salary) else "—")
    with c3: st.metric("Highest Investors (Gender)", (lambda s: f"{s.value_counts().idxmax()} ({(s.value_counts().max()*100.0/s.size):.1f}%)" if s.size else "—")(finance[col_gender].dropna() if (col_gender and (col_gender in finance.columns)) else pd.Series(dtype=object)))


    with c4: st.metric("Most Common Education Level", common_edu)

    st.divider()
    st.subheader("Data Overview Visualizations")

    left, right = st.columns(2)

    # -----------------------------
    # Pie chart: Most popular investment type
    # -----------------------------
    with left:
        rank_like = [
            c for c in finance.columns
            if re.search(r"^preference\s*rank\s*for\s+.+\s+investment$", c, flags=re.I)
        ]

        pie_df = None
        if rank_like:
            rows = []
            for colr in rank_like:
                s = pd.to_numeric(finance[colr], errors="coerce")
                cnt = int((s == 1).sum())
                m = re.search(r"preference\s*rank\s*for\s+(.+)\s+investment", colr, flags=re.I)
                opt = m.group(1).title().strip() if m else colr.title()
                rows.append({"Option": opt, "Rank1_Count": cnt})
            pie_df = pd.DataFrame(rows)
            pie_df = pie_df[pie_df["Rank1_Count"] > 0]
        else:
            inv_col = None
            for c in finance.columns:
                lc = c.lower()
                if ("invest" in lc or "investment" in lc) and ("rank" not in lc):
                    inv_col = c
                    break
            if inv_col and not finance[inv_col].dropna().empty:
                pie_df = finance[inv_col].value_counts(dropna=False).reset_index()
                pie_df.columns = ["Option", "Rank1_Count"]

        if pie_df is not None and not pie_df.empty:
            fig_pie = px.pie(
                pie_df,
                names="Option",
                values="Rank1_Count",
                title="Most Popular Investment Type",
                color_discrete_sequence=px.colors.sequential.Blues
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown(
                "This pie chart shows which investment types people like the most. "
                "Each slice is like a piece of a cake — the bigger the slice, the more people prefer that option!"
            )
        else:
            st.info("No usable investment preference columns found for the pie chart.")

    # -----------------------------
    # Bar chart: Average salary by education level
    # -----------------------------
    with right:
        if col_edu and col_salary and not f_sal.empty:
            avg_sal = (
                f_sal.groupby(col_edu, dropna=False)[col_salary]
                .mean()
                .reset_index()
                .sort_values(by=col_salary, ascending=False)
            )
            fig_bar = px.bar(
                avg_sal, x=col_edu, y=col_salary,
                title="Average Salary by Education Level",
                color=col_salary, color_continuous_scale="Blues"
            )
            fig_bar.update_layout(xaxis_title="Education Level", yaxis_title="Average Salary")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown(
                "This bar chart tells us how much money people earn on average "
                "depending on their education level. The taller the bar, the higher the salary. "
                "It’s like comparing steps on a ladder — more education can often mean a higher step!"
            )
        else:
            st.info("Education or Salary column not found / no rows after filters.")

    st.divider()
    st.subheader("Correlation Heatmap (Age, Salary, Experience, Expected Return)")

    # -----------------------------
    # Heatmap: correlation logic
    # -----------------------------
    fn = f_fin.copy().add_prefix("fin_")
    sl = f_sal.copy().add_prefix("sal_")

    num_fn = [c for c in fn.columns if pd.api.types.is_numeric_dtype(fn[c])]
    num_sl = [c for c in sl.columns if pd.api.types.is_numeric_dtype(sl[c])]
    num_df = pd.concat([fn[num_fn], sl[num_sl]], axis=1)
    num_df = num_df.loc[:, num_df.std(numeric_only=True) > 0]

    preferred_keys = ["age", "expected", "salary", "experience", "monitor", "invests", "rank", "return"]
    prioritized = [c for c in num_df.columns if any(k in c.lower() for k in preferred_keys)]
    base = num_df[prioritized] if len(prioritized) >= 2 else num_df

    if not base.empty and base.shape[1] >= 2:
        corr = base.corr(numeric_only=True)
        x_labels = make_unique([pretty_heat_label(c) for c in corr.columns])
        y_labels = make_unique([pretty_heat_label(r) for r in corr.index])

        fig_hm = px.imshow(
            corr.to_numpy(),
            x=x_labels, y=y_labels,
            text_auto=".2f",
            aspect="auto",
            title="Correlation between numeric features (Finance + Salary)"
        )
        fig_hm.update_xaxes(tickangle=45)
        fig_hm.update_yaxes(tickangle=0, automargin=True)
        st.plotly_chart(fig_hm, use_container_width=True)
        
        st.markdown("""
            ### Understanding the Correlation Heatmap
                This chart shows how different numerical factors are connected in your dataset — like Age, Experience, Salary, and Expected Return.

                Each square compares two factors and tells you how strongly they move together:
                    - Dark Blue → Strong positive relationship (when one increases, the other also increases)
                    - Light Blue / White → Weak or no relationship
                    - Very Light / Opposite Shades → Negative relationship (when one increases, the other decreases)

            ### What You Can See:
                - Age ↔ Experience: Older people usually have more years of experience → Strong positive relationship.  
                - Experience ↔ Salary: More experience often means a higher salary → Moderate to strong positive correlation.  
                - Age ↔ Salary: Older respondents generally earn a bit more → Moderate positive relationship.  
                - Age ↔ Expected Return: Younger people expect higher returns, older people prefer safer options → Slight negative correlation.  
                - Experience ↔ Expected Return: Experienced professionals have more realistic return goals → Weak negative relationship.  
                - Salary ↔ Expected Return:** Higher-income respondents prefer stability → Weak negative correlation.

            ### Quick Summary:
                | Variables Compared | What It Means | Correlation Strength |
                |--------------------|----------------|----------------------|
                | Age & Experience | Older respondents have more experience | Strong |
                | Experience & Salary | More experience → higher salary | Moderate–Strong |
                | Age & Salary | Age often relates to salary | Moderate |
                | Age & Expected Return | Younger = higher return expectations | Weak–Negative |
                | Experience & Expected Return | More experience = realistic goals | Weak–Negative |
                | Salary & Expected Return | Higher salary = safer investment choices | Weak–Negative |

            > This heatmap helps you understand how professional growth, age, and income patterns connect to financial expectations.""")
    else:
        st.info("Not enough informative numeric columns to render a heatmap under current filters.")

    st.divider()

    # -----------------------------
    # Summary Statistics (Numeric Fields)
    # -----------------------------
    st.subheader("Summary Statistics (Numeric Fields)")

    numeric_cols = finance.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        desc = numeric_cols.describe().T
        desc["median"] = numeric_cols.median()
        st.dataframe(desc[["mean", "median", "min", "max"]].round(2))
        st.markdown("""
                This table summarizes important numerical data. It shows:  
                    - **Mean:** The average value  
                    - **Median:** The middle value  
                    - **Min/Max:** The smallest and largest numbers

            For example, in salary data:
                - The *mean* tells you the average income.  
                - The *min* and *max* show the lowest and highest salaries recorded.  
            This helps compare ranges and understand overall data distribution.""")
    else:
        st.warning("No numeric columns available for summary statistics.")


# =============================================================
# PAGE 2: INVESTMENT BEHAVIOR
# =============================================================
elif page == "Investment Behavior":
    #st.subheader("Investment Behavior Analysis")

    # ---------- Helper: robust numeric parser for "Expected Return %" ----------
    # Extracts the FIRST number from strings like "10%", "8–12%", "~12", "10 to 12".
    import math
    num_pat = re.compile(r"(-?\d+(?:\.\d+)?)")
    def parse_expected_return(series: pd.Series) -> pd.Series:
        def _grab_first_num(x):
            if pd.isna(x): return np.nan
            s = str(x)
            m = num_pat.findall(s)
            if not m: return np.nan
            try:
                v = float(m[0])  # take first numeric token
            except Exception:
                return np.nan
            # sanity clamp for % values
            if v < 0 or v > 200:
                return np.nan
            return v
        return series.apply(_grab_first_num).astype(float)

    # ---------------------------
    # Page-level filters (collapsed + empty by default)
    # ---------------------------
    with st.expander("Refine Page Filters", expanded=False):
        # Build option lists from globally filtered frames (f_fin / f_sal)
        _age_opts     = sorted(f_fin[col_agegrp].dropna().unique()) if col_agegrp else []
        _gen_opts     = sorted(f_fin[col_gender].dropna().unique()) if col_gender else []
        _edu_opts     = sorted(f_sal[col_edu].dropna().unique())     if col_edu else []
        _tenure_opts  = sorted(f_fin[col_tenure].dropna().unique())  if col_tenure else []
        _mon_opts     = sorted(f_fin[col_monitor].dropna().unique()) if col_monitor else []

        c1, c2, c3 = st.columns(3)

        # NOTE: default=[] → nothing selected initially (no extra page-level filtering)
        with c1:
            _flt_age = st.multiselect(
                "Age Group", _age_opts, default=[], key="p2_age"
            ) if _age_opts else []
            _flt_gender = st.multiselect(
                "Gender", _gen_opts, default=[], key="p2_gender"
            ) if _gen_opts else []
        with c2:
            _flt_edu = st.multiselect(
                "Education", _edu_opts, default=[], key="p2_edu"
            ) if _edu_opts else []
            _flt_tenure = st.multiselect(
                "Tenure Preference", _tenure_opts, default=[], key="p2_tenure"
            ) if _tenure_opts else []
        with c3:
            _flt_monitor = st.multiselect(
                "Monitor Frequency", _mon_opts, default=[], key="p2_monitor"
            ) if _mon_opts else []

    # Apply page-level filters on top of global ones (f_fin/f_sal)
    ib_fin = f_fin.copy()
    ib_sal = f_sal.copy()
    if col_agegrp  and _flt_age:     ib_fin = ib_fin[ib_fin[col_agegrp].isin(_flt_age)]
    if col_gender  and _flt_gender:  ib_fin = ib_fin[ib_fin[col_gender].isin(_flt_gender)]
    if col_tenure  and _flt_tenure:  ib_fin = ib_fin[ib_fin[col_tenure].isin(_flt_tenure)]
    if col_monitor and _flt_monitor: ib_fin = ib_fin[ib_fin[col_monitor].isin(_flt_monitor)]
    if col_edu     and _flt_edu:     ib_sal = ib_sal[ib_sal[col_edu].isin(_flt_edu)]

    # Pre-compute a CLEAN expected-return series we can reuse below
    clean_return = None
    if col_return and (col_return in ib_fin.columns):
        clean_return = parse_expected_return(ib_fin[col_return])

    

    # ---------------------------
    # Chart 1: Risk Appetite by Age Group (stacked bar)
    # ---------------------------
    st.markdown("### Risk Appetite by Age Group")
    if (col_agegrp in ib_fin.columns) and (clean_return is not None):
        tmp = pd.DataFrame({
            col_agegrp: ib_fin[col_agegrp],
            "ExpectedReturnPct": clean_return
        }).dropna(subset=["ExpectedReturnPct"])
        if not tmp.empty:
            # Create coarse risk bands based on expected % return
            tmp["Risk Appetite"] = pd.cut(
                tmp["ExpectedReturnPct"],
                bins=[-np.inf, 6, 10, np.inf],
                labels=["Low (<6%)", "Moderate (6–10%)", "Aggressive (>10%)"]
            )
            risk_ct = tmp.value_counts([col_agegrp, "Risk Appetite"]).reset_index(name="Count")
            fig1 = px.bar(
                risk_ct, x=col_agegrp, y="Count", color="Risk Appetite",
                barmode="stack", title="Risk Appetite by Age Group"
            )
            fig1.update_layout(xaxis_title="Age Group", yaxis_title="Respondents")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** Respondents grouped into **Low / Moderate / Aggressive** bands by their expected % return.\n"
                "- **How to read:** Taller **Aggressive** segments for younger groups suggest **higher risk appetite**.\n"
                "- **Tip:** Use filters to compare two age bands (e.g., 25–34 vs 45–54) for a sharper contrast."
            )
        else:
            st.info("No numeric expected return values under current filters (after cleaning).")
    else:
        st.info("Expected Return or Age Group column not found.")

    st.divider()

    # ---------------------------
    # Chart 2: Tenure Preference by Age Group (grouped bar)
    # ---------------------------
    st.markdown("### Preferred Tenure Duration by Age Group")
    if col_tenure and (col_tenure in ib_fin.columns) and (col_agegrp in ib_fin.columns):
        tenure_df = ib_fin[[col_agegrp, col_tenure]].dropna()
        if not tenure_df.empty:
            ct = tenure_df.value_counts([col_agegrp, col_tenure]).reset_index(name="Count")
            fig2 = px.bar(
                ct, x=col_agegrp, y="Count", color=col_tenure, barmode="group",
                title="Tenure Preference by Age Group"
            )
            fig2.update_layout(xaxis_title="Age Group", yaxis_title="Respondents", legend_title="Tenure")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** How **investment horizon** (short/medium/long) varies across ages.\n"
                "- **How to read:** If **older groups** show higher long-term bars, it indicates **greater patience/stability**.\n"
                "- **Tip:** Add a **Gender** filter to see tenure differences within the same age band."
            )
        else:
            st.info("No tenure preference data under the current filters.")
    else:
        st.info("Tenure preference or Age Group column not found.")

    st.divider()

    # ---------------------------
    # Chart 3: Monitoring Frequency by Gender (line with markers)
    # ---------------------------
    st.markdown("### Portfolio Monitoring Frequency by Gender")
    if col_monitor and col_gender and (col_monitor in ib_fin.columns) and (col_gender in ib_fin.columns):
        freq_df = ib_fin[[col_monitor, col_gender]].dropna().copy()
        if not freq_df.empty:
            ct = freq_df.value_counts([col_monitor, col_gender]).reset_index(name="Count")
            # Sort x-axis into a logical cadence
            ord_map = {
                "Daily": 1, "Weekly": 2, "Bi-Weekly": 3, "Monthly": 4, "Quarterly": 5,
                "Half-Yearly": 6, "Yearly": 7, "Rarely": 8, "Never": 9
            }
            ct["Order"] = ct[col_monitor].map(lambda x: ord_map.get(str(x).title(), 99))
            ct = ct.sort_values("Order")
            fig3 = px.line(
                ct, x=col_monitor, y="Count", color=col_gender, markers=True,
                title="Monitoring Frequency by Gender"
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** How often each gender **checks portfolios**.\n"
                "- **How to read:** Peaks at **Weekly/Monthly** indicate a common cadence; separation by color shows **diligence differences**.\n"
                "- **Tip:** Add an **Age Group** filter to see if monitoring patterns change with age."
            )
        else:
            st.info("No monitoring frequency data under the current filters.")
    else:
        st.info("Monitor frequency or Gender column not found.")

    st.divider()

    # ---------------------------
    # Chart 4: Expected Return (%) by Education Level (violin + box)
    # ---------------------------
    st.markdown("### Expected Return (%) by Education Level")
    if col_edu and (col_edu in ib_sal.columns) and (clean_return is not None):
        # Align education (salary table) with cleaned expected return (finance table) by index
        merged = pd.DataFrame({
            col_edu: ib_sal[col_edu]
        }).join(
            pd.DataFrame({"ExpectedReturnPct": clean_return})
        ).dropna()
        if not merged.empty:
            fig4 = px.violin(
                merged, x=col_edu, y="ExpectedReturnPct", box=True, points="all",
                title="Expected Return by Education Level"
            )
            fig4.update_layout(xaxis_title="Education Level", yaxis_title="Expected Return (%)")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** Distribution (spread + median) of **return expectations** by education.\n"
                "- **How to read:** Compact violins/boxes suggest **consistent expectations**; wide shapes imply **greater variability**.\n"
                "- **Tip:** If outliers appear, narrow **Age Group** or **Gender** to isolate cohort effects."
            )
        else:
            st.info("No numeric expected return values after cleaning / merging.")
    else:
        st.info("Education column not found or could not parse expected return.")

    st.divider()

    # ---------------------------
    # Insights (answers your 3 questions)
    # ---------------------------
    with st.container(border=True):
        st.markdown("#### Insights Summary")

        insights = []

        # 1) Are younger respondents more aggressive investors?
        if (col_agegrp in ib_fin.columns) and (clean_return is not None):
            tmp2 = pd.DataFrame({
                col_agegrp: ib_fin[col_agegrp],
                "ExpectedReturnPct": clean_return
            }).dropna(subset=["ExpectedReturnPct"])
            if not tmp2.empty:
                tmp2["Risk Appetite"] = pd.cut(
                    tmp2["ExpectedReturnPct"],
                    bins=[-np.inf, 6, 10, np.inf],
                    labels=["Low (<6%)", "Moderate (6–10%)", "Aggressive (>10%)"]
                )
                young = tmp2[tmp2[col_agegrp].isin(["18-24", "25-34"])]
                older = tmp2[~tmp2[col_agegrp].isin(["18-24", "25-34"])]
                if not young.empty and not older.empty:
                    y_aggr = 100 * (young["Risk Appetite"] == "Aggressive (>10%)").mean()
                    o_aggr = 100 * (older["Risk Appetite"] == "Aggressive (>10%)").mean()
                    insights.append(f"- **Younger investors** show more aggressive targets: {y_aggr:.1f}% vs {o_aggr:.1f}% (35+).")

        # 2) Do men/women differ in preferred investment monitoring?
        if col_monitor and col_gender and (col_monitor in ib_fin.columns) and (col_gender in ib_fin.columns):
            freq_df2 = ib_fin[[col_monitor, col_gender]].dropna()
            if not freq_df2.empty:
                top_by_gender = (
                    freq_df2.value_counts([col_gender, col_monitor])
                            .groupby(level=0)
                            .apply(lambda s: (s/s.sum()).idxmax()[1])
                )
                if not top_by_gender.empty:
                    msg = "; ".join([f"{g}: {opt}" for g, opt in top_by_gender.items()])
                    insights.append(f"- **Most common monitoring cadence by gender:** {msg}.")

        # 3) How often do investors check their portfolios? (overall)
        if col_monitor and (col_monitor in ib_fin.columns):
            mf = ib_fin[col_monitor].dropna()
            if not mf.empty:
                top_freq = mf.value_counts().idxmax()
                insights.append(f"- **Overall most common monitoring frequency:** {top_freq}.")

        if insights:
            st.markdown("\n".join(insights))
        else:
            st.markdown("- No clear patterns under the current filters. Try broadening the selection.")


# =============================================================
# PAGE 3: SALARY & EDUCATION INSIGHTS
# =============================================================
elif page == "Salary & Education Insights":
    #st.subheader("Salary & Education Insights")

    # ---------- Helpers ----------
    def _to_num(s: pd.Series) -> pd.Series:
        """Safe numeric conversion for mixed-type columns (returns float)."""
        return pd.to_numeric(s, errors="coerce")

    def _mid_from_band(txt):
        """
        Convert Experience Band text (e.g., '0-1 yrs', '3–5 years', '20+ yrs')
        into a numeric midpoint (float). If we can’t parse, return NaN.
        """
        if pd.isna(txt): 
            return np.nan
        t = str(txt).lower().strip()
        # 20+ yrs -> 22 (small cushion)
        m = re.match(r"(\d+)\s*\+\s*", t)
        if m:
            return float(m.group(1)) + 2.0
        # 3-5 years -> 4
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)", t)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return (a + b) / 2.0
        # 'less than 1 year' -> 0.5
        if "less" in t and "1" in t:
            return 0.5
        # plain number fallback
        m = re.search(r"(\d+(?:\.\d+)?)", t)
        if m:
            return float(m.group(1))
        return np.nan

    # ---------- Join selected fields from Finance (by index) ----------
    fin_join = {}
    if col_gender and (col_gender in finance.columns):
        fin_join["__Gender"] = finance[col_gender]
    if col_agegrp and (col_agegrp in finance.columns):
        fin_join["__Age_Group"] = finance[col_agegrp]
    if col_age and (col_age in finance.columns):
        fin_join["__Age_Num"] = _to_num(finance[col_age])

    sal_base = f_sal.copy()
    if fin_join:
        sal_base = sal_base.join(pd.DataFrame(fin_join))

    # ---------- Derive numeric Years of Experience ----------
    years_col = None
    if col_expyrs and (col_expyrs in sal_base.columns):
        years_col = "_Years"
        sal_base[years_col] = _to_num(sal_base[col_expyrs])
    elif col_expband and (col_expband in sal_base.columns):
        years_col = "_Years"
        sal_base[years_col] = sal_base[col_expband].apply(_mid_from_band)

    # ---------- Page-level filters (collapsed, empty by default) ----------
    with st.expander("Refine Page Filters", expanded=False):
        _gender_opts  = sorted(sal_base["__Gender"].dropna().unique()) if "__Gender" in sal_base.columns else []
        _edu_opts     = sorted(sal_base[col_edu].dropna().unique())    if col_edu and (col_edu in sal_base.columns) else []
        _expband_opts = sorted(sal_base[col_expband].dropna().unique()) if col_expband and (col_expband in sal_base.columns) else []

        c1, c2, c3 = st.columns(3)
        with c1:
            _flt_gender  = st.multiselect("Gender", _gender_opts, default=[], key="p3_gender") if _gender_opts else []
        with c2:
            _flt_edu     = st.multiselect("Education Level", _edu_opts, default=[], key="p3_edu") if _edu_opts else []
        with c3:
            _flt_expband = st.multiselect("Experience Band", _expband_opts, default=[], key="p3_expband") if _expband_opts else []

    # Apply page-level filters
    ib_sal = sal_base.copy()
    if "__Gender" in ib_sal.columns and _flt_gender:
        ib_sal = ib_sal[ib_sal["__Gender"].isin(_flt_gender)]
    if col_edu and (col_edu in ib_sal.columns) and _flt_edu:
        ib_sal = ib_sal[ib_sal[col_edu].isin(_flt_edu)]
    if col_expband and (col_expband in ib_sal.columns) and _flt_expband:
        ib_sal = ib_sal[ib_sal[col_expband].isin(_flt_expband)]

    # ---------------------------
    # Chart 1: Histogram — Salary distribution by experience band
    # ---------------------------
    st.markdown("### Salary Distribution by Experience Band")
    if col_salary and col_expband and (col_salary in ib_sal.columns) and (col_expband in ib_sal.columns):
        tmp = ib_sal[[col_salary, col_expband]].dropna()
        if not tmp.empty:
            fig_h = px.histogram(tmp, x=col_salary, color=col_expband, nbins=30,
                                 title="Salary Distribution by Experience Band")
            fig_h.update_layout(xaxis_title="Salary", yaxis_title="Respondents", legend_title="Experience Band")
            st.plotly_chart(fig_h, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** The **spread** of salaries within each **experience band**.\n"
                "- **How to read:** More right-tail for senior bands ⇒ **higher earning potential** with experience.\n"
                "- **Tip:** Filter to a single education level to compare distributions more cleanly."
            )
        else:
            st.info("No rows available for the histogram under current filters.")
    else:
        st.info("Required columns for the histogram are missing.")

    st.divider()

    # ---------------------------
    # Chart 2: Bar — Average salary by education level
    # ---------------------------
    st.markdown("### Average Salary by Education Level")
    if col_salary and col_edu and (col_salary in ib_sal.columns) and (col_edu in ib_sal.columns):
        tmp = ib_sal[[col_edu, col_salary]].dropna()
        if not tmp.empty:
            tmp[col_salary] = _to_num(tmp[col_salary])
            avg_sal = tmp.groupby(col_edu, dropna=False)[col_salary].mean().reset_index()
            fig_bar = px.bar(avg_sal, x=col_edu, y=col_salary,
                             title="Average Salary by Education Level",
                             color=col_salary, color_continuous_scale="Blues")
            fig_bar.update_layout(xaxis_title="Education Level", yaxis_title="Average Salary")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** Mean earnings by **education**.\n"
                "- **How to read:** Taller bars suggest higher average pay; confirms/denies **education → salary**."
            )
        else:
            st.info("No rows for the education bar chart under current filters.")
    else:
        st.info("Required columns for the education bar chart are missing.")

    st.divider()

    # ---------------------------
    # Chart 3: Box — Salary by Age Group (Age_Group from Finance)
    # ---------------------------
    st.markdown("### Salary by Age Group (Box Plot)")
    if col_salary and (col_salary in ib_sal.columns) and ("__Age_Group" in ib_sal.columns):
        tmp = ib_sal[["__Age_Group", col_salary]].dropna()
        if not tmp.empty:
            tmp[col_salary] = _to_num(tmp[col_salary])
            fig_box = px.box(tmp, x="__Age_Group", y=col_salary, title="Salary by Age Group")
            fig_box.update_layout(xaxis_title="Age Group", yaxis_title="Salary")
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** Salary **distribution** within each **age group**.\n"
                "- **How to read:** Compare medians/IQRs to find **which ages earn the most** and where variability is highest."
            )
        else:
            st.info("No rows for the age-group box plot under current filters.")
    else:
        st.info("Age Group (from finance) or Salary column not available for the box plot.")

    st.divider()

    # ---------------------------
    # Chart 4: Correlation Heatmap — Salary, Experience (years), Age
    # ---------------------------
    st.markdown("### Correlation: Salary, Experience, and Age")
    heat_df = pd.DataFrame()
    if col_salary and (col_salary in ib_sal.columns):
        heat_df["Salary"] = _to_num(ib_sal[col_salary])
    if years_col and (years_col in ib_sal.columns):
        heat_df["ExperienceYears"] = _to_num(ib_sal[years_col])
    if "__Age_Num" in ib_sal.columns:
        heat_df["Age"] = _to_num(ib_sal["__Age_Num"])

    heat_df = heat_df.dropna(how="all")
    if not heat_df.empty:
        nz = heat_df.loc[:, heat_df.std(numeric_only=True) > 0]
        if nz.shape[1] >= 2:
            corr = nz.corr(numeric_only=True)
            fig_hm = px.imshow(corr, text_auto=".2f", aspect="auto",
                               title="Correlation: Salary, Experience, and Age")
            fig_hm.update_xaxes(tickangle=45)
            st.plotly_chart(fig_hm, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** Pairwise **correlations** among the three metrics.\n"
                "- **How to read:** Positive values mean the variables rise together; larger magnitudes = **stronger ties**."
            )
        else:
            st.info("Not enough informative numeric columns to render the correlation heatmap.")
    else:
        st.info("No numeric data available for the correlation heatmap under current filters.")

    st.divider()

    # ---------------------------
    # Chart 5: Scatter — Salary vs Years of Experience (trendline)
    # ---------------------------
    st.markdown("### Salary vs Years of Experience (Scatter + Trendline)")
    if col_salary and (col_salary in ib_sal.columns) and years_col and (years_col in ib_sal.columns):
        sc = pd.DataFrame({
            "Salary": _to_num(ib_sal[col_salary]),
            "ExperienceYears": _to_num(ib_sal[years_col]),
        })
        if col_edu and (col_edu in ib_sal.columns):
            sc[col_edu] = ib_sal[col_edu]
        sc = sc.dropna(subset=["Salary", "ExperienceYears"])
        sc = sc[np.isfinite(sc["Salary"]) & np.isfinite(sc["ExperienceYears"])]
        if not sc.empty:
            scatter_kwargs = dict(
                x="ExperienceYears",
                y="Salary",
                trendline="ols",
                title="Salary vs Years of Experience (with Trendline)",
            )
            if col_edu and (col_edu in sc.columns):
                scatter_kwargs["color"] = col_edu
            fig_sc = px.scatter(sc, **scatter_kwargs)
            fig_sc.update_layout(xaxis_title="Years of Experience", yaxis_title="Salary")
            st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** How **experience** relates to **salary**, with an OLS trendline.\n"
                "- **How to read:** An **upward** slope suggests pay rises with experience; coloring shows differences by **education**.\n"
                "- **Tip:** Filter by education or gender to see how the slope changes across cohorts."
            )
        else:
            st.info("No numeric data available for the scatter under current filters after cleaning.")
    else:
        st.info("Salary or a usable Years-of-Experience measure is not available for the scatter plot.")

    st.divider()

    # ---------------------------
    # Insights — answers the 3 questions
    # ---------------------------
    with st.container(border=True):
        st.markdown("#### Insights Summary")

        bullets = []

        # 1) Education -> Salary
        if col_salary and col_edu and (col_salary in ib_sal.columns) and (col_edu in ib_sal.columns):
            edu_tmp = ib_sal[[col_edu, col_salary]].dropna()
            edu_tmp[col_salary] = _to_num(edu_tmp[col_salary])
            if not edu_tmp.empty:
                means = edu_tmp.groupby(col_edu)[col_salary].mean().sort_values(ascending=False)
                if not means.empty:
                    bullets.append(f"- 🎓 **Education → Salary:** Highest average salary for **{means.index[0]}** (~{means.iloc[0]:,.0f}).")

        # 2) Which age groups earn the most?
        if col_salary and (col_salary in ib_sal.columns) and ("__Age_Group" in ib_sal.columns):
            age_tmp = ib_sal[["__Age_Group", col_salary]].dropna()
            age_tmp[col_salary] = _to_num(age_tmp[col_salary])
            if not age_tmp.empty:
                age_means = age_tmp.groupby("__Age_Group")[col_salary].mean().sort_values(ascending=False)
                if not age_means.empty:
                    bullets.append(f"- 👥 **Top-earning age group:** **{age_means.index[0]}** (avg ~{age_means.iloc[0]:,.0f}).")

        # 3) Salary trend with experience
        if col_salary and (col_salary in ib_sal.columns) and years_col and (years_col in ib_sal.columns):
            trend_tmp = pd.DataFrame({"Salary": _to_num(ib_sal[col_salary]),
                                      "Years": _to_num(ib_sal[years_col])}).dropna()
            if len(trend_tmp) >= 2:
                try:
                    slope = np.polyfit(trend_tmp["Years"], trend_tmp["Salary"], 1)[0]
                    if np.isfinite(slope):
                        direction = "rises" if slope > 0 else "falls"
                        bullets.append(f"- 📈 **Experience trend:** Salary **{direction}** with experience (slope ≈ {slope:,.0f} per year).")
                except Exception:
                    pass

        if bullets:
            st.markdown("\n".join(bullets))
        else:
            st.markdown("- No clear patterns under the current filters. Try widening the selection.")


# =============================================================
# PAGE 4: INCOME & INVESTMENT RELATIONSHIP
# =============================================================
elif page == "Income & Investment Relationship":
    #st.subheader("Relationship Between Income & Investment")

    # ---------------------------
    # Helpers (kept local to this page)
    # ---------------------------
    def _to_num(s: pd.Series) -> pd.Series:
        """Safe numeric conversion that tolerates strings; returns float Series."""
        return pd.to_numeric(s, errors="coerce")

    # Parse messy expected-return strings like "10%", "8–12%", "~12", "10 to 12"
    import math
    _num_pat = re.compile(r"(-?\d+(?:\.\d+)?)")
    def parse_expected_return(series: pd.Series) -> pd.Series:
        def _first_num(x):
            if pd.isna(x): return np.nan
            m = _num_pat.findall(str(x))
            if not m: return np.nan
            v = float(m[0])
            # sanity clamp (percent range)
            if v < 0 or v > 200: return np.nan
            return v
        return series.apply(_first_num).astype(float)

    # Derive numeric "Years of Experience" from explicit years or from band midpoints
    def _mid_from_band(txt):
        if pd.isna(txt): return np.nan
        t = str(txt).lower().strip()
        m = re.match(r"(\d+)\s*\+\s*", t)           # e.g., "20+"
        if m: return float(m.group(1)) + 2
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)", t)    # e.g., "3-5"
        if m: 
            a, b = float(m.group(1)), float(m.group(2))
            return (a + b) / 2.0
        if "less" in t and "1" in t:                # "less than 1 year"
            return 0.5
        m = re.search(r"(\d+(?:\.\d+)?)", t)        # fallback: first number
        return float(m.group(1)) if m else np.nan

    # Find "rank-like" investment columns: "Preference rank for ... investment"
    rank_like_cols = [
        c for c in finance.columns
        if re.search(r"^preference\s*rank\s*for\s+.+\s+investment$", c, flags=re.I)
    ]
    def _pretty_inv(colname: str) -> str:
        m = re.search(r"preference\s*rank\s*for\s+(.+)\s+investment", colname, flags=re.I)
        return (m.group(1).title().strip() if m else colname.title())

    def derive_top_choice(fin_df: pd.DataFrame) -> pd.Series:
        """
        Determine each respondent's most-preferred investment.
        - If rank-like columns exist: pick column with MIN rank (1 = most preferred).
        - Else fall back to the single "investment type" column if present.
        Returns a Series aligned to fin_df.index with strings (or NaN).
        """
        if rank_like_cols:
            ranks = fin_df[rank_like_cols].apply(pd.to_numeric, errors="coerce")
            idx = ranks.idxmin(axis=1)
            return idx.apply(lambda c: _pretty_inv(c) if pd.notna(c) else np.nan)
        elif col_invtype and (col_invtype in fin_df.columns):
            return fin_df[col_invtype]
        else:
            return pd.Series(np.nan, index=fin_df.index)

    # ---------------------------
    # 1) Merge salary + finance views (using your globally-filtered f_fin/f_sal)
    # ---------------------------
    merge_cols_fin = {}
    if col_gender and (col_gender in f_fin.columns):  merge_cols_fin["Gender"] = f_fin[col_gender]
    if col_agegrp and (col_agegrp in f_fin.columns):  merge_cols_fin["AgeGroup"] = f_fin[col_agegrp]
    if col_age and (col_age in f_fin.columns):        merge_cols_fin["AgeNum"] = _to_num(f_fin[col_age])
    if col_return and (col_return in f_fin.columns):  merge_cols_fin["ExpectedReturnRaw"] = f_fin[col_return]

    merge_cols_sal = {}
    if col_salary and (col_salary in f_sal.columns):  merge_cols_sal["Salary"] = _to_num(f_sal[col_salary])
    if col_edu and (col_edu in f_sal.columns):        merge_cols_sal["Education"] = f_sal[col_edu]
    if col_expyrs and (col_expyrs in f_sal.columns):  merge_cols_sal["ExpYears"] = _to_num(f_sal[col_expyrs])
    if col_expband and (col_expband in f_sal.columns):merge_cols_sal["ExpBand"] = f_sal[col_expband]

    fin_view = pd.DataFrame(merge_cols_fin)
    sal_view = pd.DataFrame(merge_cols_sal)
    merged = fin_view.join(sal_view, how="outer")

    # Clean / engineer fields needed for plots
    merged["ExpectedReturn"] = parse_expected_return(merged["ExpectedReturnRaw"]) if "ExpectedReturnRaw" in merged else np.nan
    # Experience years: prefer numeric, else derive from band
    if "ExpYears" not in merged and "ExpBand" in merged:
        merged["ExpYears"] = merged["ExpBand"].apply(_mid_from_band)
    elif "ExpYears" in merged and merged["ExpYears"].isna().all() and "ExpBand" in merged:
        merged["ExpYears"] = merged["ExpBand"].apply(_mid_from_band)

    # Most preferred investment per person
    merged["TopInvestment"] = derive_top_choice(f_fin)

    # ---------------------------
    # 2) Page-level filters (collapsed & empty by default)
    # ---------------------------
    with st.expander("Refine Page Filters", expanded=False):
        _g_opts   = sorted(merged["Gender"].dropna().unique())   if "Gender"   in merged else []
        _a_opts   = sorted(merged["AgeGroup"].dropna().unique()) if "AgeGroup" in merged else []
        _e_opts   = sorted(merged["Education"].dropna().unique())if "Education" in merged else []

        c1, c2, c3 = st.columns(3)
        with c1:
            _flt_gender  = st.multiselect("Gender", _g_opts,   default=[], key="p4_gender")  if _g_opts else []
        with c2:
            _flt_agegrp  = st.multiselect("Age Group", _a_opts, default=[], key="p4_agegrp")  if _a_opts else []
        with c3:
            _flt_edu     = st.multiselect("Education Level", _e_opts, default=[], key="p4_edu") if _e_opts else []

    # Apply page filters
    rel_df = merged.copy()
    if "Gender"   in rel_df.columns and _flt_gender: rel_df = rel_df[rel_df["Gender"].isin(_flt_gender)]
    if "AgeGroup" in rel_df.columns and _flt_agegrp: rel_df = rel_df[rel_df["AgeGroup"].isin(_flt_agegrp)]
    if "Education"in rel_df.columns and _flt_edu:    rel_df = rel_df[rel_df["Education"].isin(_flt_edu)]

    # ---------------------------
    # Chart A: Scatter — Salary vs Expected Return (OLS trendline)
    # ---------------------------
    st.markdown("### Salary vs Expected Return (%)")
    if {"Salary","ExpectedReturn"}.issubset(rel_df.columns):
        a = rel_df[["Salary","ExpectedReturn","Education"]].dropna(subset=["Salary","ExpectedReturn"])
        a = a[np.isfinite(a["Salary"]) & np.isfinite(a["ExpectedReturn"])]
        if not a.empty:
            kwargs = dict(x="Salary", y="ExpectedReturn", trendline="ols",
                          title="Salary vs Expected Return (%)")
            if "Education" in a.columns: kwargs["color"] = "Education"
            figA = px.scatter(a, **kwargs)
            figA.update_layout(xaxis_title="Salary", yaxis_title="Expected Return (%)")
            st.plotly_chart(figA, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** How **income** relates to **return expectations**.\n"
                "- **How to read:** An **upward** trend suggests higher earners expect **higher returns** (more risk), "
                "while a **downward** trend suggests they prefer **safer** expectations."
            )
        else:
            st.info("Not enough rows with both Salary and Expected Return after filtering.")
    else:
        st.info("Required fields (Salary, Expected Return) are not available for the scatter.")

    st.divider()

    # ---------------------------
    # Chart B: Grouped Bar — Education vs Most-Preferred Investment
    # ---------------------------
    st.markdown("### Education Level vs Most-Preferred Investment Type")
    if {"Education","TopInvestment"}.issubset(rel_df.columns):
        b = rel_df[["Education","TopInvestment"]].dropna()
        if not b.empty:
            ct = b.value_counts(["Education","TopInvestment"]).reset_index(name="Count")
            figB = px.bar(ct, x="Education", y="Count", color="TopInvestment",
                          barmode="group", title="Most-Preferred Investment by Education Level")
            figB.update_layout(xaxis_title="Education Level", yaxis_title="Respondents", legend_title="Investment Type")
            st.plotly_chart(figB, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** For each **education level**, which **investment type** is chosen most often.\n"
                "- **How to read:** Compare bar heights across colors within an education group to see **preference shifts**."
            )
        else:
            st.info("No data available to compute preferred investment by education under current filters.")
    else:
        st.info("Education or investment-preference fields are missing.")

    st.divider()

    # ---------------------------
    # Chart C: Bubble — Experience (x) vs Salary (y) vs Expected Return (size/color)
    # ---------------------------
    st.markdown("### Experience vs Salary, sized & colored by Expected Return")
    needed = {"ExpYears","Salary","ExpectedReturn"}.issubset(rel_df.columns)
    if needed:
        c = rel_df[["ExpYears","Salary","ExpectedReturn","Education"]].dropna(subset=["ExpYears","Salary","ExpectedReturn"])
        c = c[np.isfinite(c["ExpYears"]) & np.isfinite(c["Salary"]) & np.isfinite(c["ExpectedReturn"])]
        if not c.empty:
            # Size must be positive/finites → clip tiny values
            c["RetSize"] = np.clip(c["ExpectedReturn"].astype(float), 1e-3, None)
            figC = px.scatter(
                c, x="ExpYears", y="Salary",
                size="RetSize", size_max=32,
                color="ExpectedReturn", color_continuous_scale="Viridis",
                hover_data=["Education"],
                title="Experience vs Salary (bubble size & color = Expected Return %)"
            )
            figC.update_layout(xaxis_title="Years of Experience", yaxis_title="Salary", coloraxis_colorbar_title="Exp. Return %")
            st.plotly_chart(figC, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** **Experience** on X, **Salary** on Y, and **Expected Return** encoded by **bubble size & color**.\n"
                "- **How to read:** Larger/brighter bubbles indicate **higher return expectations** at a given salary/experience level."
            )
        else:
            st.info("Not enough valid rows for the bubble chart after filtering.")
    else:
        st.info("Need ExpYears, Salary, and ExpectedReturn to render the bubble chart.")

    st.divider()

    # ---------------------------
    # Chart D: Correlation Heatmap — combined numeric features
    # ---------------------------
    st.markdown("### Correlation (Combined Finance + Salary Features)")
    num_cols = {}
    if "Salary"          in rel_df.columns:     num_cols["Salary"] = rel_df["Salary"]
    if "ExpectedReturn"  in rel_df.columns:     num_cols["ExpectedReturnPct"] = rel_df["ExpectedReturn"]
    if "ExpYears"        in rel_df.columns:     num_cols["ExperienceYears"] = rel_df["ExpYears"]
    if "AgeNum"          in rel_df.columns:     num_cols["Age"] = rel_df["AgeNum"]
    heat = pd.DataFrame(num_cols).dropna(how="all")
    if not heat.empty:
        heat = heat.loc[:, heat.std(numeric_only=True) > 0]
        if heat.shape[1] >= 2:
            corr = heat.corr(numeric_only=True)
            figD = px.imshow(corr, text_auto=".2f", aspect="auto",
                             title="Correlation Heatmap: Salary, Expected Return, Experience, Age")
            figD.update_xaxes(tickangle=45)
            st.plotly_chart(figD, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** **Pairwise correlations** across salary, expected return, experience, and age.\n"
                "- **How to read:** Values near **1/-1** imply **strong** positive/negative relationships."
            )
        else:
            st.info("Not enough variance across numeric features to draw a heatmap.")
    else:
        st.info("No numeric data available for the combined correlation heatmap.")

    st.divider()

    # ---------------------------
    # Insights — Answer the three business questions
    # ---------------------------
    with st.container(border=True):
        st.markdown("#### Insights Summary")

        insights = []

        # 1) Do higher earners take more risks (higher expected returns)?
        if {"Salary","ExpectedReturn"}.issubset(rel_df.columns):
            s1 = rel_df[["Salary","ExpectedReturn"]].dropna()
            if len(s1) >= 5:
                corr_se = np.corrcoef(s1["Salary"], s1["ExpectedReturn"])[0,1]
                if np.isfinite(corr_se):
                    if corr_se > 0.1:
                        insights.append(f"- **Earnings vs Risk:** Higher salaries correlate with **higher** expected returns (r ≈ {corr_se:.2f}).")
                    elif corr_se < -0.1:
                        insights.append(f"- **Earnings vs Risk:** Higher salaries correlate with **lower** expected returns (r ≈ {corr_se:.2f}).")
                    else:
                        insights.append(f"- **Earnings vs Risk:** Little to no linear relationship (r ≈ {corr_se:.2f}).")

        # 2) Are experienced pros more likely to invest in mutual funds?
        if {"ExpYears","TopInvestment"}.issubset(rel_df.columns):
            df2 = rel_df[["ExpYears","TopInvestment"]].dropna()
            if not df2.empty:
                hi = df2["ExpYears"].median()
                grp = df2.assign(Bucket=np.where(df2["ExpYears"]>=hi,"Experienced (≥ median)","Less Experienced"))
                share = (grp.groupby("Bucket")["TopInvestment"]
                            .apply(lambda s: (s.str.contains("mutual", case=False, na=False)).mean()*100)
                            .sort_index())
                if len(share) == 2:
                    insights.append(f"- **Experience → Mutual Funds:** {share.index[0]}: {share.iloc[0]:.1f}% vs {share.index[1]}: {share.iloc[1]:.1f}% choosing Mutual Funds.")
        
        # 3) Does education affect investment diversity?
        if {"Education","TopInvestment"}.issubset(rel_df.columns):
            df3 = rel_df[["Education","TopInvestment"]].dropna()
            if not df3.empty:
                # diversity proxy: number of unique top choices per education, normalized
                diversity = (df3.groupby("Education")["TopInvestment"]
                                .nunique()
                                .sort_values(ascending=False))
                if not diversity.empty:
                    top_ed = diversity.index[0]
                    insights.append(f"- **Education & Diversity:** **{top_ed}** shows the **widest mix** of preferred investments "
                                    f"({diversity.iloc[0]} distinct types).")

        if insights:
            st.markdown("\n".join(insights))
        else:
            st.markdown("- No clear patterns under current filters. Try broadening the selection.")

# =============================================================
# PAGE 5: DEMOGRAPHICS & BEHAVIORAL PATTERNS
# Purpose: Examine user characteristics and behavioral choices.
# =============================================================
elif page == "Demographics & Behavioral Patterns":
    #st.subheader("Demographics & Behavioral Patterns")

    # ---------------------------
    # Helpers
    # ---------------------------
    def _norm_list_cell(x):
        """Split a free-text 'goals' cell into a list by common delimiters."""
        if pd.isna(x):
            return []
        parts = re.split(r"[;,|/]+", str(x))
        return [p.strip() for p in parts if p and p.strip().lower() not in {"nan", "none"}]

    # Detect a likely "goals" column in the finance dataset
    goal_col = None
    goal_candidates = [c for c in f_fin.columns if re.search(r"(goal|objective|purpose|saving)", c, flags=re.I)]
    if goal_candidates:
        goal_candidates.sort(key=lambda c: (0 if re.search("goal|objective", c, flags=re.I) else 1, len(c)))
        goal_col = goal_candidates[0]

    # ---------------------------
    # Combine demographic columns from both datasets
    # ---------------------------
    combo = pd.DataFrame(index=f_fin.index)
    if col_gender and col_gender in f_fin.columns:
        combo["Gender"] = f_fin[col_gender]
    if col_agegrp and col_agegrp in f_fin.columns:
        combo["AgeGroup"] = f_fin[col_agegrp]
    if col_expband and col_expband in f_sal.columns:
        combo["ExpBand"] = f_sal[col_expband]
    if col_edu and col_edu in f_sal.columns:
        combo["Education"] = f_sal[col_edu]

    # Attach goals if detected
    if goal_col and goal_col in f_fin.columns:
        combo["GoalsRaw"] = f_fin[goal_col]
        combo["GoalsList"] = combo["GoalsRaw"].apply(_norm_list_cell)

    # ---------------------------
    # Page-level filters (collapsed, empty defaults)
    # ---------------------------
    with st.expander("Refine Page Filters", expanded=False):
        g_opts = sorted(combo["Gender"].dropna().unique()) if "Gender" in combo else []
        a_opts = sorted(combo["AgeGroup"].dropna().unique()) if "AgeGroup" in combo else []
        e_opts = sorted(combo["ExpBand"].dropna().unique()) if "ExpBand" in combo else []

        c1, c2, c3 = st.columns(3)
        with c1:
            _p5_gender = st.multiselect("Gender", g_opts, default=[], key="p5_gender") if g_opts else []
        with c2:
            _p5_agegrp = st.multiselect("Age Group", a_opts, default=[], key="p5_agegrp") if a_opts else []
        with c3:
            _p5_expband = st.multiselect("Experience Band", e_opts, default=[], key="p5_expband") if e_opts else []

    # Apply filters
    dem = combo.copy()
    if "Gender" in dem.columns and _p5_gender:
        dem = dem[dem["Gender"].isin(_p5_gender)]
    if "AgeGroup" in dem.columns and _p5_agegrp:
        dem = dem[dem["AgeGroup"].isin(_p5_agegrp)]
    if "ExpBand" in dem.columns and _p5_expband:
        dem = dem[dem["ExpBand"].isin(_p5_expband)]

    # ---------------------------
    # Charts 1 & 2 side-by-side: Gender Ratio (Pie) + Age Group Distribution (Bar)
    # ---------------------------
    left, right = st.columns(2)

    with left:
        st.markdown("### Gender Ratio")
        if "Gender" in dem.columns:
            g = dem["Gender"].dropna()
            if not g.empty:
                fig1 = px.pie(g, names="Gender", title="Gender Ratio")
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown(
                    "- **What you’re seeing:** The **share of respondents by gender** in the current filtered view.\n"
                    "- **How to read:** The **largest slice** indicates the **dominant gender**. "
                    "If filters are applied, this reflects the filtered subset rather than the whole dataset.\n"
                    "- **Why it matters:** Gender mix provides context for later charts (e.g., if one gender dominates, "
                    "their preferences may drive overall patterns)."
                )
            else:
                st.info("No gender data available under the current filters.")
        else:
            st.info("Gender column not found.")

    with right:
        st.markdown("### Age Group Distribution")
        if "AgeGroup" in dem.columns:
            a = dem["AgeGroup"].dropna()
            if not a.empty:
                # preserve a natural order if possible
                order = ["18-24", "25-34", "35-44", "45-54", "55+"]
                counts = a.value_counts().reset_index()
                counts.columns = ["AgeGroup", "Count"]
                if all(x in counts["AgeGroup"].values for x in order):
                    counts["AgeGroup"] = pd.Categorical(counts["AgeGroup"], categories=order, ordered=True)
                    counts = counts.sort_values("AgeGroup")
                fig2 = px.bar(counts, x="AgeGroup", y="Count", title="Age Group Distribution")
                fig2.update_layout(xaxis_title="Age Group", yaxis_title="Respondents")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown(
                    "- **What you’re seeing:** **Headcount by age band** in the filtered data.\n"
                    "- **How to read:** **Taller bars** signal **larger cohorts**; a skew to a single band "
                    "means insights may be shaped by that group’s behavior.\n"
                    "- **Why it matters:** Age often correlates with **risk tolerance**, **tenure preference**, and **monitoring frequency**."
                )
            else:
                st.info("No age group data under the current filters.")
        else:
            st.info("AgeGroup column not found.")

    st.divider()

    # ---------------------------
    # Chart 3: Stacked Bar — Education level vs Gender
    # ---------------------------
    st.markdown("### Education Level by Gender (Stacked)")
    if {"Education", "Gender"}.issubset(dem.columns):
        e = dem[["Education", "Gender"]].dropna()
        if not e.empty:
            ct = e.value_counts(["Education", "Gender"]).reset_index(name="Count")
            fig3 = px.bar(
                ct, x="Education", y="Count", color="Gender",
                barmode="stack", title="Education Level vs Gender"
            )
            fig3.update_layout(xaxis_title="Education Level", yaxis_title="Respondents")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** The **education mix** broken down by **gender**.\n"
                "- **How to read:** Within each education level, the **stacked colors** show the **proportion** of genders. "
                "Strong imbalances may indicate **pipeline effects** (e.g., some education tracks attracting more of one gender).\n"
                "- **Why it matters:** Education interacts with income and investment complexity; "
                "gender composition by education can hint at **preference differences** downstream."
            )
        else:
            st.info("No Education vs Gender data after filtering.")
    else:
        st.info("Education or Gender column missing.")

    st.divider()

    # ---------------------------
    # Chart 4: Treemap — Age group → Education → Investment Goal
    # ---------------------------
    st.markdown("### Treemap: Age Group → Education → Investment Goal")
    if {"AgeGroup", "Education"}.issubset(dem.columns) and ("GoalsList" in dem.columns):
        t = dem[["AgeGroup", "Education", "GoalsList"]].dropna(subset=["AgeGroup", "Education"])
        t = t[t["GoalsList"].apply(lambda lst: isinstance(lst, list) and len(lst) > 0)]
        if not t.empty:
            t = t.explode("GoalsList").rename(columns={"GoalsList": "Goal"})
            t = t.dropna(subset=["Goal"])
            t["Goal"] = t["Goal"].astype(str).str.strip().str.title()
            t["Count"] = 1
            fig4 = px.treemap(
                t, path=["AgeGroup", "Education", "Goal"], values="Count",
                title="Treemap — Age Group → Education → Investment Goal"
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(
                "- **What you’re seeing:** A **hierarchical map** of respondents by **Age Group**, then **Education**, then **Investment Goal**.\n"
                "- **How to read:** The **area** of each rectangle encodes **how many respondents** fall into that branch "
                "(e.g., 25–34 → Bachelor's → Retirement). Hover to see exact counts.\n"
                "- **Why it matters:** Treemaps quickly reveal **where concentration lies**—which demographics gravitate to "
                "specific goals like **Retirement**, **Healthcare**, or **Wealth Building**."
            )
        else:
            st.info("No goal data available under the current filters.")
    else:
        st.info("AgeGroup, Education, or Goal data missing for treemap.")

    st.divider()

    # ---------------------------
    # Insights Summary
    # ---------------------------
    with st.container(border=True):
        st.markdown("#### Insights Summary")

        insights = []

        # Dominant groups (gender / education / age)
        if "Gender" in dem.columns and dem["Gender"].notna().any():
            g_share = dem["Gender"].value_counts(normalize=True) * 100
            insights.append(f"- **Dominant gender:** {g_share.index[0]} (~{g_share.iloc[0]:.1f}%).")

        if "Education" in dem.columns and dem["Education"].notna().any():
            e_share = dem["Education"].value_counts(normalize=True) * 100
            insights.append(f"- **Most common education:** {e_share.index[0]} (~{e_share.iloc[0]:.1f}%).")

        if "AgeGroup" in dem.columns and dem["AgeGroup"].notna().any():
            a_share = dem["AgeGroup"].value_counts(normalize=True) * 100
            insights.append(f"- **Largest age cohort:** {a_share.index[0]} (~{a_share.iloc[0]:.1f}%).")

        # How demographics shape investment decisions (use goals if present)
        if {"AgeGroup", "Education", "GoalsList"}.issubset(dem.columns):
            t2 = dem[["AgeGroup", "Education", "GoalsList"]].dropna()
            t2 = t2[t2["GoalsList"].apply(lambda lst: isinstance(lst, list) and len(lst) > 0)]
            if not t2.empty:
                t2 = t2.explode("GoalsList").rename(columns={"GoalsList": "Goal"})
                t2["Goal"] = t2["Goal"].astype(str).str.strip().str.title()
                # (a) Top goal by age group
                top_goal_by_age = (
                    t2.groupby("AgeGroup")["Goal"]
                      .agg(lambda s: s.value_counts().index[0] if not s.empty else np.nan)
                )
                if top_goal_by_age.notna().any():
                    items = "; ".join([f"{k}: {v}" for k, v in top_goal_by_age.dropna().items()])
                    insights.append(f"- **Top goal by age group:** {items}.")
                # (b) Optional: breadth of goals (diversity) by education
                breadth = (t2.groupby("Education")["Goal"].nunique().sort_values(ascending=False))
                if not breadth.empty:
                    insights.append(f"- **Goal diversity by education:** widest mix for **{breadth.index[0]}** "
                                    f"({breadth.iloc[0]} distinct goals).")

        if insights:
            st.markdown("\n".join(insights))
        else:
            st.markdown("- No clear demographic patterns detected under current filters.")


# =============================================================
# PAGE 6: INTERACTIVE DATA EXPLORER
# =============================================================
elif page == "Interactive Data Explorer":
    #st.subheader("🧮 Interactive Data Explorer")

    # ---------------------------
    # Helper Functions
    # ---------------------------
    def _to_num(s: pd.Series) -> pd.Series:
        """Convert safely to numeric (NaN for invalid)."""
        return pd.to_numeric(s, errors="coerce")

    def _split_goals(val):
        """Split goal-like free text by comma/semicolon/pipe/slash."""
        if pd.isna(val):
            return []
        parts = re.split(r"[;,|/]+", str(val))
        return [p.strip() for p in parts if p and p.strip().lower() not in {"nan", "none"}]

    def _detect_goal_col(df: pd.DataFrame):
        """Try to detect a goal/objective/saving column in a dataframe."""
        cand = [c for c in df.columns if re.search(r"(goal|objective|purpose|saving)", c, flags=re.I)]
        if not cand:
            return None
        cand.sort(key=lambda c: (0 if re.search("goal|objective", c, flags=re.I) else 1, len(c)))
        return cand[0]

    def _missing_counts(df: pd.DataFrame) -> pd.DataFrame:
        """Return summary of missing values per column, including 'Unknown'/'Nan' strings."""
        miss_strings = {"unknown", "nan"}
        miss_mask = df.isna().copy()
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].astype(str).str.strip().str.lower()
                miss_mask[c] = miss_mask[c] | s.isin(miss_strings)
        counts = miss_mask.sum().rename("MissingCount").to_frame()
        counts["TotalRows"] = len(df)
        counts["MissingPct"] = (counts["MissingCount"] / counts["TotalRows"]) * 100
        return counts.sort_values("MissingPct", ascending=False)

    # ---------------------------
    # Tabs for Finance and Salary datasets
    # ---------------------------
    
    tab_fin, tab_sal = st.tabs(["Finance", "Salary"])

    # ---------------------------
    # PAGE-LEVEL FILTERS (SIDEBAR)
    # ---------------------------
    with st.sidebar:
        st.markdown("### Interactive Data Explorer Filters")

        _fin_gen_opts = sorted(f_fin[col_gender].dropna().unique()) if col_gender and col_gender in f_fin.columns else []
        _fin_age_opts = sorted(f_fin[col_agegrp].dropna().unique()) if col_agegrp and col_agegrp in f_fin.columns else []
        invtype_col = col_invtype if (col_invtype and col_invtype in f_fin.columns) else None

        goal_col_fin = _detect_goal_col(f_fin)
        _fin_goal_opts = []
        if goal_col_fin:
            all_goals = f_fin[goal_col_fin].dropna().apply(_split_goals)
            if not all_goals.empty:
                _fin_goal_opts = sorted({g for lst in all_goals for g in lst})

        _sal_edu_opts = sorted(f_sal[col_edu].dropna().unique()) if col_edu and col_edu in f_sal.columns else []
        _sal_exp_opts = sorted(f_sal[col_expband].dropna().unique()) if col_expband and col_expband in f_sal.columns else []

        st.caption("Selections apply dynamically to the charts below.")
        c1, c2 = st.columns(2)
        with c1:
            p6_gender = st.multiselect("Gender", _fin_gen_opts, default=[], key="p6_gen")
            p6_age    = st.multiselect("Age Group", _fin_age_opts, default=[], key="p6_age")
            p6_edu    = st.multiselect("Education", _sal_edu_opts, default=[], key="p6_edu")
        with c2:
            p6_exp    = st.multiselect("Experience Band", _sal_exp_opts, default=[], key="p6_exp")
            p6_inv    = st.multiselect("Investment Type", _fin_goal_opts, default=[], key="p6_inv")
            p6_goal   = st.multiselect("Investment Goal", _fin_goal_opts, default=[], key="p6_goal")

    # ---------------------------
    # Apply Filters
    # ---------------------------
    def apply_page_filters(fin_df: pd.DataFrame, sal_df: pd.DataFrame):
        fin2, sal2 = fin_df.copy(), sal_df.copy()
        if col_gender and p6_gender:
            fin2 = fin2[fin2[col_gender].isin(p6_gender)]
        if col_agegrp and p6_age:
            fin2 = fin2[fin2[col_agegrp].isin(p6_age)]
        if col_edu and p6_edu:
            sal2 = sal2[sal2[col_edu].isin(p6_edu)]
        if col_expband and p6_exp:
            sal2 = sal2[sal2[col_expband].isin(p6_exp)]
        return fin2, sal2

    fin_view, sal_view = apply_page_filters(f_fin, f_sal)

    # ---------------------------
    # FINANCE TAB
    # ---------------------------
    with tab_fin:
        st.markdown("### Finance Dataset")

        #grid1a, grid1b, grid1c = st.columns([1, 1, 1])
        grid1a, grid1c = st.columns([1, 1])

        # ---------------------------
        # 1️⃣ Investment Type Distribution (Pie)
        # ---------------------------
        with grid1a:
            inv_cols = [c for c in fin_view.columns if re.search(r"invest", c, flags=re.I)]
            inv_col = inv_cols[0] if inv_cols else None

            if inv_col and not fin_view.empty:
                fig_inv = px.pie(fin_view, names=inv_col, title="Investment Type Distribution",
                                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_inv.update_layout(title_font_size=16)
                st.plotly_chart(fig_inv, use_container_width=True)
                st.caption("""
                **What it shows:**  
                - Breakdown of how respondents allocate their investments across different types (Stocks, Mutual Funds, etc.).  
                - Helps identify diversification and risk preferences.
                """)
            else:
                st.warning("⚠️ Investment type column not found in dataset.")


        # ---------------------------
        # 3️⃣ Monitoring Frequency (Bar)
        # ---------------------------
        with grid1c:
            mon_cols = [c for c in fin_view.columns if re.search(r"monitor|frequency", c, flags=re.I)]
            mon_col = mon_cols[0] if mon_cols else None

            if mon_col:
                mf = fin_view[mon_col].dropna()
                if not mf.empty:
                    ct = mf.value_counts().reset_index()
                    ct.columns = ["Frequency", "Count"]
                    fig_mon = px.bar(ct, x="Frequency", y="Count",
                                     title="Portfolio Monitoring Frequency",
                                     color="Frequency", color_discrete_sequence=px.colors.sequential.Blues)
                    fig_mon.update_layout(xaxis_title="Frequency", yaxis_title="Respondents")
                    st.plotly_chart(fig_mon, use_container_width=True)
                    st.caption("""
                    **What it shows:**  
                    - Frequency of portfolio monitoring among investors.  
                    - Indicates how financially engaged respondents are.  
                    - Frequent monitoring may reflect higher involvement or anxiety around market changes.
                    """)

        st.divider()

        # ---------------------------
        # Missing Values Section
        # ---------------------------
        st.markdown("### Missing Values — Finance")
        if not fin_view.empty:
            miss_fin = _missing_counts(fin_view)
            st.markdown("**Summary Table:** Missing counts and percentages (includes 'Unknown').")
            st.dataframe(miss_fin)

            top_fin = miss_fin.head(20).reset_index().rename(columns={"index": "Column"})
            fig_mbar = px.bar(top_fin, x="Column", y="MissingPct", title="Top 20 Missing Columns (%)",
                              color="MissingPct", color_continuous_scale="Teal")
            fig_mbar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_mbar, use_container_width=True)

            # Interactive Heatmap
            show_cols = fin_view.columns[:30]
            heat_mask = fin_view[show_cols].isna().copy()
            for c in show_cols:
                if fin_view[c].dtype == object:
                    s = fin_view[c].astype(str).str.strip().str.lower()
                    heat_mask[c] = heat_mask[c] | s.isin({"unknown", "nan"})
            hm = heat_mask.head(200).astype(int)
            if hm.shape[1] > 1:
                fig_hm = px.imshow(
                    hm.T,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    title="Missingness Heatmap (1=Missing, 0=Present)",
                    labels=dict(x="Respondents", y="Columns", color="Missingness")
                )
                fig_hm.update_traces(hovertemplate="Column: %{y}<br>Respondent Index: %{x}<br>Missing: %{z}<extra></extra>")
                st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("""
            **How to read:**  
            - Each cell shows if a respondent’s value is missing (1) or present (0).  
            - Taller regions of blue suggest specific columns have more data quality issues.
            """)

        st.divider()

        show_fin = st.toggle("Show Raw Finance Data", value=False, key="p6_show_fin")
        if show_fin:
            st.dataframe(fin_view, use_container_width=True)
        st.download_button(
            "Download Filtered Finance Data",
            fin_view.to_csv(index=False).encode("utf-8"),
            "finance_filtered.csv",
            "text/csv",
            key="p6_dl_fin"
        )

    # ---------------------------
    # SALARY TAB
    # ---------------------------
    with tab_sal:
        st.markdown("#### Salary Dataset")

        grid2a, grid2b, grid2c = st.columns([1, 1, 1])

        # 1) Salary Distribution by Experience Band (Histogram)
        with grid2a:
            if col_salary and col_expband and (col_salary in sal_view.columns) and (col_expband in sal_view.columns):
                tmp = sal_view[[col_salary, col_expband]].dropna()
                if not tmp.empty:
                    fig_sd = px.histogram(tmp, x=col_salary, color=col_expband, nbins=30,
                                          title="Salary Distribution by Experience Band")
                    fig_sd.update_layout(xaxis_title="Salary", yaxis_title="Respondents", legend_title="Experience Band")
                    st.plotly_chart(fig_sd, use_container_width=True)
                    st.caption("**What it shows:** How salaries spread across experience levels in the filtered cohort.")

        # 2) Average Salary by Education Level (Bar)
        with grid2b:
            if col_salary and col_edu and (col_salary in sal_view.columns) and (col_edu in sal_view.columns):
                tmp = sal_view[[col_edu, col_salary]].dropna()
                if not tmp.empty:
                    tmp[col_salary] = _to_num(tmp[col_salary])
                    avg_sal = tmp.groupby(col_edu)[col_salary].mean().reset_index()
                    fig_avg = px.bar(avg_sal, x=col_edu, y=col_salary, title="Average Salary by Education Level",
                                     color=col_salary, color_continuous_scale="Blues")
                    fig_avg.update_layout(xaxis_title="Education Level", yaxis_title="Average Salary")
                    st.plotly_chart(fig_avg, use_container_width=True)
                    st.caption("**What it shows:** Typical salary by education level for the current selection.")

        # 3) Salary vs Years of Experience (Scatter with trendline)
        with grid2c:
            # Prefer numeric Years of Experience; otherwise try to derive midpoint from Experience Band
            years_col = None
            if col_expyrs and (col_expyrs in sal_view.columns):
                years_col = col_expyrs
            elif col_expband and (col_expband in sal_view.columns):
                # quick midpoint converter
                def _mid_from_band(txt):
                    if pd.isna(txt): return np.nan
                    t = str(txt).lower().strip()
                    m = re.match(r"(\d+)\s*\+\s*", t)
                    if m: return float(m.group(1)) + 2
                    m = re.match(r"(\d+)\s*[-–]\s*(\d+)", t)
                    if m: 
                        a, b = float(m.group(1)), float(m.group(2))
                        return (a + b) / 2.0
                    if "less" in t and "1" in t:
                        return 0.5
                    m = re.search(r"(\d+(?:\.\d+)?)", t)
                    return float(m.group(1)) if m else np.nan

                years_col = "_p6_years_mid"
                sal_view[years_col] = sal_view[col_expband].apply(_mid_from_band)

            if years_col and (years_col in sal_view.columns) and (col_salary in sal_view.columns):
                sc = pd.DataFrame({
                    "Salary": _to_num(sal_view[col_salary]),
                    "Years": _to_num(sal_view[years_col])
                }).dropna()
                if not sc.empty:
                    fig_sc = px.scatter(sc, x="Years", y="Salary", trendline="ols",
                                        title="Salary vs Years of Experience (Trendline)")
                    fig_sc.update_layout(xaxis_title="Years of Experience", yaxis_title="Salary")
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.caption("**What it shows:** Relationship between experience and salary (trendline = overall slope).")

        st.divider()

        # ---------- Missing Values (Summary + Bar + Heatmap)
        st.markdown("#### Missing Values — Salary")
        if not sal_view.empty:
            miss_sal = _missing_counts(sal_view)
            st.markdown("**Summary Table** — counts/pct of missing values (including literal 'Unknown'/'Nan').")
            st.dataframe(miss_sal)

            # Bar chart — top 20 missing columns
            top_sal = miss_sal.head(20).reset_index().rename(columns={"index": "Column"})
            fig_mbar2 = px.bar(top_sal, x="Column", y="MissingPct", title="Missing Values (Top 20 columns)",
                               labels={"MissingPct": "% Missing"})
            fig_mbar2.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_mbar2, use_container_width=True)

            # Heatmap — missingness matrix (limit for readability)
            show_cols2 = sal_view.columns[:30]
            heat_mask2 = sal_view[show_cols2].isna().copy()
            for c in show_cols2:
                if sal_view[c].dtype == object:
                    s = sal_view[c].astype(str).str.strip().str.lower()
                    heat_mask2[c] = heat_mask2[c] | s.isin({"unknown", "nan"})
            hm2 = heat_mask2.head(200).astype(int)
            if hm2.shape[1] > 1:
                fig_hm2 = px.imshow(hm2.T, aspect="auto", title="Missingness Heatmap (1=Missing, 0=Present)")
                st.plotly_chart(fig_hm2, use_container_width=True)
            st.caption("**How to read:** Columns with high missingness may need data cleaning or imputation.")

        st.divider()

        # ---------- Show raw data + download
        show_sal = st.toggle("Show raw Salary data", value=False, key="p6_show_sal")
        if show_sal:
            st.dataframe(sal_view, use_container_width=True)
        st.download_button(
            "Download Salary (filtered)", sal_view.to_csv(index=False).encode("utf-8"),
            "salary_filtered.csv", "text/csv", key="p6_dl_sal"
        )


# =============================================================
# PAGE 7: INSIGHTS & STORYTELLING
# =============================================================
elif page == "Insights & Storytelling":
    st.subheader("💡 Key Insights & Storytelling")

    st.markdown("""
    - **Younger investors (<35)** prefer stocks and mutual funds, while **older participants** lean toward gold and fixed deposits.  
    - **Higher education** correlates with **higher salaries and diversified portfolios**.  
    - **Females** check investments less frequently than **males**, showing cautious behavior.  
    - **Experienced professionals (5+ yrs)** prefer long-term, safer investment goals.  
    """)


# =============================================================
# Card Theam Style
# =============================================================

st.markdown("""
<style>
:root{
  --ibox-bg: #E0E0E0;
  --ibox-border: #D9C7FF;
  --ibox-radius: 16px;
  --ibox-pad: 1px;
  --ibox-shadow: 0 10px 24px rgba(92, 52, 179, .22), inset 0 1px 0 rgba(255,255,255,.75);
  --ibox-shadow-hover: 0 16px 36px rgba(92, 52, 179, .35), inset 0 1px 0 rgba(255,255,255,.9);
  --ibox-transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}

div.stMetric, 
div.stPlotlyChart, 
div.stDataFrame, 
div.stTable {
  background: var(--ibox-bg) !important;
  border: 0.5px solid var(--ibox-border) !important;
  border-radius: var(--ibox-radius) !important;
  padding: var(--ibox-pad) !important;
  box-shadow: var(--ibox-shadow) !important;
  transition: var(--ibox-transition) !important;
  position: relative !important;
  overflow: hidden !important;
  margin-bottom: 12px !important;
  will-change: transform, box-shadow, filter;
}

div.stMetric:hover, 
div.stPlotlyChart:hover, 
div.stDataFrame:hover, 
div.stTable:hover {
  transform: translateY(-4px) scale(1.012) !important;
  box-shadow: var(--ibox-shadow-hover) !important;
  filter: saturate(1.02) !important;
}

div.stPlotlyChart > div {
  background: transparent !important;
  border-radius: var(--ibox-radius) !important;
  overflow: hidden !important;
}

div.stMetric > div:first-child {
  margin-top: 2px !important;
}

div.stDataFrame {
  overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# =====================[ CHANGES START — KPI center + dark gray (final) ]=====================
st.markdown("""
<style>
/* 1) Make the metric's inner wrapper a flex column and center it */
div[data-testid="stMetric"] > div {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;      /* horizontal center */
  justify-content: center !important;  /* vertical center */
  text-align: center !important;
  min-height: 100% !important;
}

/* 2) Center and recolor each metric part explicitly */
span[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"] {
  text-align: center !important;
  color: #2B2B2B !important;           /* dark gray */
}

/* 3) Some Streamlit themes wrap value inside another div — target it too */
div[data-testid="stMetricValue"] > div {
  text-align: center !important;
  color: #2B2B2B !important;
  font-weight: 700 !important;         /* keep value readable on light card */
}

/* 4) Inherit color for any descendants (helps against theme overrides) */
div[data-testid="stMetric"] * {
  color: #2B2B2B !important;
  text-align: center !important;
}
</style>
""", unsafe_allow_html=True)
# =====================[ CHANGES END — KPI center + dark gray (final) ]=====================

# --------------------------- Sidebar label colors (add-on) ---------------------------
FILTER_TITLE_COLOR = "#2C3539"   # Filter / section headings
FILTER_LABEL_COLOR = "#2C3539"   # Field labels: Gender, Age Group, etc.
FILTER_CAPTION_COLOR = "#313234" # Small helper text (optional)

st.markdown(f"""
<style>
/* A) Main "Filter" header (you already use .filter-header) */
[data-testid="stSidebar"] .filter-header {{
  color: {FILTER_TITLE_COLOR} !important;
}}

/* B) Any other sidebar headings created via markdown (h2/h3/h4) */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {{
  color: {FILTER_TITLE_COLOR} !important;
}}

/* C) Widget labels (Selectbox / Multiselect / Radio / Slider / Date picker, etc.) */
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stMultiSelect > label,
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] legend p {{
  color: {FILTER_LABEL_COLOR} !important;
  font-weight: 600 !important;
}}

/* D) Optional: small helper/caption text in the sidebar */
[data-testid="stSidebar"] .stCaption, 
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p:has(small),
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] small {{
  color: {FILTER_CAPTION_COLOR} !important;
}}
</style>
""", unsafe_allow_html=True)
# -------------------------------------------------------------------------------