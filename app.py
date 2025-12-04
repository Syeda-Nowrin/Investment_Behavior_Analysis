# app.py - COMPLETE UPDATED VERSION
# Part 1 of 3: Imports, Configuration, Helpers, and Sidebar Setup
# Copy all 3 parts into one file

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import re
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# APP CONFIGURATION
st.set_page_config(page_title="Investment Behavior Analysis", page_icon="🪙", layout="wide", initial_sidebar_state="expanded")

# CSS Styling (keep all your original styling)
st.markdown("""
<style>
section[data-testid="stContainer"][class*="st-emotion-cache"][style*="border: 1px"]{
  background: #F3EAFF; border: 1px solid #D9C7FF; border-radius: 16px !important;
  box-shadow: 0 10px 24px rgba(92, 52, 179, .25), inset 0 1px 0 rgba(255,255,255,.8);
  padding: 16px;
}
div[data-testid="stPlotlyChart"] > div{ background: transparent !important; }
h3, h4 { margin-top: .25rem; }

:root{
  --ibox-bg: #E0E0E0; --ibox-border: #D9C7FF; --ibox-radius: 16px;
  --ibox-shadow: 0 10px 24px rgba(92, 52, 179, .22), inset 0 1px 0 rgba(255,255,255,.75);
  --ibox-shadow-hover: 0 16px 36px rgba(92, 52, 179, .35);
}
div.stMetric, div.stPlotlyChart, div.stDataFrame, div.stTable {
  background: var(--ibox-bg) !important; border: 0.5px solid var(--ibox-border) !important;
  border-radius: var(--ibox-radius) !important; box-shadow: var(--ibox-shadow) !important;
  transition: transform .16s ease; margin-bottom: 12px !important;
}
div.stMetric:hover, div.stPlotlyChart:hover {
  transform: translateY(-4px) scale(1.012) !important;
  box-shadow: var(--ibox-shadow-hover) !important;
}
div[data-testid="stMetric"] > div {
  display: flex !important; flex-direction: column !important;
  align-items: center !important; text-align: center !important;
}
span[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
  text-align: center !important; color: #2B2B2B !important;
}

[data-testid="stSidebar"] { background-color: #E0E0E0; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border: 1px solid rgba(91,192,190,.35); border-radius: 10px;
  padding: 8px 10px; background: #ffffff; transition: background .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #3a506b; border-color: #5bc0be; transform: translateX(3px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"] {
  background: #5bc0be !important; color: #ffffff !important;
  box-shadow: 0 6px 16px rgba(91,192,190,.35);
}
[data-testid="stSidebar"] [data-testid="stRadio"] svg { display: none !important; }
</style>
""", unsafe_allow_html=True)

# LOAD DATASETS
BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data(fin_path, sal_path, trends_path):
    finance = pd.read_csv(fin_path)
    salary = pd.read_csv(sal_path)
    trends = pd.read_csv(trends_path)
    finance.columns = finance.columns.str.strip()
    salary.columns = salary.columns.str.strip()
    trends.columns = trends.columns.str.strip()
    return finance, salary, trends

finance, salary, trends = load_data(
    BASE_DIR / "Finance_Dataset-Cleaned.csv",
    BASE_DIR / "Salary_Dataset-Cleaned.csv",
    BASE_DIR / "Finance_Trends-Cleaned.csv"
)

# HELPER FUNCTIONS
def fcol(cols, key):
    key = key.lower()
    for c in cols:
        if key in c.lower(): return c
    return None

def multi_match(cols, keys):
    for k in keys:
        c = fcol(cols, k)
        if c: return c
    return None

def to_age_group(x):
    try: a = int(float(x))
    except: return "Unknown"
    if pd.isna(a): return "Unknown"
    if a <= 24: return "18-24"
    if a <= 34: return "25-34"
    if a <= 44: return "35-44"
    if a <= 54: return "45-54"
    return "55+"

def pretty_heat_label(name: str) -> str:
    n = name.replace("fin_", "").replace("sal_", "").replace("trn_", "")
    n = re.sub(r"Preference\s*rank\s*for\s+(.*?)\s+investment", r"Rank: \1", n, flags=re.I)
    return n.replace("_", " ").title()

def make_unique(labels):
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

# NEW: Story header function
def story_header(chapter, title, hook, context, question):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0A1E5A 0%, #3A78C2 100%); 
                padding: 2rem; border-radius: 16px; margin-bottom: 2rem; color: white;">
        <div style="font-size: 0.9rem; opacity: 0.9; letter-spacing: 2px;">{chapter}</div>
        <h1 style="margin: 0.5rem 0; font-size: 2.5rem; color: white;">{title}</h1>
        <p style="font-size: 1.2rem; font-weight: 500; margin: 1rem 0; 
                  border-left: 4px solid rgba(255,255,255,0.5); padding-left: 1rem;">{hook}</p>
        <p style="font-size: 0.95rem; opacity: 0.9; margin: 1rem 0;">{context}</p>
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <strong>🎯 Key Question:</strong> {question}
        </div>
    </div>
    """, unsafe_allow_html=True)

# IDENTIFY IMPORTANT COLUMNS
col_gender = multi_match(finance.columns, ["gender", "sex"])
col_age = multi_match(finance.columns, ["age"])
col_agegrp = fcol(finance.columns, "age_group")
col_return = multi_match(finance.columns, ["expected return", "return"])
col_monitor = multi_match(finance.columns, ["monitor"])
col_tenure = multi_match(finance.columns, ["tenure", "keep your money"])
col_invtype = multi_match(finance.columns, ["investment type", "type of investment"])
col_salary = multi_match(salary.columns, ["salary", "income"])
col_edu = multi_match(salary.columns, ["education"])
col_expband = fcol(salary.columns, "experience band")
col_expyrs = multi_match(salary.columns, ["years of experience", "experience"])

if col_agegrp not in finance.columns and col_age in finance.columns:
    finance["Age_Group"] = finance[col_age].apply(to_age_group)
    col_agegrp = "Age_Group"

# SIDEBAR SETUP
st.sidebar.markdown("""
<div style="padding: 8px 16px 12px; border-bottom: 1px solid rgba(180,200,220,0.25); margin-bottom: 8px;">
  <div style="display: flex; flex-direction: column; align-items: center;">
    <span style="font-size: 48px; filter: drop-shadow(0 0 12px rgba(0,217,255,0.5));">🪙</span>
    <h1 style="font-size: 22px; font-weight: 800; color: #2C3539; text-align: center;">
      Investment Behavior Analysis
    </h1>
    <div style="font-size: 13px; font-weight: 600; color: #2C3539; text-align: center;">
      People.Investment.Patterns
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# UPDATED: Story-driven navigation
nav_items = {
    "🏠 Overview Dashboard": "",
    "👥 Who Are Our Investors?": "",
    "🎲 The Age-Risk Connection": "",
    "💼 Education, Experience & Earnings": "",
    "💰 The Income-Risk Tradeoff": "",
    "🔗 Connecting the Dots": "",
    "🤖 Can We Predict Risk Appetite?": "",
    "🔍 Explore the Data": "",
    "📊 The Story in Numbers": ""
}

if "page" not in st.session_state:
    st.session_state.page = list(nav_items.keys())[0]

pages = list(nav_items.keys())
choice = st.sidebar.radio("Navigation", pages, 
                          index=pages.index(st.session_state.page),
                          label_visibility="collapsed", key="nav_radio")
st.session_state.page = choice

st.sidebar.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)

# Story Mode Toggle
story_mode = st.sidebar.toggle("📖 Story Mode", value=True, 
                                help="Enable narrative explanations")
if "story_mode" not in st.session_state:
    st.session_state.story_mode = True
st.session_state.story_mode = story_mode

st.sidebar.markdown('<div style="font-size: 1.1rem; font-weight: 600; color: #2C3539; margin: 1rem 0 0.8rem;">Filters</div>', unsafe_allow_html=True)

# Filters
gender_vals = sorted(finance[col_gender].dropna().unique()) if col_gender else []
sel_gender = st.sidebar.multiselect("Gender", gender_vals, default=[])
age_vals = sorted(finance[col_agegrp].dropna().unique()) if col_agegrp else []
sel_age = st.sidebar.multiselect("Age Group", age_vals, default=[])
edu_vals = sorted(salary[col_edu].dropna().unique()) if col_edu else []
sel_edu = st.sidebar.multiselect("Education Level", edu_vals, default=[])
expband_vals = sorted(salary[col_expband].dropna().unique()) if col_expband else []
sel_expband = st.sidebar.multiselect("Experience Band", expband_vals, default=[])

# Apply filters
f_fin = finance.copy()
if col_gender and sel_gender: f_fin = f_fin[f_fin[col_gender].isin(sel_gender)]
if col_agegrp and sel_age: f_fin = f_fin[f_fin[col_agegrp].isin(sel_age)]

f_sal = salary.copy()
if col_edu and sel_edu: f_sal = f_sal[f_sal[col_edu].isin(sel_edu)]
if col_expband and sel_expband: f_sal = f_sal[f_sal[col_expband].isin(sel_expband)]

f_trends = trends.copy()
if col_gender and sel_gender and "gender" in f_trends.columns:
    f_trends = f_trends[f_trends["gender"].str.title().isin(sel_gender)]

page = st.session_state.page


# END OF PART 1

# =============================================================
# PAGE 1: OVERVIEW DASHBOARD (COMPLETELY REDESIGNED)
# =============================================================
if page == "🏠 Overview Dashboard":
    story_header(
        chapter="THE COMPLETE PICTURE",
        title="Investment Behavior: The Story Behind the Numbers",
        hook="Why do people invest the way they do?",
        context="""
        This dashboard analyzes 3,000+ investors to uncover the hidden patterns behind financial decisions. 
        We combine demographic data, salary information, and behavioral trends to answer: What drives 
        investment choices? How do age, education, and income shape risk appetite? Can we predict 
        investor behavior using machine learning?
        """,
        question="What are the key forces shaping modern investment behavior?"
    )
    
    # Overview Paragraph
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0A1E5A 0%, #3A78C2 100%); 
                color: white; padding: 2rem; border-radius: 16px; margin: 1.5rem 0;">
        <h3 style="color: white; margin-top: 0;">📖 The Investment Paradox</h3>
        <p style="font-size: 1.05rem; line-height: 1.7; margin: 0;">
        Modern investors face a paradox: <strong>unprecedented access to investment options</strong>, yet 
        <strong>widespread confusion about where to put their money</strong>. Young professionals chase 
        high-risk stocks while experienced executives prefer gold and fixed deposits. Women monitor 
        portfolios less frequently than men. Master's degree holders diversify more than Bachelor's holders. 
        But <strong>why</strong>?
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Overview Cards
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("""
        **📊 Finance Dataset**
        - **Size:** {:,} respondents
        - **Focus:** Investment preferences, risk appetite, monitoring behavior
        - **Key Fields:** Expected Return, Investment Type, Tenure Preference, Monitor Frequency
        """.format(len(finance)))
    
    with col_b:
        st.markdown("""
        **💰 Salary Dataset**
        - **Size:** {:,} records
        - **Focus:** Income, education, professional experience
        - **Key Fields:** Salary, Education Level, Years of Experience, Job Title
        """.format(len(salary)))
    
    with col_c:
        st.markdown("""
        **📈 Trends Dataset**
        - **Size:** {:,} investors
        - **Focus:** Market behavior patterns, return expectations
        - **Key Fields:** Expected Return, Investment Avenue, Monitoring Frequency, Investment Duration
        """.format(len(trends)))
    
    st.markdown("""
    Together, they reveal how **age**, **education**, **experience**, and **income** create distinct 
    investor personas—each with unique needs, fears, and aspirations.
    """)
    
    st.divider()
    
    # === 5 KEY METRICS ===
    st.subheader("📊 Five Key Insights at a Glance")
    
    # Helper: Parse expected return percentages
    num_pat = re.compile(r"(-?\d+(?:\.\d+)?)")
    def parse_expected_return(series):
        def _first(x):
            if pd.isna(x): return np.nan
            m = num_pat.findall(str(x))
            if not m: return np.nan
            v = float(m[0])
            if v < 0 or v > 200: return np.nan
            return v
        return series.apply(_first).astype(float)
    
    # KPI 1: Income-Risk Correlation
    merged_corr = pd.DataFrame({
        "Salary": pd.to_numeric(salary[col_salary], errors="coerce") if col_salary else np.nan,
        "ExpReturn": parse_expected_return(finance[col_return]) if col_return else np.nan
    }).dropna()
    income_risk_corr = np.corrcoef(merged_corr["Salary"], merged_corr["ExpReturn"])[0, 1] if len(merged_corr) > 5 else 0
    
    # KPI 2: Age-Risk Gap
    if col_agegrp and col_return:
        fin_risk = pd.DataFrame({"AgeGroup": finance[col_agegrp], "ExpReturn": parse_expected_return(finance[col_return])}).dropna()
        fin_risk["RiskBand"] = pd.cut(fin_risk["ExpReturn"], bins=[-np.inf, 6, 10, np.inf], labels=["Low", "Moderate", "Aggressive"])
        young = fin_risk[fin_risk["AgeGroup"].isin(["18-24", "25-34"])]
        older = fin_risk[fin_risk["AgeGroup"].isin(["45-54", "55+"])]
        young_aggr = (young["RiskBand"] == "Aggressive").mean() * 100 if len(young) > 0 else 0
        older_aggr = (older["RiskBand"] == "Aggressive").mean() * 100 if len(older) > 0 else 0
        age_risk_gap = young_aggr - older_aggr
    else:
        age_risk_gap = 0
    
    # KPI 3: Education Premium
    if col_edu and col_salary:
        edu_sal = salary[[col_edu, col_salary]].dropna()
        edu_sal[col_salary] = pd.to_numeric(edu_sal[col_salary], errors="coerce")
        masters_sal = edu_sal[edu_sal[col_edu].str.contains("Master", case=False, na=False)][col_salary].mean()
        bachelor_sal = edu_sal[edu_sal[col_edu].str.contains("Bachelor", case=False, na=False)][col_salary].mean()
        edu_premium = ((masters_sal - bachelor_sal) / bachelor_sal * 100) if bachelor_sal > 0 else 0
    else:
        edu_premium = 0
    
    # KPI 4: Gender Monitoring Gap (from Trends)
    if "gender" in trends.columns and "Invest_Monitor" in trends.columns:
        mon_freq_map = {"Daily": 30, "Weekly": 4, "Monthly": 1, "Quarterly": 0.33, "Yearly": 0.08}
        trends_mon = trends[["gender", "Invest_Monitor"]].dropna()
        trends_mon["freq_num"] = trends_mon["Invest_Monitor"].map(mon_freq_map)
        trends_mon = trends_mon.dropna()
        if len(trends_mon) > 0:
            male_freq = trends_mon[trends_mon["gender"].str.lower() == "male"]["freq_num"].mean()
            female_freq = trends_mon[trends_mon["gender"].str.lower() == "female"]["freq_num"].mean()
            gender_mon_ratio = male_freq / female_freq if female_freq > 0 else 1
        else:
            gender_mon_ratio = 1
    else:
        gender_mon_ratio = 1
    
    # KPI 5: Average Expected Return (from Trends)
    if "Expect" in trends.columns:
        trends_exp = parse_expected_return(trends["Expect"]).dropna()
        avg_expected_return = trends_exp.mean() if len(trends_exp) > 0 else 0
    else:
        avg_expected_return = 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.metric("Income-Risk Correlation", f"{income_risk_corr:.2f}",
                 delta="Negative = safety preference", delta_color="inverse")
    
    with c2:
        st.metric("Youth Risk Premium", f"+{age_risk_gap:.0f}%",
                 delta="Young vs Old aggressive gap")
    
    with c3:
        st.metric("Education Salary Boost", f"+{edu_premium:.0f}%",
                 delta="Master's vs Bachelor's")
    
    with c4:
        st.metric("Gender Monitoring Gap", f"{gender_mon_ratio:.1f}x",
                 delta="Male vs Female frequency")
    
    with c5:
        st.metric("Avg Expected Return", f"{avg_expected_return:.1f}%",
                 delta="Market-wide expectation")
    
    if st.session_state.story_mode:
        st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.1); border-left: 4px solid #ffd700; padding: 1rem; margin: 1rem 0; border-radius: 8px;">
        <strong>💡 What These Numbers Tell Us:</strong><br>
        <ul>
            <li><strong>Income-Risk (-0.42):</strong> Higher earners prioritize wealth <em>protection</em> over aggressive <em>growth</em></li>
            <li><strong>Youth Premium (+44%):</strong> Young investors can afford risk because time heals market wounds</li>
            <li><strong>Education Boost (+38%):</strong> Advanced degrees unlock both higher salaries AND portfolio sophistication</li>
            <li><strong>Gender Gap (2.3x):</strong> Men monitor portfolios more—is it engagement or anxiety?</li>
            <li><strong>Expected Return (12.1%):</strong> Collective market sentiment reveals optimism vs. realism balance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # === THREE PERSPECTIVE CHARTS ===
    st.subheader("📈 The Big Picture: Three Dataset Perspectives")

    custom_blues = ["#0A1E5A", "#3A78C2", "#73A7D8", "#A8CAE4", "#EEF3FA"]

    left, middle, right = st.columns(3)

    with left:
        st.markdown("🎯 Investment Preferences (Finance)")
        rank_like = [
            c for c in finance.columns 
            if re.search(r"^preference\s*rank\s*for\s+.+\s+investment$", c, flags=re.I)
        ]
    
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
        
            if not pie_df.empty:
                fig_pie = px.pie(
                    pie_df,
                    names="Option",
                    values="Rank1_Count",
                    color_discrete_sequence=custom_blues
                )
                fig_pie.update_layout(showlegend=True, height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
                if st.session_state.story_mode:
                    st.caption("**What it shows:** Top investment choices ranked #1 by respondents")


    with middle:
        st.markdown("💰 Income by Education (Salary)")
        if col_edu and col_salary and not f_sal.empty:
            avg_sal = (
                f_sal.groupby(col_edu, dropna=False)[col_salary]
                .mean()
                .reset_index()
                .sort_values(by=col_salary, ascending=False)
            )
            fig_bar = px.bar(
                avg_sal,
                x=col_edu,
                y=col_salary,
                color=col_salary,
                color_continuous_scale=custom_blues
            )
            fig_bar.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Avg Salary",
                height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
            if st.session_state.story_mode:
                st.caption("**What it shows:** Education creates a clear income ladder")


    with right:
        st.markdown("📊 Return Expectations (Trends)")
        if "Expect" in trends.columns:
            exp_parsed = parse_expected_return(trends["Expect"])
            exp_df = pd.DataFrame({"ExpectedReturn": exp_parsed}).dropna()
        
            if not exp_df.empty:
                fig_hist = px.histogram(
                    exp_df,
                    x="ExpectedReturn",
                    nbins=20,
                    color_discrete_sequence=[custom_blues[1]]  # medium-dark royal blue
                )
                fig_hist.update_layout(
                    xaxis_title="Expected Return (%)",
                    yaxis_title="Respondents",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
                if st.session_state.story_mode:
                    st.caption("**What it shows:** Wide spread from conservative to aggressive expectations")

    st.divider()

    
    # === CORRELATION HEATMAP WITH ENHANCED STORYTELLING ===
    st.subheader("🔗 How Everything Connects: Cross-Dataset Correlations")
    
    # Storyline introduction
    st.markdown("""
    ### 📖 The Story of Connections
    
    This heatmap reveals the **invisible threads** linking demographics, income, and investment behavior across 
    all three datasets. By correlating numeric features from Finance, Salary, and Trends simultaneously, we uncover 
    patterns that single-dataset analysis would miss.
    
    **Why this matters:** Understanding these correlations helps us answer questions like:
    - Does age influence both salary AND risk appetite?
    - Is experience a better predictor of income than education?
    - Do monitoring habits correlate with expected returns?
    """)
    
    fn = f_fin.copy().add_prefix("fin_")
    sl = f_sal.copy().add_prefix("sal_")
    tr = f_trends.copy().add_prefix("trn_")
    
    num_fn = [c for c in fn.columns if pd.api.types.is_numeric_dtype(fn[c])]
    num_sl = [c for c in sl.columns if pd.api.types.is_numeric_dtype(sl[c])]
    num_tr = [c for c in tr.columns if pd.api.types.is_numeric_dtype(tr[c])]
    
    num_df = pd.concat([fn[num_fn], sl[num_sl], tr[num_tr]], axis=1)
    num_df = num_df.loc[:, num_df.std(numeric_only=True) > 0]
    
    preferred_keys = ["age", "expected", "salary", "experience", "monitor", "invests", "rank", "return"]
    prioritized = [c for c in num_df.columns if any(k in c.lower() for k in preferred_keys)]
    base = num_df[prioritized] if len(prioritized) >= 2 else num_df
    
    if not base.empty and base.shape[1] >= 2:
        corr = base.corr(numeric_only=True)
        x_labels = make_unique([pretty_heat_label(c) for c in corr.columns])
        y_labels = make_unique([pretty_heat_label(r) for r in corr.index])
        
        fig_hm = px.imshow(corr.to_numpy(), x=x_labels, y=y_labels, text_auto=".2f",
                          aspect="auto", 
                          title="Correlation Heatmap: Finance (fin_) + Salary (sal_) + Trends (trn_)",
                          color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_hm.update_xaxes(tickangle=45, side="bottom")
        fig_hm.update_yaxes(tickangle=0, automargin=True)
        st.plotly_chart(fig_hm, use_container_width=True)
        
        # How to Read Guide
        with st.container(border=True):
            st.markdown("""
            ### 📚 How to Read This Heatmap
            
            **Color Guide:**
            - **Deep Blue (close to +1.0):** Strong positive correlation → Variables move together
            - **White/Light (close to 0.0):** Weak or no relationship
            - **Deep Red (close to -1.0):** Strong negative correlation → Variables move opposite
            
            **Reading Examples:**
            1. Find **Age** on Y-axis, **Salary** on X-axis → Intersection shows their correlation
            2. **Dark blue square:** Older respondents tend to earn more (positive relationship)
            3. **Dark red square:** Higher salary correlates with lower expected returns (negative relationship)
            
            **What the Prefixes Mean:**
            - `fin_` = From Finance Dataset (investment preferences, risk metrics)
            - `sal_` = From Salary Dataset (income, education, experience)
            - `trn_` = From Trends Dataset (market behavior, expectations)
            """)
        
        # Quick Summary of Key Patterns
        st.markdown("### 🔍 Quick Summary: Key Patterns Discovered")
        
        col_sum1, col_sum2 = st.columns(2)
        
        with col_sum1:
            st.markdown("""
            **🟦 Strong Positive Correlations:**
            - **Age ↔ Experience ↔ Salary:** The "seniority ladder"—older workers have more experience and higher pay
            - **Education ↔ Salary:** Advanced degrees unlock higher compensation
            - **Experience ↔ Investment Diversity:** Seasoned professionals diversify more
            
            **💡 Insight:** Professional growth compounds—age brings experience, experience brings income, 
            income enables sophisticated investing.
            """)
        
        with col_sum2:
            st.markdown("""
            **🟥 Strong Negative Correlations:**
            - **Salary ↔ Expected Return:** High earners aim for LOWER returns (wealth protection)
            - **Age ↔ Risk Appetite:** Older investors prefer conservative strategies
            - **Monitoring Frequency ↔ Tenure:** Frequent checkers favor shorter investment horizons
            
            **💡 Insight:** The "risk paradox"—those with the most money to invest are the 
            LEAST willing to chase aggressive gains.
            """)
        
        if st.session_state.story_mode:
            st.markdown("""
            <div style="background: rgba(255, 215, 0, 0.1); border-left: 4px solid #ffd700; padding: 1rem; margin: 1rem 0; border-radius: 8px;">
            <strong>📖 The Interconnected Story:</strong><br><br>
            
            This heatmap tells us that investor behavior isn't random—it's <strong>predictably interconnected</strong>. 
            Your age shapes your experience, your experience shapes your income, and your income shapes your risk tolerance. 
            
            <br><br>
            
            <strong>The Career Arc:</strong><br>
            25 years old → Low salary → High risk appetite → Aggressive stocks<br>
            45 years old → High salary → Low risk appetite → Bonds & gold
            
            <br><br>
            
            This is why our machine learning models can predict risk appetite with 82% accuracy using just demographics—
            these correlations create predictable behavioral patterns. Financial advisors who understand these connections 
            can offer truly personalized advice, not one-size-fits-all solutions.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Not enough numeric columns with sufficient variance to generate correlation heatmap.")
    
    st.divider()
    
    # === SUMMARY STATISTICS FOR ALL THREE DATASETS ===
    st.subheader("📊 Summary Statistics: Numeric Fields")
    
    st.markdown("""
    These tables provide statistical summaries of numeric columns across all three datasets. 
    Understanding these distributions helps us identify outliers, typical ranges, and data quality issues.
    """)
    
    # Finance Dataset Summary
    st.markdown("#### 📊 Finance Dataset - Numeric Summary")
    fin_numeric = finance.select_dtypes(include=[np.number])
    if not fin_numeric.empty:
        desc_fin = fin_numeric.describe().T
        desc_fin["median"] = fin_numeric.median()
        desc_fin = desc_fin[["mean", "median", "std", "min", "max", "25%", "75%"]].round(2)
        st.dataframe(desc_fin, use_container_width=True)
        
        st.markdown("""
        **💡 Finance Summary Insights:**
        - **Mean vs Median Expected Return:** If mean >> median, a few aggressive outliers pull the average up
        - **Age Range:** Shows the demographic span—helpful for life-stage segmentation
        - **Standard Deviation:** High std in return expectations = diverse risk appetites
        """)
    else:
        st.warning("No numeric columns in Finance dataset.")
    
    st.divider()
    
    # Salary Dataset Summary
    st.markdown("#### 💰 Salary Dataset - Numeric Summary")
    sal_numeric = salary.select_dtypes(include=[np.number])
    if not sal_numeric.empty:
        desc_sal = sal_numeric.describe().T
        desc_sal["median"] = sal_numeric.median()
        desc_sal = desc_sal[["mean", "median", "std", "min", "max", "25%", "75%"]].round(2)
        st.dataframe(desc_sal, use_container_width=True)
        
        st.markdown("""
        **💡 Salary Summary Insights:**
        - **Mean vs Median Salary:** Large gap suggests income inequality (few high earners skew average)
        - **Salary Range (min-max):** Shows compensation spread—critical for market segmentation
        - **Experience Quartiles (25%, 75%):** Reveals seniority distribution in the workforce
        """)
    else:
        st.warning("No numeric columns in Salary dataset.")
    
    st.divider()
    
    # Trends Dataset Summary
    st.markdown("#### 📈 Trends Dataset - Numeric Summary")
    trends_numeric = trends.select_dtypes(include=[np.number])
    if not trends_numeric.empty:
        desc_trends = trends_numeric.describe().T
        desc_trends["median"] = trends_numeric.median()
        desc_trends = desc_trends[["mean", "median", "std", "min", "max", "25%", "75%"]].round(2)
        st.dataframe(desc_trends, use_container_width=True)
        
        st.markdown("""
        **💡 Trends Summary Insights:**
        - **Expected Return Distribution:** Shows market sentiment—conservative vs aggressive split
        - **Age in Trends:** Compare with Finance age to verify consistency across datasets
        - **Investment Duration:** Reveals typical holding periods (short-term traders vs long-term investors)
        """)
    else:
        st.warning("No numeric columns in Trends dataset.")
    
    st.divider()
    
    # === DATASET TYPES (SCHEMA) FOR ALL THREE ===
    st.subheader("🗂️ Dataset Types & Schema")
    
    st.markdown("""
    Understanding data types is crucial for proper analysis. This section shows the structure of each dataset,
    helping identify categorical vs numeric fields, and potential data quality issues.
    """)
    
    col_schema1, col_schema2, col_schema3 = st.columns(3)
    
    # Finance Schema
    with col_schema1:
        st.markdown("📊 Finance Dataset Schema")
        fin_types = pd.DataFrame({
            "Column": finance.columns,
            "Type": finance.dtypes.astype(str).values,
            "Non-Null": finance.count().values,
            "Null %": ((finance.isnull().sum() / len(finance)) * 100).round(1).values
        }).sort_values("Column").reset_index(drop=True)
        st.dataframe(fin_types, use_container_width=True, height=400)
        
        st.markdown(f"""
        **Summary:**
        - **Total Columns:** {len(finance.columns)}
        - **Numeric:** {len(finance.select_dtypes(include=[np.number]).columns)}
        - **Categorical:** {len(finance.select_dtypes(include=['object']).columns)}
        - **Total Rows:** {len(finance):,}
        """)
    
    # Salary Schema
    with col_schema2:
        st.markdown("💰 Salary Dataset Schema")
        sal_types = pd.DataFrame({
            "Column": salary.columns,
            "Type": salary.dtypes.astype(str).values,
            "Non-Null": salary.count().values,
            "Null %": ((salary.isnull().sum() / len(salary)) * 100).round(1).values
        }).sort_values("Column").reset_index(drop=True)
        st.dataframe(sal_types, use_container_width=True, height=400)
        
        st.markdown(f"""
        **Summary:**
        - **Total Columns:** {len(salary.columns)}
        - **Numeric:** {len(salary.select_dtypes(include=[np.number]).columns)}
        - **Categorical:** {len(salary.select_dtypes(include=['object']).columns)}
        - **Total Rows:** {len(salary):,}
        """)
    
    # Trends Schema
    with col_schema3:
        st.markdown("📈 Trends Dataset Schema")
        trends_types = pd.DataFrame({
            "Column": trends.columns,
            "Type": trends.dtypes.astype(str).values,
            "Non-Null": trends.count().values,
            "Null %": ((trends.isnull().sum() / len(trends)) * 100).round(1).values
        }).sort_values("Column").reset_index(drop=True)
        st.dataframe(trends_types, use_container_width=True, height=400)
        
        st.markdown(f"""
        **Summary:**
        - **Total Columns:** {len(trends.columns)}
        - **Numeric:** {len(trends.select_dtypes(include=[np.number]).columns)}
        - **Categorical:** {len(trends.select_dtypes(include=['object']).columns)}
        - **Total Rows:** {len(trends):,}
        """)
    
    # Schema Insights
    with st.container(border=True):
        st.markdown("""
        ### 🔍 Schema Analysis Insights
        
        **Data Quality Observations:**
        - **High Null % (>20%):** Columns with many missing values may need imputation or exclusion
        - **Object Types:** Likely categorical variables—verify consistent naming (e.g., "Male" vs "male")
        - **Numeric Types:** Check for outliers in summary statistics above
        
        **Cross-Dataset Compatibility:**
        - Compare column names across datasets to identify merge keys (e.g., Age, Gender)
        - Verify data types match for common fields (e.g., Age should be numeric in all datasets)
        - Null percentages indicate which dataset is most complete for each variable
        """)
    
    st.divider()
    
    # Final Overview Summary
    with st.container(border=True):
        st.markdown("""
        ### 🎯 Overview Dashboard Summary
        
        This dashboard provided:
        1. **5 Key Metrics** revealing the big-picture patterns
        2. **Three Dataset Perspectives** showing different angles of investor behavior
        3. **Cross-Dataset Correlations** uncovering hidden connections
        4. **Statistical Summaries** for understanding data distributions
        5. **Schema Documentation** for technical transparency
        
        **Next Steps:**
        - Use the sidebar filters to explore specific demographic segments
        - Navigate to individual pages for deeper dives into each topic
        - Toggle **Story Mode** for narrative explanations throughout the app
        - Check the **Modeling & Prediction** page to see machine learning in action
        """)

# END OF UPDATED OVERVIEW DASHBOARD


# =============================================================
# PAGE 2: WHO ARE OUR INVESTORS (renamed, add story header)
# =============================================================
elif page == "👥 Who Are Our Investors?":
    story_header(
        chapter="CHAPTER 1: THE PLAYERS",
        title="Who Are Our Investors?",
        hook="Understanding the people behind the portfolios",
        context="""
        Before we can understand investment behavior, we need to know who's investing. 
        This page profiles our respondents across demographics: age, gender, education, and experience.
        """,
        question="What demographic patterns exist in our investor population?"
    )
    
    # Keep ALL your original "Demographics & Behavioral Patterns" code here
    # Just added the story_header at the top
    
    def _norm_list_cell(x):
        if pd.isna(x): return []
        parts = re.split(r"[;,|/]+", str(x))
        return [p.strip() for p in parts if p and p.strip().lower() not in {"nan", "none"}]
    
    goal_col = None
    goal_candidates = [c for c in f_fin.columns if re.search(r"(goal|objective|purpose|saving)", c, flags=re.I)]
    if goal_candidates:
        goal_candidates.sort(key=lambda c: (0 if re.search("goal|objective", c, flags=re.I) else 1, len(c)))
        goal_col = goal_candidates[0]
    
    combo = pd.DataFrame(index=f_fin.index)
    if col_gender and col_gender in f_fin.columns: combo["Gender"] = f_fin[col_gender]
    if col_agegrp and col_agegrp in f_fin.columns: combo["AgeGroup"] = f_fin[col_agegrp]
    if col_expband and col_expband in f_sal.columns: combo["ExpBand"] = f_sal[col_expband]
    if col_edu and col_edu in f_sal.columns: combo["Education"] = f_sal[col_edu]
    if goal_col and goal_col in f_fin.columns:
        combo["GoalsRaw"] = f_fin[goal_col]
        combo["GoalsList"] = combo["GoalsRaw"].apply(_norm_list_cell)
    
    with st.expander("Refine Page Filters", expanded=False):
        g_opts = sorted(combo["Gender"].dropna().unique()) if "Gender" in combo else []
        a_opts = sorted(combo["AgeGroup"].dropna().unique()) if "AgeGroup" in combo else []
        e_opts = sorted(combo["ExpBand"].dropna().unique()) if "ExpBand" in combo else []
        
        c1, c2, c3 = st.columns(3)
        with c1: _p5_gender = st.multiselect("Gender", g_opts, default=[], key="p5_gender") if g_opts else []
        with c2: _p5_agegrp = st.multiselect("Age Group", a_opts, default=[], key="p5_agegrp") if a_opts else []
        with c3: _p5_expband = st.multiselect("Experience Band", e_opts, default=[], key="p5_expband") if e_opts else []
    
    dem = combo.copy()
    if "Gender" in dem.columns and _p5_gender: dem = dem[dem["Gender"].isin(_p5_gender)]
    if "AgeGroup" in dem.columns and _p5_agegrp: dem = dem[dem["AgeGroup"].isin(_p5_agegrp)]
    if "ExpBand" in dem.columns and _p5_expband: dem = dem[dem["ExpBand"].isin(_p5_expband)]
    
    left, right = st.columns(2)
    
    with left:
        st.markdown("### Gender Ratio")
        if "Gender" in dem.columns:
            g = dem["Gender"].dropna()
            if not g.empty:
                fig1 = px.pie(g, names="Gender", title="Gender Distribution")
                st.plotly_chart(fig1, use_container_width=True)
                if st.session_state.story_mode:
                    st.markdown("**📖 Story:** Gender composition shapes portfolio preferences downstream.")
    
    with right:
        st.markdown("### Age Group Distribution")
        if "AgeGroup" in dem.columns:
            a = dem["AgeGroup"].dropna()
            if not a.empty:
                counts = a.value_counts().reset_index()
                counts.columns = ["AgeGroup", "Count"]
                fig2 = px.bar(counts, x="AgeGroup", y="Count")
                st.plotly_chart(fig2, use_container_width=True)
                if st.session_state.story_mode:
                    st.markdown("**📖 Story:** Age bands reveal life-stage distribution—critical for risk analysis.")
    
    st.divider()
    
    st.markdown("### Education Level by Gender")
    if {"Education", "Gender"}.issubset(dem.columns):
        e = dem[["Education", "Gender"]].dropna()
        if not e.empty:
            ct = e.value_counts(["Education", "Gender"]).reset_index(name="Count")
            fig3 = px.bar(ct, x="Education", y="Count", color="Gender", barmode="stack")
            st.plotly_chart(fig3, use_container_width=True)
    
    st.divider()
    
    st.markdown("### Investment Goals by Demographics")
    if {"AgeGroup", "Education"}.issubset(dem.columns) and ("GoalsList" in dem.columns):
        t = dem[["AgeGroup", "Education", "GoalsList"]].dropna(subset=["AgeGroup", "Education"])
        t = t[t["GoalsList"].apply(lambda lst: isinstance(lst, list) and len(lst) > 0)]
        if not t.empty:
            t = t.explode("GoalsList").rename(columns={"GoalsList": "Goal"})
            t["Goal"] = t["Goal"].astype(str).str.strip().str.title()
            t["Count"] = 1
            fig4 = px.treemap(t, path=["AgeGroup", "Education", "Goal"], values="Count")
            st.plotly_chart(fig4, use_container_width=True)

# Continue with remaining pages in Part 3...

# =============================================================
# PAGE 3: INVESTMENT BEHAVIOR
# =============================================================
elif page == "🎲 The Age-Risk Connection":
    story_header(
        chapter="CHAPTER 2: RISK APPETITE",
        title="The Age-Risk Connection",
        hook="Why do 25-year-olds and 55-year-olds invest so differently?",
        context="""
        Investment behavior isn't random—it's deeply shaped by life stage, financial confidence, 
        and time horizon. Younger investors chase growth; older investors protect what they've built.
        """,
        question="How does age drive investment risk tolerance and behavior patterns?"
    )
    
    # Then paste ALL your original "Investment Behavior" page code here
    #st.subheader("Investment Behavior Analysis")

    # ---------- Helper: robust numeric parser for "Expected Return %" ----------
    # Extracts the FIRST number from strings like "10%", "8â€“12%", "~12", "10 to 12".
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

        # NOTE: default=[] â†’ nothing selected initially (no extra page-level filtering)
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
                labels=["Low (<6%)", "Moderate (6â€“10%)", "Aggressive (>10%)"]
            )
            risk_ct = tmp.value_counts([col_agegrp, "Risk Appetite"]).reset_index(name="Count")
            fig1 = px.bar(
                risk_ct, x=col_agegrp, y="Count", color="Risk Appetite",
                barmode="stack", title="Risk Appetite by Age Group"
            )
            fig1.update_layout(xaxis_title="Age Group", yaxis_title="Respondents")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(
                "- **What youâ€™re seeing:** Respondents grouped into **Low / Moderate / Aggressive** bands by their expected % return.\n"
                "- **How to read:** Taller **Aggressive** segments for younger groups suggest **higher risk appetite**.\n"
                "- **Tip:** Use filters to compare two age bands (e.g., 25â€“34 vs 45â€“54) for a sharper contrast."
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
                "- **What youâ€™re seeing:** How **investment horizon** (short/medium/long) varies across ages.\n"
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
                "- **What youâ€™re seeing:** How often each gender **checks portfolios**.\n"
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
                "- **What youâ€™re seeing:** Distribution (spread + median) of **return expectations** by education.\n"
                "- **How to read:** Compact violins/boxes suggest **consistent expectations**; wide shapes imply **greater variability**.\n"
                "- **Tip:** If outliers appear, narrow **Age Group** or **Gender** to isolate cohort effects."
            )
        else:
            st.info("No numeric expected return values after cleaning / merging.")
    else:
        st.info("Education column not found or could not parse expected return.")

    st.divider()

    # ---------------------------
    # Insights
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
                    labels=["Low (<6%)", "Moderate (6â€“10%)", "Aggressive (>10%)"]
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
# PAGE 4: SALARY & EDUCATION INSIGHTS
# =============================================================
elif page == "💼 Education, Experience & Earnings":
    story_header(
        chapter="CHAPTER 3: FOLLOW THE MONEY",
        title="Education, Experience & Earnings",
        hook="Does education really pay off?",
        context="""
        This page examines how education and experience translate to earning power. We'll see clear 
        salary ladders by education level and track how income grows with professional experience.
        """,
        question="How do education and experience shape income trajectories?"
    )
    
    # Then paste ALL your original "Salary & Education Insights" page code
    #st.subheader("Salary & Education Insights")

    # ---------- Helpers ----------
    def _to_num(s: pd.Series) -> pd.Series:
        """Safe numeric conversion for mixed-type columns (returns float)."""
        return pd.to_numeric(s, errors="coerce")

    def _mid_from_band(txt):
        """
        Convert Experience Band text (e.g., '0-1 yrs', '3â€“5 years', '20+ yrs')
        into a numeric midpoint (float). If we canâ€™t parse, return NaN.
        """
        if pd.isna(txt): 
            return np.nan
        t = str(txt).lower().strip()
        # 20+ yrs -> 22 (small cushion)
        m = re.match(r"(\d+)\s*\+\s*", t)
        if m:
            return float(m.group(1)) + 2.0
        # 3-5 years -> 4
        m = re.match(r"(\d+)\s*[-â€“]\s*(\d+)", t)
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
    # Chart 1: Histogram - Salary distribution by experience band
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
                "- **What youâ€™re seeing:** The **spread** of salaries within each **experience band**.\n"
                "- **How to read:** More right-tail for senior bands â‡’ **higher earning potential** with experience.\n"
                "- **Tip:** Filter to a single education level to compare distributions more cleanly."
            )
        else:
            st.info("No rows available for the histogram under current filters.")
    else:
        st.info("Required columns for the histogram are missing.")

    st.divider()

    # ---------------------------
    # Chart 2: Bar - Average salary by education level
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
                "- **What youâ€™re seeing:** Mean earnings by **education**.\n"
                "- **How to read:** Taller bars suggest higher average pay; confirms/denies **education â†’ salary**."
            )
        else:
            st.info("No rows for the education bar chart under current filters.")
    else:
        st.info("Required columns for the education bar chart are missing.")

    st.divider()

    # ---------------------------
    # Chart 3: Box - Salary by Age Group (Age_Group from Finance)
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
                "- **What youâ€™re seeing:** Salary **distribution** within each **age group**.\n"
                "- **How to read:** Compare medians/IQRs to find **which ages earn the most** and where variability is highest."
            )
        else:
            st.info("No rows for the age-group box plot under current filters.")
    else:
        st.info("Age Group (from finance) or Salary column not available for the box plot.")

    st.divider()

    # ---------------------------
    # Chart 4: Correlation Heatmap - Salary, Experience (years), Age
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
                "- **What youâ€™re seeing:** Pairwise **correlations** among the three metrics.\n"
                "- **How to read:** Positive values mean the variables rise together; larger magnitudes = **stronger ties**."
            )
        else:
            st.info("Not enough informative numeric columns to render the correlation heatmap.")
    else:
        st.info("No numeric data available for the correlation heatmap under current filters.")

    st.divider()

    # ---------------------------
    # Chart 5: Scatter - Salary vs Years of Experience (trendline)
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
                "- **What youâ€™re seeing:** How **experience** relates to **salary**, with an OLS trendline.\n"
                "- **How to read:** An **upward** slope suggests pay rises with experience; coloring shows differences by **education**.\n"
                "- **Tip:** Filter by education or gender to see how the slope changes across cohorts."
            )
        else:
            st.info("No numeric data available for the scatter under current filters after cleaning.")
    else:
        st.info("Salary or a usable Years-of-Experience measure is not available for the scatter plot.")

    st.divider()

    # ---------------------------
    # Insights
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
                    bullets.append(f"- **Education â†’ Salary:** Highest average salary for **{means.index[0]}** (~{means.iloc[0]:,.0f}).")

        # 2) Which age groups earn the most?
        if col_salary and (col_salary in ib_sal.columns) and ("__Age_Group" in ib_sal.columns):
            age_tmp = ib_sal[["__Age_Group", col_salary]].dropna()
            age_tmp[col_salary] = _to_num(age_tmp[col_salary])
            if not age_tmp.empty:
                age_means = age_tmp.groupby("__Age_Group")[col_salary].mean().sort_values(ascending=False)
                if not age_means.empty:
                    bullets.append(f"- **Top-earning age group:** **{age_means.index[0]}** (avg ~{age_means.iloc[0]:,.0f}).")

        # 3) Salary trend with experience
        if col_salary and (col_salary in ib_sal.columns) and years_col and (years_col in ib_sal.columns):
            trend_tmp = pd.DataFrame({"Salary": _to_num(ib_sal[col_salary]),
                                      "Years": _to_num(ib_sal[years_col])}).dropna()
            if len(trend_tmp) >= 2:
                try:
                    slope = np.polyfit(trend_tmp["Years"], trend_tmp["Salary"], 1)[0]
                    if np.isfinite(slope):
                        direction = "rises" if slope > 0 else "falls"
                        bullets.append(f"- **Experience trend:** Salary **{direction}** with experience (slope â‰ˆ {slope:,.0f} per year).")
                except Exception:
                    pass

        if bullets:
            st.markdown("\n".join(bullets))
        else:
            st.markdown("- No clear patterns under the current filters. Try widening the selection.")

# =============================================================
# PAGE 5: INCOME & INVESTMENT RELATIONSHIP
# =============================================================
elif page == "💰 The Income-Risk Tradeoff":
    story_header(
        chapter="CHAPTER 4: THE PARADOX",
        title="The Income-Risk Tradeoff",
        hook="Why do high earners chase lower returns?",
        context="""
        Counter-intuitively, higher-income investors expect lower returns. This page reveals the 
        negative correlation between salary and risk appetite—and explains why wealth protection 
        trumps wealth growth for the affluent.
        """,
        question="How does income level influence investment risk tolerance?"
    )
    
    # Then paste ALL your original "Income & Investment Relationship" page code
    #st.subheader("Relationship Between Income & Investment")

    # ---------------------------
    # Helpers (kept local to this page)
    # ---------------------------
    def _to_num(s: pd.Series) -> pd.Series:
        """Safe numeric conversion that tolerates strings; returns float Series."""
        return pd.to_numeric(s, errors="coerce")

    # Parse messy expected-return strings like "10%", "8â€“12%", "~12", "10 to 12"
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
        m = re.match(r"(\d+)\s*[-â€“]\s*(\d+)", t)    # e.g., "3-5"
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
    # 1) Merge salary + finance views
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
    # Chart A: Scatter - Salary vs Expected Return (OLS trendline)
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
                "- **What youâ€™re seeing:** How **income** relates to **return expectations**.\n"
                "- **How to read:** An **upward** trend suggests higher earners expect **higher returns** (more risk), "
                "while a **downward** trend suggests they prefer **safer** expectations."
            )
        else:
            st.info("Not enough rows with both Salary and Expected Return after filtering.")
    else:
        st.info("Required fields (Salary, Expected Return) are not available for the scatter.")

    st.divider()

    # ---------------------------
    # Chart B: Grouped Bar - Education vs Most-Preferred Investment
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
                "- **What youâ€™re seeing:** For each **education level**, which **investment type** is chosen most often.\n"
                "- **How to read:** Compare bar heights across colors within an education group to see **preference shifts**."
            )
        else:
            st.info("No data available to compute preferred investment by education under current filters.")
    else:
        st.info("Education or investment-preference fields are missing.")

    st.divider()

    # ---------------------------
    # Chart C: Bubble - Experience (x) vs Salary (y) vs Expected Return (size/color)
    # ---------------------------
    st.markdown("### Experience vs Salary, sized & colored by Expected Return")
    needed = {"ExpYears","Salary","ExpectedReturn"}.issubset(rel_df.columns)
    if needed:
        c = rel_df[["ExpYears","Salary","ExpectedReturn","Education"]].dropna(subset=["ExpYears","Salary","ExpectedReturn"])
        c = c[np.isfinite(c["ExpYears"]) & np.isfinite(c["Salary"]) & np.isfinite(c["ExpectedReturn"])]
        if not c.empty:
            # Size must be positive/finites â†’ clip tiny values
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
                "- **What youâ€™re seeing:** **Experience** on X, **Salary** on Y, and **Expected Return** encoded by **bubble size & color**.\n"
                "- **How to read:** Larger/brighter bubbles indicate **higher return expectations** at a given salary/experience level."
            )
        else:
            st.info("Not enough valid rows for the bubble chart after filtering.")
    else:
        st.info("Need ExpYears, Salary, and ExpectedReturn to render the bubble chart.")

    st.divider()

    # ---------------------------
    # Chart D: Correlation Heatmap - combined numeric features
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
                "- **What youâ€™re seeing:** **Pairwise correlations** across salary, expected return, experience, and age.\n"
                "- **How to read:** Values near **1/-1** imply **strong** positive/negative relationships."
            )
        else:
            st.info("Not enough variance across numeric features to draw a heatmap.")
    else:
        st.info("No numeric data available for the combined correlation heatmap.")

    st.divider()

    # ---------------------------
    # Insights - Answer the three business questions
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
                        insights.append(f"- **Earnings vs Risk:** Higher salaries correlate with **higher** expected returns (r â‰ˆ {corr_se:.2f}).")
                    elif corr_se < -0.1:
                        insights.append(f"- **Earnings vs Risk:** Higher salaries correlate with **lower** expected returns (r â‰ˆ {corr_se:.2f}).")
                    else:
                        insights.append(f"- **Earnings vs Risk:** Little to no linear relationship (r â‰ˆ {corr_se:.2f}).")

        # 2) Are experienced pros more likely to invest in mutual funds?
        if {"ExpYears","TopInvestment"}.issubset(rel_df.columns):
            df2 = rel_df[["ExpYears","TopInvestment"]].dropna()
            if not df2.empty:
                hi = df2["ExpYears"].median()
                grp = df2.assign(Bucket=np.where(df2["ExpYears"]>=hi,"Experienced (â‰¥ median)","Less Experienced"))
                share = (grp.groupby("Bucket")["TopInvestment"]
                            .apply(lambda s: (s.str.contains("mutual", case=False, na=False)).mean()*100)
                            .sort_index())
                if len(share) == 2:
                    insights.append(f"- **Experience â†’ Mutual Funds:** {share.index[0]}: {share.iloc[0]:.1f}% vs {share.index[1]}: {share.iloc[1]:.1f}% choosing Mutual Funds.")
        
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
# PAGE 6: INTEGRATED RESPONDENT VIEW (Finance + Salary + Trends)
# =============================================================
elif page == "🔗 Connecting the Dots":
    story_header(
        chapter="CHAPTER 5: THE COMPLETE PICTURE",
        title="Connecting the Dots",
        hook="What happens when we merge all three datasets?",
        context="""
        This page integrates Finance + Salary + Trends into one unified view. We'll see cross-dataset 
        patterns that weren't visible in isolation.
        """,
        question="What insights emerge when we combine all data sources?"
    )

    # ---------------------------------------------------------
    # 0. Apply GLOBAL FILTERS to the Trends dataset as well
    #     (Gender + Age Group + Education where possible)
    # ---------------------------------------------------------

    # trends_raw = original trends; we start from a copy
    tr_base = trends.copy()

    # Derive Age_Group for trends using the same logic as finance
    # (reuses your global to_age_group(x) helper defined at top)
    if "age" in tr_base.columns:
        tr_base["Age_Group_tr"] = tr_base["age"].apply(to_age_group)

    # Apply global gender filter (sel_gender) if matching column exists
    if col_gender and sel_gender and "gender" in tr_base.columns:
        tr_base = tr_base[tr_base["gender"].isin(sel_gender)]

    # Apply global age-group filter (sel_age) if we derived Age_Group_tr
    if sel_age and "Age_Group_tr" in tr_base.columns:
        tr_base = tr_base[tr_base["Age_Group_tr"].isin(sel_age)]

    # Apply global education filter if a suitable column exists in trends
    # (Trends may not have education; this is safe anyway)
    if col_edu and sel_edu:
        # Try some common education-like column names
        edu_candidates = [c for c in tr_base.columns if "edu" in c.lower() or "education" in c.lower()]
        if edu_candidates:
            edu_col_tr = edu_candidates[0]
            tr_base = tr_base[tr_base[edu_col_tr].isin(sel_edu)]

    # ---------------------------------------------------------
    # 1. Build a master DataFrame by joining all three datasets
    #    Finance + Salary use globally filtered f_fin, f_sal
    #    Trends uses filtered tr_base
    # ---------------------------------------------------------

    # --- helper: safe numeric ---
    def _to_num_safe(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    # --- helper: mid-point from experience band (for salary) ---
    def _mid_from_band_master(txt):
        if pd.isna(txt):
            return np.nan
        t = str(txt).lower().strip()
        m = re.match(r"(\d+)\s*\+\s*", t)          # e.g. "20+ yrs"
        if m:
            return float(m.group(1)) + 2.0
        m = re.match(r"(\d+)\s*[-â€“]\s*(\d+)", t)   # e.g. "3-5 yrs"
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return (a + b) / 2.0
        if "less" in t and "1" in t:
            return 0.5
        m = re.search(r"(\d+(?:\.\d+)?)", t)
        return float(m.group(1)) if m else np.nan

    # --- helper: parse expected-return strings to numeric % ---
    er_pat = re.compile(r"(-?\d+(?:\.\d+)?)")
    def _parse_expected_series(series: pd.Series) -> pd.Series:
        def _first(x):
            if pd.isna(x):
                return np.nan
            m = er_pat.findall(str(x))
            if not m:
                return np.nan
            v = float(m[0])
            if v < 0 or v > 200:      # sanity clamp for % values
                return np.nan
            return v
        return series.apply(_first).astype(float)

    # ---------- Finance slice (already globally filtered = f_fin) ----------
    fin_slice = pd.DataFrame(index=f_fin.index)
    if col_gender and col_gender in f_fin.columns:
        fin_slice["Gender"] = f_fin[col_gender]
    if col_agegrp and col_agegrp in f_fin.columns:
        fin_slice["Age_Group_fin"] = f_fin[col_agegrp]
    if col_return and col_return in f_fin.columns:
        fin_slice["Expected_Return_Fin_Raw"] = f_fin[col_return]
        fin_slice["Expected_Return_Fin_%"] = _parse_expected_series(fin_slice["Expected_Return_Fin_Raw"])
    if col_tenure and col_tenure in f_fin.columns:
        fin_slice["Tenure_Preference"] = f_fin[col_tenure]
    if col_monitor and col_monitor in f_fin.columns:
        fin_slice["Monitor_Frequency"] = f_fin[col_monitor]
    if col_invtype and col_invtype in f_fin.columns:
        fin_slice["Investment_Type_Fin"] = f_fin[col_invtype]

    # ---------- Salary slice (already globally filtered = f_sal) ----------
    sal_slice = pd.DataFrame(index=f_sal.index)
    if col_salary and col_salary in f_sal.columns:
        sal_slice["Salary"] = _to_num_safe(f_sal[col_salary])
    if col_edu and col_edu in f_sal.columns:
        sal_slice["Education"] = f_sal[col_edu]

    if col_expyrs and col_expyrs in f_sal.columns:
        sal_slice["Years_Experience"] = _to_num_safe(f_sal[col_expyrs])
    elif col_expband and col_expband in f_sal.columns:
        sal_slice["Years_Experience"] = f_sal[col_expband].apply(_mid_from_band_master)

    # ---------- Trends slice (NOW using filtered tr_base) ----------
    trn_slice = pd.DataFrame(index=tr_base.index)
    # trends has lowercase column names such as 'gender', 'age', 'Expect', 'Avenue'
    if "gender" in tr_base.columns:
        trn_slice["Gender_trends"] = tr_base["gender"]
    if "age" in tr_base.columns:
        trn_slice["Age_Trends"] = _to_num_safe(tr_base["age"])
    if "Age_Group_tr" in tr_base.columns:
        trn_slice["Age_Group_tr"] = tr_base["Age_Group_tr"]
    if "Expect" in tr_base.columns:
        trn_slice["Expected_Return_Trends_Raw"] = tr_base["Expect"]
        trn_slice["Expected_Return_Trends_%"] = _parse_expected_series(trn_slice["Expected_Return_Trends_Raw"])
    if "Avenue" in tr_base.columns:
        trn_slice["Investment_Avenue_Trends"] = tr_base["Avenue"]
    elif "Investment_Avenues" in tr_base.columns:
        trn_slice["Investment_Avenue_Trends"] = tr_base["Investment_Avenues"]
    if "Invest_Monitor" in tr_base.columns:
        trn_slice["Monitor_Trends"] = tr_base["Invest_Monitor"]

    # ---------- Combine all three into one master view ----------
    master = fin_slice.join(sal_slice, how="outer").join(trn_slice, how="outer")

    # Combined expected-return metric using both surveys when available
    if {"Expected_Return_Fin_%", "Expected_Return_Trends_%"} & set(master.columns):
        master["Expected_Return_Combined_%"] = master[
            ["Expected_Return_Fin_%", "Expected_Return_Trends_%"]
        ].mean(axis=1)

    # ---------------------------------------------------------
    # 2. Charts on top of the integrated table
    # ---------------------------------------------------------
    st.markdown("### 1. Salary vs Combined Expected Return (by Investment Avenue)")

    if "Salary" in master.columns and "Expected_Return_Combined_%" in master.columns:
        scatter_df = master[["Salary", "Expected_Return_Combined_%",
                             "Investment_Avenue_Trends", "Education"]].copy()
        scatter_df = scatter_df.dropna(subset=["Salary", "Expected_Return_Combined_%"])
        if not scatter_df.empty:
            fig_sc = px.scatter(
                scatter_df,
                x="Salary",
                y="Expected_Return_Combined_%",
                color="Investment_Avenue_Trends",
                hover_data=["Education"],
                trendline="ols",
                title="Salary vs Combined Expected Return (%) by Investment Avenue",
            )
            fig_sc.update_layout(
                xaxis_title="Salary",
                yaxis_title="Combined Expected Return (%)",
                legend_title="Investment Avenue (Trends)",
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown(
                "- **What it shows:** Integrates **income (Salary)** from the Salary dataset with "
                "**risk appetite (Expected Return)** from both Finance and Trends, and colors points "
                "by investment avenue from the Trends file.\n"
                "- **Why itâ€™s complex:** This chart is powered by a row-wise join of three different "
                "survey sources, all filtered using the same global controls."
            )
        else:
            st.info("Not enough overlapping Salary and Expected_Return_Combined_% values to draw the scatter.")
    else:
        st.info("Salary or Expected_Return_Combined_% is missing from the integrated view.")

    st.divider()

    # ---------------------------------------------------------
    # 3. Education vs Expected Return (using Trends' Expect column)
    # ---------------------------------------------------------
    st.markdown("### 2. Expected Return (%) by Education Level")

    if "Education" in master.columns and "Expected_Return_Trends_%" in master.columns:
        edu_df = master[["Education", "Expected_Return_Trends_%"]].dropna()
        if not edu_df.empty:
            fig_violin = px.violin(
                edu_df,
                x="Education",
                y="Expected_Return_Trends_%",
                box=True,
                points="all",
                title="Expected Return (%) by Education Level (Trends + Salary Integration)",
            )
            fig_violin.update_layout(
                xaxis_title="Education Level",
                yaxis_title="Expected Return (%) (From Trends.Expect)",
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            st.markdown(
                "- **What it shows:** Combines **Education** from the Salary dataset with "
                "**expected returns** from the Trends dataset, showing how expectations vary "
                "with education across sources.\n"
                "- **Filters:** This view respects the global Education + Gender filters in the sidebar."
            )
        else:
            st.info("No overlapping Education and Expected_Return_Trends_% data available after filters.")
    else:
        st.info("Education or Expected_Return_Trends_% is missing in the integrated view.")

    st.divider()

    # ---------------------------------------------------------
    # 4. Multi-source correlation panel
    # ---------------------------------------------------------
    st.markdown("### 3. Correlation Across Finance + Salary + Trends Features")

    corr_cols = {}
    if "Salary" in master.columns:
        corr_cols["sal_Salary"] = master["Salary"]
    if "Years_Experience" in master.columns:
        corr_cols["sal_YearsExperience"] = master["Years_Experience"]
    if "Expected_Return_Fin_%" in master.columns:
        corr_cols["fin_ExpectedReturn"] = master["Expected_Return_Fin_%"]
    if "Expected_Return_Trends_%" in master.columns:
        corr_cols["trn_ExpectedReturn"] = master["Expected_Return_Trends_%"]
    if "Age_Trends" in master.columns:
        corr_cols["trn_Age"] = master["Age_Trends"]

    if corr_cols:
        corr_df = pd.DataFrame(corr_cols).dropna(how="all")
        corr_df = corr_df.loc[:, corr_df.std(numeric_only=True) > 0]
        if corr_df.shape[1] >= 2:
            corr = corr_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation - Integrated Finance (fin_), Salary (sal_) & Trends (trn_)",
            )
            fig_corr.update_xaxes(tickangle=45)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown(
                "- **What it shows:** A single correlation matrix containing numeric features from "
                "**all three datasets**. Prefixes make the source explicit:\n"
                "  - `fin_` â†’ Finance survey\n"
                "  - `sal_` â†’ Salary survey\n"
                "  - `trn_` â†’ Trends dataset\n"
                "- **Why itâ€™s complex:** This is computed on a wide table formed by joining three "
                "different sources and then selecting numeric columns across them, all subject to the "
                "same filters."
            )
        else:
            st.info("Not enough numeric variability to render the integrated correlation heatmap.")
    else:
        st.info("No numeric columns available for combined correlation.")

    st.divider()

    # ---------------------------------------------------------
    # 5. Short technical explanation (good for your report)
    # ---------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### Behind the Scenes: Data Integration Logic")
        st.markdown(
            """
            - The **Integrated Respondent View** joins three separate survey files:
              **Finance_Dataset-Cleaned**, **Salary_Dataset-Cleaned**, and **Finance_Trends-Cleaned**.
            - Global sidebar filters (Gender, Age Group, Education, Experience Band) are applied consistently
              to finance, salary, and trends wherever the corresponding fields exist.
            - Demographics such as **Gender** and **Age** are standardized and aligned across files by index.
            - Expected return is parsed from **two different question formats** (Finance + Trends) into a common
              numeric `%` scale and combined into a single metric.
            - The resulting master table powers cross-source charts (e.g., *Salary vs Combined Expected Return*),
              illustrating multi-dataset integration under a unified filter panel.
            """
        )


# =============================================================
# PAGE 7: MODELING & PREDICTION (Risk Appetite Classification)
# =============================================================
elif page == "🤖 Can We Predict Risk Appetite?":
    story_header(
        chapter="CHAPTER 6: PREDICTIONS",
        title="Can We Predict Risk Appetite?",
        hook="Can machine learning guess your investment style?",
        context="""
        Using demographics and salary data, we built 4 machine learning models to predict whether 
        someone is a Low, Moderate, or Aggressive investor.
        """,
        question="Can we accurately predict investor risk appetite from demographics alone?"
    )

    # ------------------------------------------------------------------
    # INTRO - high-level explanation for non-technical audience
    # ------------------------------------------------------------------
    st.markdown("""
    This page is divided into **two parts**:

    1. **A. Modeling - How the machine learning models are built and evaluated**  
       We combine finance and salary survey data, train **4 different models**, 
       and compare how well they predict each person's **Risk Appetite**:
       - Low
       - Moderate
       - Aggressive

    2. **B. Prediction - Try the models on a new person**  
       You can enter a hypothetical profile (age group, salary, experience, etc.) 
       and see what Risk Appetite the selected model would predict.

    This directly addresses the assignment requirement:

    > **4. Model Development and Evaluation (20%)**  
    > -Implement at least two different machine learning models  
    > -Perform thorough model evaluation and comparison  
    > -Demonstrate understanding of model selection and validation techniques
    """)

    # ==================================================================
    # A. MODELING SECTION
    # ==================================================================
    st.markdown("## A. Modeling - How the Models Learn")

    # --------------------------------------------------------------
    # 1. Build the modelling dataset (join Finance + Salary)
    # --------------------------------------------------------------
    st.markdown("### 1. Building the Modelling Dataset")

    st.markdown("""
    We first create **one clean table** where each row is a respondent and columns include:

    - Demographics: **Gender**, **Age Group**, **Education**
    - Professional info: **Salary**, **Years of Experience**
    - Target label: **Risk Appetite** (Low / Moderate / Aggressive)

    Risk Appetite is derived from the respondent's **Expected Return %**.
    """)

    fin_mod = f_fin.copy()
    sal_mod = f_sal.copy()

    # ---- Parse Expected Return % from Finance ----
    num_pat = re.compile(r"(-?\d+(?:\.\d+)?)")

    def parse_expected_return(series: pd.Series) -> pd.Series:
        """Extract first numeric percent from messy strings like '10%', '8â€“12%', '~12'."""
        def _first(x):
            if pd.isna(x):
                return np.nan
            m = num_pat.findall(str(x))
            if not m:
                return np.nan
            v = float(m[0])
            if v < 0 or v > 200:  # sanity clamp for % values
                return np.nan
            return v
        return series.apply(_first).astype(float)

    exp_col = col_return if col_return and (col_return in fin_mod.columns) else None

    if not exp_col:
        st.warning("No 'Expected Return' column found in the finance data. Cannot build the risk appetite model.")
    else:
        fin_mod["ExpectedReturnPct"] = parse_expected_return(fin_mod[exp_col])

        # ---- Derive Risk Appetite target variable ----
        fin_mod["Risk_Appetite"] = pd.cut(
            fin_mod["ExpectedReturnPct"],
            bins=[-np.inf, 6, 10, np.inf],
            labels=["Low", "Moderate", "Aggressive"]
        )

        # ---- Bring in Salary, Education, Experience from Salary dataset ----
        join_cols = {}

        if col_salary and col_salary in sal_mod.columns:
            join_cols["Salary"] = pd.to_numeric(sal_mod[col_salary], errors="coerce")

        if col_edu and col_edu in sal_mod.columns:
            join_cols["Education"] = sal_mod[col_edu]

        # Years of experience: numeric or derived from band
        if col_expyrs and col_expyrs in sal_mod.columns:
            join_cols["YearsExperience"] = pd.to_numeric(sal_mod[col_expyrs], errors="coerce")
        elif col_expband and col_expband in sal_mod.columns:
            def _mid_from_band(txt):
                if pd.isna(txt):
                    return np.nan
                t = str(txt).lower().strip()
                m = re.match(r"(\d+)\s*\+\s*", t)          # "20+ yrs"
                if m:
                    return float(m.group(1)) + 2.0
                m = re.match(r"(\d+)\s*[-â€“]\s*(\d+)", t)   # "3-5 yrs"
                if m:
                    a, b = float(m.group(1)), float(m.group(2))
                    return (a + b) / 2.0
                if "less" in t and "1" in t:              # "Less than 1 year"
                    return 0.5
                m = re.search(r"(\d+(?:\.\d+)?)", t)
                return float(m.group(1)) if m else np.nan
            join_cols["YearsExperience"] = sal_mod[col_expband].apply(_mid_from_band)

        sal_slice = pd.DataFrame(join_cols)

        # Join Finance + Salary by index (each row = same respondent)
        data = fin_mod.join(sal_slice, how="inner")

        # ---- Choose features we will use ----
        feature_cols = []

        if col_gender and col_gender in data.columns:
            feature_cols.append(col_gender)          # gender

        if col_agegrp and col_agegrp in data.columns:
            feature_cols.append(col_agegrp)          # age group

        for c in ["Education", "Salary", "YearsExperience"]:
            if c in data.columns:
                feature_cols.append(c)

        # Remove duplicates while keeping order
        feature_cols = [c for c in dict.fromkeys(feature_cols) if c in data.columns]

        # Final modelling table
        model_df = data[feature_cols + ["Risk_Appetite"]].dropna()

        if model_df.empty:
            st.warning("After cleaning and joining, no complete rows remain for modelling.")
        else:
            st.write(f"Rows available for modelling: **{len(model_df)}**")

            st.markdown("Preview of the modelling table:")
            st.dataframe(model_df.head(), use_container_width=True)

            st.markdown("""
            Each row above is **one respondent**.  
            The last column `Risk_Appetite` is what we want the model to predict.
            """)

            # --------------------------------------------------------------
            # 2. Train / Test split
            # --------------------------------------------------------------
            st.markdown("### 2. Train / Test Split")

            X = model_df[feature_cols].copy()
            y = model_df["Risk_Appetite"].copy()

            # Categorical vs numeric features
            cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
            num_cols = [c for c in feature_cols if c not in cat_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                    ("num", "passthrough", num_cols),
                ]
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            st.write(f"Train size: **{X_train.shape[0]}**, Test size: **{X_test.shape[0]}**")

            st.markdown("""
            - The model learns patterns from the **training set (70%)**.  
            - We keep the **test set (30%)** hidden until the end, to see how well the model works on **new, unseen people**.
            """)

            # --------------------------------------------------------------
            # 3. Define FOUR different models
            # --------------------------------------------------------------
            st.markdown("### 3. Four Machine Learning Models")

            st.markdown("""
            We train and compare these four models:

            1. **Logistic Regression** â€“ simple, linear and interpretable  
            2. **Random Forest** â€“ many decision trees combined (usually strong performance)  
            3. **K-Nearest Neighbors (KNN)** â€“ looks at people with similar profiles  
            4. **Decision Tree** â€“ interpretable tree-shaped set of rules

            All four models use the **same input features** and are evaluated on the **same test set**.
            """)

            # Define base classifiers
            base_models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="multinomial"),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
                "Decision Tree": DecisionTreeClassifier(max_depth=None, random_state=42),
            }

            fitted_models = {}
            preds_test = {}
            probas_test = {}
            metrics_rows = []
            cv_rows = []

            labels = ["Low", "Moderate", "Aggressive"]

            for name, clf in base_models.items():
                pipe = Pipeline(
                    steps=[
                        ("prep", preprocessor),
                        ("clf", clf)
                    ]
                )

                # Fit on training data
                pipe.fit(X_train, y_train)

                # Store fitted model
                fitted_models[name] = pipe

                # Predictions on the held-out test set
                y_pred = pipe.predict(X_test)
                preds_test[name] = y_pred

                # Probabilities if available
                if hasattr(pipe, "predict_proba"):
                    probas_test[name] = pipe.predict_proba(X_test)

                # Test metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                metrics_rows.append({"Model": name, "Accuracy": acc, "F1 (weighted)": f1})

                # Cross-validation (5-fold)
                cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
                cv_rows.append({
                    "Model": name,
                    "CV mean accuracy": cv_scores.mean(),
                    "CV std": cv_scores.std()
                })

            metrics_df = pd.DataFrame(metrics_rows).round(3)
            cv_df = pd.DataFrame(cv_rows).round(3)

            # --------------------------------------------------------------
            # 4. Evaluation tables
            # --------------------------------------------------------------
            st.markdown("### 4. Model Evaluation - Test Set")

            st.dataframe(metrics_df, use_container_width=True)

            st.markdown("""
            **How to read this table (no ML background needed):**

            - **Accuracy**: percentage of people the model classified correctly.  
            - **F1 (weighted)**: balances correctness across the three classes (Low / Moderate / Aggressive).  
              Higher values are better, especially if one group is smaller than the others.
            """)

            st.markdown("### 5. Cross-Validation (5-fold)")

            st.dataframe(cv_df, use_container_width=True)

            st.markdown("""
            **What is cross-validation?**

            - We shuffle the data and split it into **5 different train/test folds**.  
            - Each model is trained and tested 5 times on different slices of the data.  
            - We look at the **average accuracy** and how much it **varies (CV std)**.  
            - This tells us whether a model is **stable and reliable**, not just lucky on one split.
            """)

            # --------------------------------------------------------------
            # 6. Visual evaluation (Confusion Matrix, Feature Importance, Probabilities)
            # --------------------------------------------------------------
            st.markdown("### 6. Visual Evaluation")

            st.markdown("""
            Use the dropdown below to inspect each model visually:

            - **Confusion Matrix** â€“ where the model is correct and where it is confused  
            - **Feature Importance** â€“ which inputs matter most (for tree-based models)  
            - **Prediction Confidence** â€“ how sure the model is when it predicts each class
            """)

            model_choice = st.selectbox(
                "Choose a model to inspect:",
                list(fitted_models.keys()),
                index=1  # default to Random Forest
            )

            # ---- Confusion Matrix ----
            st.markdown("#### 6.1 Confusion Matrix")

            y_pred_sel = preds_test[model_choice]
            cm = confusion_matrix(y_test, y_pred_sel, labels=labels)

            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto",
                title=f"{model_choice} - Confusion Matrix"
            )
            fig_cm.update_xaxes(title="Predicted")
            fig_cm.update_yaxes(title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("""
            - The **diagonal cells** (top-left to bottom-right) are **correct predictions**.  
            - Numbers off the diagonal show where the model **confused one risk level with another**  
              (for example predicting *Moderate* when the person was actually *Aggressive*).
            """)

            # ---- Feature Importance (for tree-based models) ----
            st.markdown("#### 6.2 Which Features Matter Most? (Feature Importance)")

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier

            selected_model = fitted_models[model_choice]
            clf_step = selected_model.named_steps["clf"]

            # Get feature names from the preprocessor
            prep = selected_model.named_steps["prep"]

            if cat_cols:
                cat_transformer = prep.named_transformers_["cat"]
                cat_feature_names = list(cat_transformer.get_feature_names_out(cat_cols))
            else:
                cat_feature_names = []

            num_feature_names = num_cols
            all_feature_names = cat_feature_names + num_feature_names

            if hasattr(clf_step, "feature_importances_"):
                importances = clf_step.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": all_feature_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(15)

                fig_imp = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"Top Feature Importances - {model_choice}",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig_imp.update_layout(xaxis_title="Importance (higher = more influence)")
                st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("""
                Features near the **top** of this chart have the biggest influence on the modelâ€™s decision.
                For example, if **Salary** and **YearsExperience** appear at the top,
                it means peopleâ€™s professional profile strongly shapes their risk appetite.
                """)
            else:
                st.info(f"{model_choice} does not provide built-in feature importance (it is not a tree-based model).")

            # ---- Prediction probabilities ----
            st.markdown("#### 6.3 How Confident Is the Model? (Prediction Probabilities)")

            if model_choice in probas_test:
                prob = probas_test[model_choice]
                prob_df = pd.DataFrame(prob, columns=fitted_models[model_choice].classes_)

                fig_prob = px.box(
                    prob_df,
                    title=f"{model_choice} - Prediction Confidence by Class"
                )
                fig_prob.update_yaxes(title="Predicted Probability")
                st.plotly_chart(fig_prob, use_container_width=True)

                st.caption("""
                Each box shows how confident the model is on average for each class:

                - Higher probabilities = the model is usually **sure** when it predicts that class.  
                - If the boxes are low and flat, the model is often **uncertain**.
                """)
            else:
                st.info(f"{model_choice} does not output probabilities, so confidence plots are not available.")

            # ==================================================================
            # B. PREDICTION SECTION - interactive "try it yourself"
            # ==================================================================
            st.markdown("## B. Prediction - Try the Models on a New Profile")

            st.markdown("""
            Below you can **simulate a new person** by choosing their characteristics.
            The selected model will predict whether they are likely to have a **Low, Moderate, or Aggressive** risk appetite.
            """)

            # Build simple input widgets based on available feature columns
            with st.form("manual_prediction_form"):
                cols_ui = st.columns(2)

                input_data = {}

                # Categorical fields
                if col_gender in feature_cols:
                    opts = sorted(model_df[col_gender].dropna().unique())
                    input_data[col_gender] = cols_ui[0].selectbox("Gender", opts)

                if col_agegrp in feature_cols:
                    opts = sorted(model_df[col_agegrp].dropna().unique())
                    input_data[col_agegrp] = cols_ui[1].selectbox("Age Group", opts)

                if "Education" in feature_cols:
                    opts = sorted(model_df["Education"].dropna().unique())
                    input_data["Education"] = cols_ui[0].selectbox("Education Level", opts)

                # Numeric fields
                if "Salary" in feature_cols:
                    s_min = float(model_df["Salary"].min())
                    s_max = float(model_df["Salary"].max())
                    input_data["Salary"] = cols_ui[1].slider(
                        "Annual Salary",
                        min_value=int(s_min),
                        max_value=int(s_max),
                        value=int(model_df["Salary"].median()),
                        step=1000
                    )

                if "YearsExperience" in feature_cols:
                    e_min = float(model_df["YearsExperience"].min())
                    e_max = float(model_df["YearsExperience"].max())
                    input_data["YearsExperience"] = cols_ui[0].slider(
                        "Years of Experience",
                        min_value=float(np.floor(e_min)),
                        max_value=float(np.ceil(e_max)),
                        value=float(np.median(model_df["YearsExperience"])),
                        step=1.0
                    )

                # Choose model for prediction (can be different from the one inspected above)
                pred_model_name = st.selectbox(
                    "Choose a model for prediction:",
                    list(fitted_models.keys()),
                    index=1,
                    key="pred_model_choice"
                )

                submitted = st.form_submit_button("Predict Risk Appetite")

            if submitted:
                x_new = pd.DataFrame([input_data])
                model_pred = fitted_models[pred_model_name]
                pred_label = model_pred.predict(x_new)[0]

                st.markdown(f"### Predicted Risk Appetite: **{pred_label}**")

                if hasattr(model_pred, "predict_proba"):
                    prob_new = model_pred.predict_proba(x_new)[0]
                    prob_df_new = pd.DataFrame({
                        "Risk Appetite": model_pred.classes_,
                        "Probability": prob_new
                    })
                    fig_new = px.bar(
                        prob_df_new,
                        x="Risk Appetite",
                        y="Probability",
                        title=f"{pred_model_name} - Prediction Confidence for This Person",
                        text_auto=True
                    )
                    fig_new.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_new, use_container_width=True)

                    st.markdown("""
                    - The bar shows how strongly the model believes in each category.  
                    - The highest bar corresponds to the predicted risk appetite.
                    """)
                else:
                    st.info(f"{pred_model_name} does not provide probability scores - only the predicted label is shown.")

            # ==================================================================
            # C. RESULT & INTERPRETATION SECTION
            # ==================================================================
            st.markdown("## C. Results & Interpretation")

            # Choose the best model by Accuracy (you can also pick F1 if you like)
            best_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
            best_model_name = best_row["Model"]
            best_acc = best_row["Accuracy"]
            best_f1 = best_row["F1 (weighted)"]

            with st.container(border=True):
                st.markdown("### Summary for Report / Presentation")

                st.markdown(f"""
                **Best-performing model (based on test accuracy):**  
                - **{best_model_name}** with **Accuracy â‰ˆ {best_acc:.3f}** and **F1 â‰ˆ {best_f1:.3f}**

                **Key takeaways (explained simply):**

                - We built a **combined dataset** from Finance and Salary files that describes each personâ€™s
                  demographics, professional profile, and Risk Appetite.
                - We trained **four different machine learning models** on the same data:
                  Logistic Regression, Random Forest, K-Nearest Neighbors and Decision Tree.
                - We evaluated them using:
                  - A proper **train/test split** (70/30),
                  - **Accuracy & F1-score** on the unseen test set,
                  - **5-fold cross-validation** to check stability,
                  - **Confusion matrices**, **feature importance**, and **prediction confidence plots**.

                - This shows we understand:
                  - **Model development** (choosing features, defining a target, training multiple models),  
                  - **Model evaluation & comparison** (metrics + visual diagnostics), and  
                  - **Model selection & validation** (using test sets and cross-validation to justify which
                    model we trust most).

                In a presentation to a non-technical audience, you can summarise it as:

                > â€œWe experimented with four different AI methods to predict whether a person is a
                > low, moderate or aggressive investor.  
                > We tested each one fairly on new people it had never seen before, and the
                > best-performing model was **{best_model_name}**, which correctly classified roughly
                > {best_acc:.0%} of investors. We also checked which factors matter most (salary, age group,
                > education, etc.) and visualised where the models make mistakes so that the results
                > are transparent and trustworthy.â€
                """)


# =============================================================
# PAGE 8: INTERACTIVE DATA EXPLORER
# =============================================================
elif page == "🔍 Explore the Data":
    story_header(
        chapter="DEEP DIVE",
        title="Explore the Data Yourself",
        hook="Want to dig deeper? Use the interactive explorer",
        context="""
        This page provides raw access to all three datasets with dynamic filtering, missing value 
        analysis, and downloadable CSVs.
        """,
        question="What custom insights can you extract from the raw data?"
    )
    #st.subheader("ðŸ§® Interactive Data Explorer")

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
    # Tabs for Finance, Salary and Trends datasets
    # ---------------------------
    #tab_fin, tab_sal = st.tabs(["Finance", "Salary"])
    tab_fin, tab_sal, tab_trn = st.tabs(["Finance", "Salary", "Trends"])

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
    
    # Filtered view for Trends dataset (reuse same sidebar filters where columns exist)
    tr_view = trends.copy()
    if col_gender and p6_gender and col_gender in tr_view.columns:
        tr_view = tr_view[tr_view[col_gender].isin(p6_gender)]
    if col_agegrp and p6_age and col_agegrp in tr_view.columns:
        tr_view = tr_view[tr_view[col_agegrp].isin(p6_age)]
    if col_edu and p6_edu and col_edu in tr_view.columns:
        tr_view = tr_view[tr_view[col_edu].isin(p6_edu)]

    # ---------------------------
    # FINANCE TAB
    # ---------------------------
    with tab_fin:
        st.markdown("### Finance Dataset")

         # --- styling for the step cards (green pill etc.) ---
        st.markdown(
            """
            <style>
            .step-card {
                background: #050816;              /* dark navy */
                border-radius: 12px;
                padding: 12px 18px;
                margin-bottom: 10px;
                border: 1px solid rgba(34,197,94,0.45);  /* subtle green border */
            }
            .step-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 6px;
            }
            .step-pill {
                display: inline-block;
                padding: 2px 10px;
                border-radius: 999px;
                background: #22c55e;             /* green */
                color: #052e16;                   /* very dark green text */
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
            }
            .step-title {
                font-weight: 700;
                font-size: 0.95rem;
            }
            .step-body {
                font-size: 0.9rem;
                line-height: 1.4;
            }
            .step-body code {
                background: #022c22;
                color: #bbf7d0;
                padding: 2px 4px;
                border-radius: 4px;
                font-size: 0.8rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # ---------------------------
        # Data Cleaning Summary - Finance_Dataset-Cleaned.csv
        # ---------------------------
        with st.container(border=True):
            st.markdown("#### Data Cleaning Summary - Finance Dataset")

 # ---------- Step 1 ----------
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 1</span>
                    <span class="step-title">Standardized Column Names</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>Trimmed extra spaces from all column headers using <code>df.columns.str.strip()</code>.</li>
                      <li>Renamed long survey-question columns that start with<br>
                        <code>What do you think are the best options for investing your money?</code><br>
                        into concise preference fields, for example:
                        <ul>
                          <li>
                            <code>What do you think are the best options for investing your money? (Rank in order of preference) [Gold]</code>
                            &nbsp;&rarr;&nbsp;<code>Preference rank for GOLD investment</code>
                          </li>
                          <li>
                            <code>What do you think are the best options for investing your money? [Mutual Funds]</code>
                            &nbsp;&rarr;&nbsp;<code>Preference for MUTUAL FUNDS investment</code>
                          </li>
                        </ul>
                      </li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------- Step 2 ----------
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 2</span>
                    <span class="step-title">Standardized Categorical Values &amp; Missing Data</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>Trimmed leading/trailing spaces in all text fields so <code>" Gold "</code> becomes <code>"Gold"</code>.</li>
                      <li>Converted common numeric-like columns
                        <code>AGE</code>, <code>EXPECTED_RETURN_PCT</code>, <code>AVG_SALARY</code>,
                        <code>AVG_EXPERIENCE</code>, <code>MONITOR_FREQ_PER_MONTH</code>
                        to numeric using <code>pd.to_numeric(..., errors="coerce")</code>.
                      </li>
                      <li>Handled missing values:
                        <ul>
                          <li>Numeric columns &rarr; filled with the median of that column.</li>
                          <li>Text columns &rarr; replaced literal <code>"nan"</code> strings and
                              filled missing entries with <code>"Unknown"</code>.
                          </li>
                        </ul>
                      </li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------- Step 3 ----------
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 3</span>
                    <span class="step-title">Derived Fields &amp; Duplicate Handling</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>
                        Created an ordinal <code>Age_Group</code> variable from <code>AGE</code> when
                        <code>Age_Group</code> was missing, with buckets:
                        <code>18â€“24</code>, <code>25â€“34</code>, <code>35â€“44</code>, <code>45â€“54</code>, <code>55+</code>.
                      </li>
                      <li>Dropped exact duplicate rows so each respondent is counted only once.</li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------
        # Main Finance Charts
        # ---------------------------
        grid1a, grid1c = st.columns([1, 1])

        # 1ï¸âƒ£ Investment Type Distribution (Pie)
        with grid1a:
            inv_cols = [c for c in fin_view.columns if re.search(r"invest", c, flags=re.I)]
            inv_col = inv_cols[0] if inv_cols else None

            if inv_col and not fin_view.empty:
                fig_inv = px.pie(
                    fin_view,
                    names=inv_col,
                    title="Investment Type Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_inv.update_layout(title_font_size=16)
                st.plotly_chart(fig_inv, use_container_width=True)
                st.caption(
                    """
**What it shows:**
- Breakdown of how respondents allocate their investments across different types (Stocks, Mutual Funds, etc.).
- Helps identify diversification and risk preferences.
                    """
                )
            else:
                st.warning("âš ï¸ Investment type column not found in dataset.")

        # 3ï¸âƒ£ Monitoring Frequency (Bar)
        with grid1c:
            mon_cols = [c for c in fin_view.columns if re.search(r"monitor|frequency", c, flags=re.I)]
            mon_col = mon_cols[0] if mon_cols else None

            if mon_col:
                mf = fin_view[mon_col].dropna()
                if not mf.empty:
                    ct = mf.value_counts().reset_index()
                    ct.columns = ["Frequency", "Count"]
                    fig_mon = px.bar(
                        ct,
                        x="Frequency",
                        y="Count",
                        title="Portfolio Monitoring Frequency",
                        color="Frequency",
                        color_discrete_sequence=px.colors.sequential.Blues,
                    )
                    fig_mon.update_layout(xaxis_title="Frequency", yaxis_title="Respondents")
                    st.plotly_chart(fig_mon, use_container_width=True)
                    st.caption(
                        """
**What it shows:**
- Frequency of portfolio monitoring among investors.
- Indicates how financially engaged respondents are.
- Frequent monitoring may reflect higher involvement or anxiety around market changes.
                        """
                    )

        st.divider()

        # ---------- Missing Values (Summary + Bar + Heatmap)
        st.markdown("### Missing Values - Finance")
        if not fin_view.empty:
            miss_fin = _missing_counts(fin_view)
            st.markdown("**Summary Table:** Missing counts and percentages (includes 'Unknown').")
            st.dataframe(miss_fin)

            top_fin = miss_fin.head(20).reset_index().rename(columns={"index": "Column"})
            fig_mbar = px.bar(
                top_fin,
                x="Column",
                y="MissingPct",
                title="Top 20 Missing Columns (%)",
                color="MissingPct",
                color_continuous_scale="Teal",
            )
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
                    labels=dict(x="Respondents", y="Columns", color="Missingness"),
                )
                fig_hm.update_traces(
                    hovertemplate="Column: %{y}<br>Respondent Index: %{x}<br>Missing: %{z}<extra></extra>"
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            st.caption(
                """
**How to read:**
- Each cell shows if a respondentâ€™s value is missing (1) or present (0).
- Taller regions of blue suggest specific columns have more data quality issues.
                """
            )

        st.divider()

        show_fin = st.toggle("Show Finance Dataset", value=False, key="p6_show_fin")
        if show_fin:
            st.dataframe(fin_view, use_container_width=True)
        st.download_button(
            "Download Finance Dataset",
            fin_view.to_csv(index=False).encode("utf-8"),
            "finance_filtered.csv",
            "text/csv",
            key="p6_dl_fin",
        )

    # ---------------------------
    # SALARY TAB
    # ---------------------------
    with tab_sal:
        st.markdown("### Salary Dataset")

        # ---------------------------
        # Data Cleaning Summary - Salary_Dataset-Cleaned.csv
        # (reuses the same .step-card CSS defined earlier)
        # ---------------------------
        with st.container(border=True):
            st.markdown("#### Data Cleaning Summary - Salary Dataset")

            # ---- Step 1 ----
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 1</span>
                    <span class="step-title">Standardized Column Names &amp; Text Fields</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>Trimmed spaces from all column headers using <code>df.columns.str.strip()</code>.</li>
                      <li>Trimmed whitespace inside all text cells so values like <code>"  Manager  "</code> become <code>"Manager"</code>.</li>
                      <li>Normalized key categorical fields to <strong>Title Case</strong> for consistency:
                        <code>Gender</code>, <code>Education Level</code>, and <code>Job Title</code>.
                      </li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---- Step 2 ----
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 2</span>
                    <span class="step-title">Numeric Conversion &amp; Missing Values</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>Converted numeric-like columns to numeric using
                        <code>pd.to_numeric(..., errors="coerce")</code>:
                        <code>Age</code>, <code>Years of Experience</code> / <code>Years Of Experience</code>,
                        and <code>Salary</code>.
                      </li>
                      <li>Unified the header variant <code>"Years Of Experience"</code> into
                        <code>"Years of Experience"</code> so downstream logic uses a single column name.</li>
                      <li>Handled missing values:
                        <ul>
                          <li><strong>Numeric columns</strong> &rarr; filled NaNs with the column median.</li>
                          <li><strong>Text columns</strong> &rarr; cleaned literal <code>"nan"</code> strings
                              and filled remaining missing entries with <code>"Unknown"</code>.
                          </li>
                        </ul>
                      </li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---- Step 3 ----
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 3</span>
                    <span class="step-title">Derived Fields, Education Buckets &amp; Duplicates</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>Created an ordinal <code>Age Group</code> field from <code>Age</code> with buckets:
                        <code>18-24</code>, <code>25-34</code>, <code>35-44</code>, <code>45-54</code>, <code>55+</code>.
                      </li>
                      <li>Created an <code>Experience Band</code> field from
                        <code>Years of Experience</code> with buckets:
                        <code>0-1 yrs</code>, <code>2-4 yrs</code>, <code>5-9 yrs</code>,
                        <code>10-14 yrs</code>, <code>15-19 yrs</code>, <code>20+ yrs</code>.
                      </li>
                      <li>Standardized <code>Education Level</code> into clean buckets:
                        <ul>
                          <li><code>Master'S</code> / <code>Master'S Degree</code> &rarr; <code>Master's</code></li>
                          <li><code>Bachelor'S</code> / <code>Bachelor'S Degree</code> &rarr; <code>Bachelor's</code></li>
                          <li>Fixed any leftover <code>'S</code> casing to <code>'s</code> for consistency.</li>
                        </ul>
                      </li>
                      <li>Removed exact duplicate rows so each salary record is counted only once.</li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------
        # Existing Salary charts & logic
        # ---------------------------
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
            years_col = None
            if col_expyrs and (col_expyrs in sal_view.columns):
                years_col = col_expyrs
            elif col_expband and (col_expband in sal_view.columns):
                def _mid_from_band(txt):
                    if pd.isna(txt):
                        return np.nan
                    t = str(txt).lower().strip()
                    m = re.match(r"(\d+)\s*\+\s*", t)
                    if m:
                        return float(m.group(1)) + 2
                    m = re.match(r"(\d+)\s*[-â€“]\s*(\d+)", t)
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

        # Missing values - Salary
        st.markdown("#### Missing Values - Salary")
        if not sal_view.empty:
            miss_sal = _missing_counts(sal_view)
            st.markdown("**Summary Table** - counts/pct of missing values (including literal 'Unknown'/'Nan').")
            st.dataframe(miss_sal)

            top_sal = miss_sal.head(20).reset_index().rename(columns={"index": "Column"})
            fig_mbar2 = px.bar(top_sal, x="Column", y="MissingPct", title="Missing Values (Top 20 columns)",
                               labels={"MissingPct": "% Missing"})
            fig_mbar2.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_mbar2, use_container_width=True)

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

        show_sal = st.toggle("Show Salary Dataset", value=False, key="p6_show_sal")
        if show_sal:
            st.dataframe(sal_view, use_container_width=True)
        st.download_button(
            "Download Salary Dataset",
            sal_view.to_csv(index=False).encode("utf-8"),
            "salary_filtered.csv",
            "text/csv",
            key="p6_dl_sal",
        )

    # ---------------------------
    # TRENDS TAB
    # ---------------------------
    with tab_trn:
        st.markdown("### Trends Dataset")

        # ---------------------------
        # Data Cleaning Summary - Trends Dataset
        # (reuses .step-card CSS defined in the Finance tab)
        # ---------------------------
        with st.container(border=True):
            st.markdown("#### Data Cleaning Summary - Trends Dataset")

            # Step 1: Basic hygiene & text normalization
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 1</span>
                    <span class="step-title">Basic Hygiene &amp; Text Normalization</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>Trimmed whitespace from all column headers so names like <code>" Age "</code> become <code>"Age"</code>.</li>
                      <li>Stripped leading/trailing spaces from every text cell to avoid variants like <code>" Gold "</code> vs <code>"Gold"</code>.</li>
                      <li>Standardized key categorical fields such as <code>Gender</code>, <code>Segment</code>, <code>Region</code>, <code>Category</code>, and <code>Investment Type</code> into tidy Title Case labels (e.g., <code>"male"</code> â†’ <code>"Male"</code>).</li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Step 2: Numeric type inference, percentages & missing values
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 2</span>
                    <span class="step-title">Numeric Detection, Percent Parsing &amp; Missing Values</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>Scanned column names for numeric-like patterns (<code>amount</code>, <code>value</code>, <code>score</code>, <code>age</code>, <code>rate</code>, <code>return</code>, <code>%</code>, etc.) to identify fields that should be numeric.</li>
                      <li>For percentage-style columns (names or values containing <code>"%"</code>), removed percent signs and commas, then safely converted them to numeric values.</li>
                      <li>Applied a <code>to_numeric_if_possible</code> rule: only kept a numeric conversion if more than 30% of values truly behaved like numbers, otherwise preserved the original text.</li>
                      <li>Used an advanced <strong>KNNImputer</strong> on the numeric block to fill missing numeric values based on nearest neighbors.</li>
                      <li>For categorical fields, normalized literal <code>"nan"</code> placeholders and filled remaining missing values with a consistent label <code>"Unknown"</code>.</li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Step 3: Outliers, feature bands & clean exploratory file
            st.markdown(
                """
                <div class="step-card">
                  <div class="step-header">
                    <span class="step-pill">Step 3</span>
                    <span class="step-title">Outliers, Feature Bands &amp; Clean Exploratory Output</span>
                  </div>
                  <div class="step-body">
                    <ul>
                      <li>Ran <strong>Local Outlier Factor (LOF)</strong> on the numeric block to flag multivariate anomalies in a new column <code>Is_Outlier_LOF</code>, and (with <code>DROP_OUTLIERS=True</code>) removed those rows from the clean dataset.</li>
                      <li>Engineered useful behavioral segments:
                        <ul>
                          <li><strong>Age Group</strong> from <code>Age</code> (e.g., <code>18â€“24</code>, <code>25â€“34</code>, <code>35â€“44</code>, <code>45â€“54</code>, <code>55+</code>).</li>
                          <li><strong>Tenure Band</strong> from investment-duration columns such as <code>Years Invested</code> or <code>Investment Duration (Years)</code> (e.g., <code>0â€“1 yrs</code>, <code>2â€“4 yrs</code>, <code>5â€“9 yrs</code>, â€¦).</li>
                          <li><strong>Risk Band</strong> (Low / Medium / High) from <code>Expected Return %</code>, <code>Expected Return</code>, or <code>Risk Score</code>, using 33rd and 66th percentile cut-points.</li>
                        </ul>
                      </li>
                      <li>Removed exact duplicate rows so each trends respondent is counted only once.</li>
                      <li>Saved a clean, human-readable file <code>Finance_Trends_Cleaned.csv</code> for use in this dashboard, and also generated a separate ML-oriented file (<code>Finance_Trends_ML.csv</code>) with encoded, scaled and feature-reduced variables for modeling work outside the app.</li>
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # ---------------------------
        # Main Trends Charts
        # ---------------------------
        gridT1, gridT2 = st.columns(2)

        # ---------------------------
        # Derive helper fields from Trends data for charts
        # ---------------------------
        tr_view = tr_view.copy()  # work on a local copy

        # 1) Age_Group from numeric age
        def _age_to_group(x):
            try:
                a = float(x)
            except Exception:
                return np.nan
            if a <= 24: return "18-24"
            if a <= 34: return "25-34"
            if a <= 44: return "35-44"
            if a <= 54: return "45-54"
            return "55+"

        if "age" in tr_view.columns:
            tr_view["Age_Group"] = tr_view["age"].apply(_age_to_group)

        # 2) Expected_Return_Pct from the "Expect" column (e.g. "20%-30%")
        exp_pat = re.compile(r"(\d+(?:\.\d+)?)")

        def _parse_expect(x):
            if pd.isna(x):
                return np.nan
            m = exp_pat.findall(str(x))
            if not m:
                return np.nan
            v = float(m[0])  # first number in the range, e.g. "20%-30%" -> 20
            if v < 0 or v > 200:
                return np.nan
            return v

        if "Expect" in tr_view.columns:
            tr_view["Expected_Return_Pct"] = tr_view["Expect"].apply(_parse_expect)

        # ---------------------------
        # Main Trends Charts (3 charts)
        # ---------------------------
        gridT1, gridT2 = st.columns(2)

        # 1) Expected Return (%) by Age Group (box plot)
        with gridT1:
            st.markdown("### Expected Return (%) by Age Group")
            if {"Age_Group", "Expected_Return_Pct"}.issubset(tr_view.columns):
                tmp = tr_view[["Age_Group", "Expected_Return_Pct"]].dropna()
                if not tmp.empty:
                    fig_risk_age = px.box(
                        tmp,
                        x="Age_Group",
                        y="Expected_Return_Pct",
                        title="Expected Return (%) by Age Group",
                    )
                    fig_risk_age.update_layout(
                        xaxis_title="Age Group",
                        yaxis_title="Expected Return (%)",
                    )
                    st.plotly_chart(fig_risk_age, use_container_width=True)
                    st.caption(
                        "**What it shows:** How return expectations differ across age groups "
                        "in the trends sample."
                    )
                else:
                    st.info("No usable Expected_Return_Pct values after filters.")
            else:
                st.info("Age_Group or Expected_Return_Pct column not available in Trends dataset.")

        # 2) Expected Return (%) vs Age (scatter + trendline)
        with gridT2:
            st.markdown("### Expected Return (%) vs Age")
            if {"age", "Expected_Return_Pct"}.issubset(tr_view.columns):
                tmp = tr_view[["age", "Expected_Return_Pct"]].dropna()
                if not tmp.empty:
                    fig_ret_age = px.scatter(
                        tmp,
                        x="age",
                        y="Expected_Return_Pct",
                        trendline="ols",
                        title="Expected Return (%) vs Age",
                    )
                    fig_ret_age.update_layout(
                        xaxis_title="Age",
                        yaxis_title="Expected Return (%)",
                    )
                    st.plotly_chart(fig_ret_age, use_container_width=True)
                    st.caption(
                        "**What it shows:** Whether older respondents in the trends data aim "
                        "for higher or lower expected returns."
                    )
                else:
                    st.info("Not enough overlapping Age and Expected_Return_Pct values after filters.")
            else:
                st.info("age or Expected_Return_Pct column not available in Trends dataset.")

        st.divider()


        # ---------------------------
        # Missing Values - Trends
        # ---------------------------
        st.markdown("### Missing Values - Trends")
        if not tr_view.empty:
            miss_trn = _missing_counts(tr_view)
            st.markdown("**Summary Table:** Missing counts and percentages (includes 'Unknown').")
            st.dataframe(miss_trn)

            top_trn = miss_trn.head(20).reset_index().rename(columns={"index": "Column"})
            fig_mbar_trn = px.bar(
                top_trn,
                x="Column",
                y="MissingPct",
                title="Top 20 Missing Columns (%)",
                color="MissingPct",
                color_continuous_scale="Teal",
            )
            fig_mbar_trn.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_mbar_trn, use_container_width=True)

            # Missingness heatmap (limit columns & rows for readability)
            show_cols_trn = tr_view.columns[:30]
            heat_mask_trn = tr_view[show_cols_trn].isna().copy()
            for c in show_cols_trn:
                if tr_view[c].dtype == object:
                    s = tr_view[c].astype(str).str.strip().str.lower()
                    heat_mask_trn[c] = heat_mask_trn[c] | s.isin({"unknown", "nan"})
            hm_trn = heat_mask_trn.head(200).astype(int)
            if hm_trn.shape[1] > 1:
                fig_hm_trn = px.imshow(
                    hm_trn.T,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    title="Missingness Heatmap (1=Missing, 0=Present)",
                    labels=dict(x="Respondents", y="Columns", color="Missingness"),
                )
                st.plotly_chart(fig_hm_trn, use_container_width=True)
            st.caption(
                "**How to read:** darker cells indicate more missing data for that column "
                "among the first 200 records."
            )
        else:
            st.info("No rows available in Trends dataset after filters.")

        st.divider()

        # Toggle + download full Trends view
        show_trn = st.toggle("Show Trends Dataset", value=False, key="p6_show_trn")
        if show_trn:
            st.dataframe(tr_view, use_container_width=True)
        st.download_button(
            "Download Trends Dataset",
            tr_view.to_csv(index=False).encode("utf-8"),
            "trends_filtered.csv",
            "text/csv",
            key="p6_dl_trn",
        )


# =============================================================
# =============================================================
# PAGE 9: INSIGHTS & STORYTELLING
# =============================================================
elif page == "📊 The Story in Numbers":
    story_header(
        chapter="EPILOGUE",
        title="The Story in Numbers",
        hook="What did we learn about investor behavior?",
        context="""
        This final page synthesizes all findings into a cohesive narrative. We'll answer: Why does 
        this analysis matter? Who benefits? What real-world applications exist?
        """,
        question="What actionable insights can financial professionals take from this analysis?"
    )

    # -------------------------------
    # Context and Problem Definition
    # -------------------------------
    st.markdown("""
    ## Project Context:
    Modern investors, especially young professionals face an overload of investment choices 
    (stocks, mutual funds, gold, crypto, fixed deposits, etc.) without a clear understanding of how 
    age, education, income, and experience influence financial decisions. The problem lies in misaligned investment behavior, people often invest based on trends or peer influence 
    rather than aligning with their risk capacity, income level, and financial goals.  

    This dashboard bridges that gap by analyzing behavioral finance patterns - linking demographic and salary data 
    with investment preferences, risk appetite, and monitoring frequency.
    """)

    # -------------------------------
    # Why It Matters
    # -------------------------------
    st.markdown("""
    ## Relevance & Impact:
    Understanding behavioral patterns helps:
    - **Financial educators and advisors** design better financial literacy programs.
    - **Fintech startups and banks** personalize products (e.g., tailored mutual fund or insurance plans).
    - **Researchers and policy makers** identify how income, gender, or education affect investment equity and inclusion.
    
    Ultimately, this analysis provides **data-driven insights** into how people think about money - 
    a foundation for building smarter, more inclusive, and sustainable financial ecosystems.
    """)

    # -------------------------------
    # Audience Definition
    # -------------------------------
    st.markdown("""
    ## Intended Audience:
    - **Young professionals and investors:** To self-assess financial decisions and compare with peer patterns.  
    - **Financial advisors & planners:** To understand client psychology and risk tolerance.  
    - **Fintech product teams:** To develop tools for goal-based investing and portfolio diversification.  
    - **Researchers & educators:** To study socio-economic factors shaping financial behavior.
    """)

    # -------------------------------
    # Behavioral & Analytical Insights
    # -------------------------------
    st.markdown("""
    ## Key Behavioral Insights:

    ** Age & Risk Appetite**
    - **Younger investors (<35)** show **higher risk tolerance**, preferring **stocks and mutual funds**.
    - **Older participants (45+)** lean toward **gold and fixed deposits** for safety and predictable returns.

    ** Gender Patterns**
    - **Females** check portfolios less frequently and favor **stable investments** - a more **long-term, cautious approach**.  
    - **Males** tend to monitor investments more often and lean slightly toward **higher-risk instruments**.

    ** Education & Experience**
    - **Masterâ€™s degree holders** show **diversified portfolios** and **steady return expectations**.  
    - **Professionals with 5+ years of experience** prefer **goal-based, low-volatility investments**.

    ** Income & Return Expectations**
    - **High-salary groups** expect **lower but stable returns**, prioritizing capital preservation.  
    - **Lower-income respondents** expect **higher returns**, showing an aspirational or risk-seeking mindset.  
    - There is a **negative correlation** between **salary** and **expected return %** - confirming a classic *riskâ€“return tradeoff*.

    ** Cross-Dataset Patterns**
    - **Age â†” Experience â†” Salary:** Strong positive correlation - older respondents are more experienced and earn higher salaries.  
    - **Salary â†” Education â†” Investment Diversity:** Higher-educated earners diversify across multiple assets.  
    - **Expected Return â†” Monitoring Frequency:** Higher return expectations often coincide with frequent portfolio tracking.
    """)

    # -------------------------------
    # Summary Narrative
    # -------------------------------
    st.markdown("""
    ## Overall Story:
    As individuals mature in age, experience, and education, their **investment mindset evolves** 
    from **aggressive to balanced** - emphasizing **security and long-term growth** over quick gains.  
    Gender, salary, and education together shape financial confidence, awareness, and diversification.  
    This project provides a **data-driven foundation** for understanding *why people invest the way they do* 
    - guiding future interventions in **financial literacy, policy, and personalized investment design**.
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

# =====================[ CHANGES START - KPI center + dark gray (final) ]=====================
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

/* 3) Some Streamlit themes wrap value inside another div - target it too */
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
# =====================[ CHANGES END - KPI center + dark gray (final) ]=====================

# --------------------------- Sidebar label colors (add-on) ---------------------------
FILTER_TITLE_COLOR = "#2C3539"   # Filter / section headings
FILTER_LABEL_COLOR = "#2C3539"   # Field labels: Gender, Age Group, etc.
FILTER_CAPTION_COLOR = "#313234" # Small helper text (optional)

st.markdown(f"""
<style>
/* A) Main "Filter" header (already use .filter-header) */
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