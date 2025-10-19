import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Investment Behavior Analysis", page_icon="📊", layout="wide")

st.title("📊 Investment Behavior Analysis")
st.write(
    """
    Welcome! This dashboard explores how people choose where to invest (bank, gold, stocks, mutual funds, etc.).
    Use the sidebar to filter, and the charts will update.
    """
)

# Sidebar filters (placeholder for your dataset)
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Age range", 18, 80, (25, 45))
gender = st.sidebar.multiselect("Gender", ["Male", "Female", "Other"], default=["Male", "Female"])
goal = st.sidebar.multiselect("Primary Goal", ["Wealth Growth", "Safety", "Liquidity", "Tax Saving"], default=["Wealth Growth", "Safety"])

# Demo data until you load the Kaggle dataset
np.random.seed(42)
df = pd.DataFrame({
    "age": np.random.randint(18, 80, 400),
    "gender": np.random.choice(["Male", "Female", "Other"], 400, p=[0.49, 0.49, 0.02]),
    "goal": np.random.choice(["Wealth Growth", "Safety", "Liquidity", "Tax Saving"], 400),
    "instrument": np.random.choice(["Bank", "Gold", "Stocks", "Mutual Funds", "Crypto", "Bonds"], 400),
    "amount": np.random.gamma(3, 500, 400).round(2),
})

# Apply filters
mask = (
    (df["age"].between(age_range[0], age_range[1])) &
    (df["gender"].isin(gender)) &
    (df["goal"].isin(goal))
)
f = df[mask]

left, right = st.columns(2)

with left:
    st.subheader("Counts by Instrument")
    counts = f["instrument"].value_counts().reset_index()
    counts.columns = ["instrument", "count"]
    fig_bar = px.bar(counts, x="instrument", y="count")
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Total Amount by Instrument")
    totals = f.groupby("instrument", as_index=False)["amount"].sum()
    fig_tot = px.bar(totals, x="instrument", y="amount")
    st.plotly_chart(fig_tot, use_container_width=True)

st.subheader("Data Preview")
st.dataframe(f.head(20))
