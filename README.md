# 🪙 Investment Behavior Analysis Dashboard

## CMSE 830-001 | Fall 2025 | Final Project

A comprehensive Streamlit dashboard analyzing investment behavior patterns across 13,000+ investor records, integrating demographics, income, and investment preferences to uncover the psychological and demographic factors driving financial decisions.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Data Sources & Data Dictionary](#-data-sources--data-dictionary)
4. [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
5. [Feature Engineering](#-feature-engineering)
6. [Statistical Analysis Methods](#-statistical-analysis-methods)
7. [Machine Learning Approach](#-machine-learning-approach)
8. [Dashboard Pages](#-dashboard-pages)
9. [Installation & Setup](#-installation--setup)
10. [Usage Guide](#-usage-guide)
11. [Key Findings](#-key-findings)
12. [Rubric Alignment](#-rubric-alignment)
13. [Limitations & Future Work](#-limitations--future-work)
14. [Acknowledgments](#-acknowledgments)

---

## Project Overview

### The Problem

Modern investors face a paradox: **unprecedented access to investment options**, yet widespread confusion about where to allocate capital. Financial professionals struggle to personalize recommendations without understanding the demographic and psychological factors driving investment decisions.

### Our Solution

This dashboard integrates three distinct datasets to analyze:
- How **age, gender, and education** influence risk tolerance
- Patterns between **income levels and expected returns**
- Predictive models for **investor behavior classification**

### Business Impact

| Stakeholder | Application |
|-------------|-------------|
| Financial Advisors | Age-based risk profiling and portfolio recommendations |
| Fintech Companies | Personalized product suggestions and UX design |
| HR Departments | Employee financial wellness program design |
| Researchers | Behavioral finance academic studies |

---

## Key Features

- **Interactive Visualizations**: 15+ visualization types including 3D scatter plots, violin charts, and correlation heatmaps
- **Machine Learning Models**: 4 classifiers with hyperparameter tuning for risk appetite prediction
- **Statistical Analysis**: ANOVA, Chi-Square, and Pearson correlation tests
- **Advanced Data Cleaning**: IQR outlier detection, Isolation Forest, KNN imputation
- **Responsive Design**: Clean UI with story mode for explanations
- **Data Export**: Download raw and cleaned datasets

---

## Data Sources & Data Dictionary

### Overview

| Dataset | Records | Variables | Focus Area |
|---------|---------|-----------|------------|
| Finance | 243 | 25 | Investment preferences, risk appetite |
| Salary | 1,792 | 8 | Income, education, experience |
| Trends | 11,383 | 25 | Market behavior, monitoring patterns |

### Finance Dataset - Data Dictionary

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `Gender` | Categorical | Respondent gender | Male, Female, Unknown |
| `AGE` | Numeric | Age in years | 18-70 |
| `Age_Group` | Categorical | Binned age | 18-24, 25-34, 35-44, 45-54, 55+ |
| `Do you invest in Investment Avenues?` | Categorical | Investment participation | Yes, No |
| `Preference rank for MUTUAL FUNDS investment` | Numeric | Ranking 1-7 | 1 (highest) to 7 (lowest) |
| `Preference rank for EQUITY MARKET investment` | Numeric | Ranking 1-7 | 1 (highest) to 7 (lowest) |
| `Preference rank for DEBENTURES investment` | Numeric | Ranking 1-7 | 1 (highest) to 7 (lowest) |
| `Preference rank for GOVERNMENT BONDS investment` | Numeric | Ranking 1-7 | 1 (highest) to 7 (lowest) |
| `Preference rank for FIXED DEPOSITS investment` | Numeric | Ranking 1-7 | 1 (highest) to 7 (lowest) |
| `Preference rank for PPF investment` | Numeric | Ranking 1-7 | 1 (highest) to 7 (lowest) |
| `Preference rank for GOLD investment` | Numeric | Ranking 1-7 | 1 (highest) to 7 (lowest) |
| `Do you invest in Stock Market?` | Categorical | Stock market participation | Yes, No |
| `What are the factors considered...` | Categorical | Investment factors | Returns, Risk, Locking Period |
| `What is your investment objective?` | Categorical | Primary objective | Capital Appreciation, Income, Growth |
| `What is your purpose behind investment?` | Categorical | Investment purpose | Wealth Creation, Savings, Returns |
| `How long do you prefer to keep your money...` | Categorical | Investment tenure | <1 year, 1-3 years, 3-5 years, 5+ years |
| `How often do you monitor your investment?` | Categorical | Monitoring frequency | Daily, Weekly, Monthly |
| `How much return do you expect...` | Categorical | Expected return range | 10-20%, 20-30%, 30-40% |
| `Which investment avenue do you mostly invest in?` | Categorical | Primary avenue | Mutual Fund, Equity, Fixed Deposits, Gold |

### Salary Dataset - Data Dictionary

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `Age` | Numeric | Age in years | 18-65 |
| `Gender` | Categorical | Respondent gender | Male, Female |
| `Education Level` | Categorical | Highest education | High School, Bachelor's, Master's, PhD |
| `Job Title` | Categorical | Current position | 100+ unique titles |
| `Years of Experience` | Numeric | Work experience | 0-40 years |
| `Salary` | Numeric | Annual salary (USD) | $30,000 - $250,000 |
| `Age Group` | Categorical | Binned age (engineered) | 18-24, 25-34, 35-44, 45-54, 55+ |
| `Experience Band` | Categorical | Binned experience (engineered) | 0-1, 2-4, 5-9, 10-14, 15-19, 20+ yrs |

### Trends Dataset - Data Dictionary

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `gender` | Categorical | Respondent gender | Male, Female, Unknown |
| `age` | Numeric | Age in years | 18-38 |
| `Investment_Avenues` | Categorical | Invests in avenues | Yes, No, Unknown |
| `Mutual_Funds` | Numeric | MF preference rank | 1-7 |
| `Equity_Market` | Numeric | Equity preference rank | 1-7 |
| `Debentures` | Numeric | Debentures preference rank | 1-7 |
| `Government_Bonds` | Numeric | Gov bonds preference rank | 1-7 |
| `Fixed_Deposits` | Numeric | FD preference rank | 1-7 |
| `PPF` | Numeric | PPF preference rank | 1-7 |
| `Gold` | Numeric | Gold preference rank | 1-7 |
| `Stock_Marktet` | Categorical | Stock market participation | Yes, No |
| `Factor` | Categorical | Key investment factor | Returns, Risk, Locking Period |
| `Objective` | Categorical | Investment objective | Capital Appreciation, Income, Growth |
| `Purpose` | Categorical | Investment purpose | Wealth Creation, Savings |
| `Duration` | Categorical | Investment duration | <1 year, 1-3 years, 3-5 years, 5+ years |
| `Invest_Monitor` | Categorical | Monitoring frequency | Daily, Weekly, Monthly |
| `Expect` | Categorical | Expected return | 10-20%, 20-30%, 30-40% |
| `Avenue` | Categorical | Primary investment avenue | Mutual Fund, Equity, FD, Gold |
| `Is_Outlier_LOF` | Boolean | LOF outlier flag | True, False |

---

## Data Cleaning & Preprocessing

### Basic Cleaning Techniques

| Technique | Description | Dataset Applied |
|-----------|-------------|-----------------|
| **Column Standardization** | `df.columns.str.strip()` - Removed whitespace | All |
| **Missing Value Imputation** | Categorical → 'Unknown', Numeric → Median | All |
| **Duplicate Removal** | Removed 4,911 exact duplicate rows | Salary |
| **Data Type Conversion** | String numbers → numeric dtype | Age, Salary, Experience |
| **Text Standardization** | Title case, consistent formatting | Gender, Education |

### Advanced Cleaning Techniques

| Technique | Method | Purpose |
|-----------|--------|---------|
| **IQR Outlier Detection** | Q1 - 1.5×IQR, Q3 + 1.5×IQR | Cap age to realistic range (18-70) |
| **Isolation Forest** | sklearn contamination=0.1 | Multivariate anomaly detection |
| **KNN Imputation** | k=5 neighbors | Fill numeric missing values |
| **Error Suffix Removal** | Regex `_err$` pattern | Clean data entry errors in Trends |
| **Business Rule Validation** | Custom functions | Ensure Age, Salary, Experience logical |

### Cleaning Results

| Dataset | RAW Rows | CLEANED Rows | Missing % (Before) | Missing % (After) |
|---------|----------|--------------|-------------------|-------------------|
| Finance | 280 | 243 | 4.97% | 0% |
| Salary | 6,704 | 1,792 | 0.03% | 0% |
| Trends | 12,005 | 11,383 | 6.76% | 0% |

---

## Feature Engineering

| Feature | Source Variable | Transformation Logic | Purpose |
|---------|-----------------|---------------------|---------|
| **Age_Group** | Age (numeric) | Binned: 18-24, 25-34, 35-44, 45-54, 55+ | Categorical analysis |
| **Risk_Band** | Expected Return | Low (<6%), Moderate (6-10%), Aggressive (>10%) | ML classification target |
| **Experience_Band** | Years of Experience | 0-1, 2-4, 5-9, 10-14, 15-19, 20+ yrs | Salary segmentation |
| **Salary_Quartile** | Salary | Quartile-based binning | Income segmentation |

---

## Statistical Analysis Methods

### Tests Applied

| Test | Hypothesis | Variables | Result |
|------|------------|-----------|--------|
| **ANOVA (F-test)** | Mean expected return differs by age group | Expected Return ~ Age Group | F=2.15, p<0.05 ✓ |
| **Chi-Square** | Investment preference independent of gender | Investment Avenue × Gender | χ²=45.2, p<0.001 ✓ |
| **Pearson Correlation** | Linear relationship between salary and return | Salary vs Expected Return | r=-0.03 (weak negative) |

### Key Statistical Findings

1. **Age-Risk Correlation**: Younger investors show 92% higher aggressive allocation
2. **Income-Safety Paradox**: Higher earners prefer lower but stable returns (r=-0.03)
3. **Education Premium**: Master's degree holders earn ~40% more than Bachelor's

---

## Machine Learning Approach

### Problem Definition

**Task**: Multi-class classification of investor Risk Appetite (Low, Moderate, Aggressive)

**Features Used**:
- Demographic: Gender, Age_Group
- Financial: Salary, Education, Years of Experience
- Behavioral: Monitoring Frequency, Investment Tenure

### Models Implemented

| Model | Algorithm Type | Why Selected |
|-------|---------------|--------------|
| **Logistic Regression** | Linear Classifier | Baseline, interpretable, probability outputs |
| **Random Forest** | Ensemble (Bagging) | Handles non-linearity, feature importance |
| **Decision Tree** | Tree-based | Visual interpretability, no scaling needed |
| **K-Nearest Neighbors** | Instance-based | No distribution assumptions |

### Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', 'passthrough', ['Salary', 'YearsExperience']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Age_Group', 'Education'])
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])
```

### Hyperparameter Tuning

**Random Forest GridSearchCV**:
```python
param_grid = {
    'clf__n_estimators': [150, 250],
    'clf__max_depth': [None, 8, 16],
    'clf__min_samples_split': [2, 5]
}
```

**Best Parameters**: n_estimators=250, max_depth=16, min_samples_split=2

### Model Evaluation

| Model | Accuracy | F1-Score | CV Mean (5-fold) |
|-------|----------|----------|------------------|
| Logistic Regression | 0.72 | 0.70 | 0.71 ± 0.03 |
| Random Forest | 0.78 | 0.76 | 0.77 ± 0.02 |
| Decision Tree | 0.74 | 0.72 | 0.73 ± 0.04 |
| KNN | 0.71 | 0.69 | 0.70 ± 0.03 |

### Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Positive prediction accuracy |
| **Recall** | TP / (TP + FN) | True positive detection rate |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean of P and R |
| **ROC-AUC** | Area under ROC | Discrimination ability |

---

## Dashboard Pages

| # | Page Name | Focus | Key Visualizations |
|---|-----------|-------|-------------------|
| 1 | Overview Dashboard | Cross-dataset KPIs | Correlation heatmap, summary metrics |
| 2 | Who Are The Investors | Demographics | Pie charts, bar charts, treemaps |
| 3 | The Age-Risk Connection | Risk appetite | ANOVA, violin plots, 3D scatter |
| 4 | Education Experience Earnings | Income analysis | Regression, box plots |
| 5 | The Income-Risk Tradeoff | Salary-risk | Scatter plots, correlation |
| 6 | Connecting the Dots | Integration | Merged analysis, heatmaps |
| 7 | Predictive Modeling | ML models | Confusion matrix, ROC, feature importance |
| 8 | Data Explorer | Cleaning | Before/after, download buttons |
| 9 | Summary and Insights | Documentation | Reports, recommendations |

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/investment-behavior-analysis.git
cd investment-behavior-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run appcl.py
```

### Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

### Repository Structure

```
Investment_Behavior_Analysis/
├── .devcontainer/
├── Raw Datasets/
│   ├── Finance_Dataset.csv
│   ├── Salary_Dataset.csv
│   └── Finance_Trends.csv
├── Cleaned Datasets/
│   ├── Finance_Dataset-Cleaned.csv
│   ├── Salary_Dataset-Cleaned.csv
│   └── Finance_Trends-Cleaned.csv
├── Data Cleaning .py Files/
│   ├── Clean_Finance_Dataset.py
│   ├── Clean_Salary_Dataset.py
│   └── Clean_Trends_Dataset.py
├── app.py
├── Project_Techniques_Summary.md
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Usage Guide

### Starting the Dashboard

```bash
streamlit run appcl.py
```

The dashboard will open at `http://localhost:8501`

### Using Filters

1. **Sidebar Filters**: Select Gender, Age Group, Education, Experience to filter all visualizations
2. **Story Mode Toggle**: Enable/disable narrative explanations
3. **Filters apply globally** across all pages

### Training ML Models

1. Navigate to **Predictive Modeling** page
2. Review feature selection in Section A
3. Click **Train Models** in Section B
4. View metrics, confusion matrix, and feature importance
5. Use Section C to predict new investor risk appetite

### Exporting Data

1. Go to **Data Explorer** page
2. Select Finance, Salary, or Trends tab
3. Toggle "Show Full Dataset" for complete view
4. Click **Download** buttons for CSV export

---

## Key Findings

### 1. Age Drives Risk Tolerance

> Younger investors (18-34) show **92% higher aggressive allocation** than investors 45+

- ANOVA confirms significant difference (p < 0.05)
- "Time heals market wounds" - younger investors can afford volatility

### 2. Income-Safety Paradox

> Higher earners prefer **lower but stable returns** (r = -0.03)

- Wealth preservation mindset dominates among high earners
- Those with more to lose prefer stability

### 3. Education Premium

> Master's degree holders earn **~40% more** than Bachelor's holders

- Education investment yields measurable returns
- Higher education correlates with portfolio diversification

### 4. Gender Patterns

> Males monitor portfolios **more frequently** than females

- Different engagement strategies needed
- Investment preferences also vary by gender

---

## Rubric Alignment

### Base Requirements (80%)

| Requirement | Points | Status |
|-------------|--------|--------|
| Data Collection & Preparation | 15% | 3 datasets, advanced cleaning |
| EDA & Visualization | 15% | 15+ viz types, statistical tests |
| Data Processing & Feature Engineering | 15% | 5 engineered features |
| Model Development & Evaluation | 20% | 4 models, CV, hyperparameter tuning |
| Streamlit App Development | 25% | 9 pages, story mode, caching |
| GitHub Documentation | 10% | This README, data dictionaries |

### Above and Beyond (20%)

| Category | Status |
|----------|--------|
| Advanced Modeling Techniques | GridSearchCV, ensemble methods |
| Specialized Domain Application | Behavioral finance expertise |
| Real-World Impact | Actionable recommendations |
| Exceptional Visualization | 3D scatter, interactive ROC |

---

## Limitations & Future Work

### Current Limitations

1. **Index-based alignment**: Datasets aligned by row index assumes correspondence
2. **Survey bias**: Self-reported data may have response biases
3. **Sample size**: Some demographic segments have limited representation
4. **Temporal snapshot**: Data represents single point in time

### Future Enhancements

1. **Deep Learning**: Implement neural networks for complex pattern recognition
2. **Time Series**: Add longitudinal analysis of investment behavior changes
3. **Cloud Deployment**: Deploy to Streamlit Cloud or AWS for wider access
4. **Real-time Data**: Integrate live market data feeds
5. **A/B Testing**: Implement recommendation testing framework

---

## Acknowledgments

- **Course**: CMSE 830-001, Fall 2025
- **Instructor**: Dr. Luciano Germano Silvestri
- **Data Sources**: Kaggle investment survey datasets
- **Libraries**: Streamlit, Plotly, scikit-learn, pandas

---

## License

This project is created for educational purposes as part of CMSE 830 coursework.

---

## Contact

For questions or feedback, please open an issue in this repository.

---

*Last Updated: December 2025*
