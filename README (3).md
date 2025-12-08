# Investment Behavior Analysis Dashboard

## CMSE 830-001 | Fall 2025 | Final Project

A Streamlit dashboard that analyzes investment behavior patterns across three datasets (Finance, Salary, Trends). The project integrates demographics, income, experience, and investment preference signals to explore behavioral finance questions and to build predictive models for investor risk appetite.

## Table of Contents

1. Project Overview  
2. Key Features  
3. Data Sources  
4. Data Cleaning and Preprocessing  
5. Feature Engineering  
6. Statistical Analysis Methods  
7. Machine Learning Approach  
8. Dashboard Pages  
9. Installation and Setup  
10. Usage Guide  
11. Key Findings  
12. Rubric Alignment  
13. Limitations and Future Work  
14. Acknowledgments  

## Project Overview

### The Problem

Modern investors have unprecedented access to investment options, yet many remain uncertain about how to allocate capital. Financial professionals and fintech tools often struggle to personalize recommendations without a clear understanding of the demographic and behavioral factors that drive risk tolerance and investment choice.

### Our Solution

This dashboard integrates three distinct datasets to analyze:

- How age, gender, and education influence risk tolerance  
- Patterns between income levels and expected returns  
- Differences in monitoring behavior and investment horizons  
- Predictive models for risk appetite classification  

### Business Impact

| Stakeholder | Application |
|-------------|-------------|
| Financial Advisors | Age-based risk profiling and portfolio recommendations |
| Fintech Companies | Personalized product suggestions and UX design |
| HR Departments | Employee financial wellness program design |
| Researchers | Behavioral finance academic studies |

## Key Features

- Interactive visualizations across all three datasets  
- Cross-dataset comparison and integration views  
- Advanced cleaning pipelines with before/after inspection  
- Feature engineering for demographic and behavioral segmentation  
- Statistical tests (where applicable) to support findings  
- Multi-model classification of risk appetite with evaluation visuals  
- Hyperparameter tuning for Random Forest using GridSearchCV  
- Global sidebar filters and story mode for guided narration  
- Data export for filtered and cleaned outputs  

## Data Sources

| Dataset | Focus Area | Example Fields |
|---------|------------|----------------|
| Finance | Investment preferences and risk signals | Gender, Age/Age_Group, expected return, monitoring frequency |
| Salary | Income, education, experience | Salary, Education, Years of Experience |
| Trends | Market behavior patterns and return expectations | Investment avenue, duration, expected return |

All three sources are used to satisfy the multi-source requirement and to enable cross-dataset pattern discovery.

## Data Cleaning and Preprocessing

The project includes dataset-specific cleaning scripts plus in-app validation and display of raw vs cleaned results.

Common techniques include:

- Column name standardization  
- Text normalization for categorical values  
- Data type conversion for numeric-like fields  
- Handling missing values with median (numeric) and Unknown (categorical) strategies  
- Duplicate removal where applicable  
- Age and experience sanity checks  

Refer to the data cleaning scripts in the repository for reproducible pipelines.

## Feature Engineering

Key engineered variables used across analysis and modeling:

- Age_Group derived from numeric ages when needed  
- Risk appetite bands derived from expected return  
- Experience bands for workforce segmentation  
- Salary-based groupings for income-driven analysis  

These features enable consistent visualization and modeling across heterogeneous survey formats.

## Statistical Analysis Methods

The dashboard uses descriptive and inferential methods as supported by the available columns:

- Descriptive summaries across numeric fields  
- Group comparisons across age, gender, and education categories  
- Correlation analysis for salary and expected return relationships  
- Significance testing on selected pages where aligned variables are available  

## Machine Learning Approach

### Problem Definition

Task: Multi-class classification of investor Risk Appetite (Low, Moderate, Aggressive)

### Features Used

- Demographic: Gender, Age_Group  
- Financial/Professional: Salary, Education, YearsExperience  

### Models Implemented

- Logistic Regression  
- Random Forest  
- K-Nearest Neighbors  
- Decision Tree  

### Preprocessing

Categorical variables are one-hot encoded using a ColumnTransformer inside a scikit-learn Pipeline to ensure consistent training and inference.

### Hyperparameter Tuning

Random Forest GridSearchCV is included to demonstrate model optimization and cross-validated performance comparison.

## Dashboard Pages

| # | Page Name | Focus |
|---|-----------|-------|
| 1 | Overview Dashboard | Cross-dataset KPIs and high-level patterns |
| 2 | Who Are Our Investors | Demographic profiling across datasets |
| 3 | The Age-Risk Connection | Age-driven risk appetite analysis |
| 4 | Education Experience Earnings | Education/experience effects on salary and strategy |
| 5 | The Income-Risk Tradeoff | Salary and expected return relationships |
| 6 | Connecting the Dots | Cross-dataset integration methodology |
| 7 | Predictive Modeling | Model training, evaluation, tuning, and prediction |
| 8 | Data Explorer | Raw vs cleaned inspection and downloads |
| 9 | Summary and Insights | Consolidated conclusions and user guide |

## Installation and Setup

### Prerequisites

- Python 3.8+  
- pip  

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/Investment_Behavior_Analysis.git
cd Investment_Behavior_Analysis

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Requirements

See `requirements.txt` in the repository.

## Repository Structure

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

If your filenames differ slightly, update the structure block to match the exact names in your repo.

## Usage Guide

### Starting the Dashboard

```bash
streamlit run app.py
```

The app will open in your browser at the default Streamlit address.

### Using Global Filters

- Sidebar filters apply across pages to keep comparisons consistent.  
- Filters typically include gender, age group, education, and experience (where available).  

### Predictive Modeling

1. Go to the Predictive Modeling page.  
2. Review the modeling dataset creation.  
3. Train and compare the four models.  
4. Review metrics, cross-validation, confusion matrix, feature importance, and ROC-AUC visuals.  
5. Use the prediction form to test a new investor profile.

### Data Downloads

The Data Explorer page provides raw and cleaned previews and allows exporting filtered datasets.

## Key Findings

- Age is a strong driver of risk tolerance, with younger groups tending toward higher-risk profiles.  
- Higher income groups may show more conservative expected return preferences in aggregate.  
- Education and experience appear to support diversified and goal-aligned investment behavior.  
- Monitoring frequency patterns differ across demographics.  

These conclusions are exploratory and grounded in survey-based evidence presented in the dashboard.

## Rubric Alignment

This project is designed to satisfy all base requirements:

- Three distinct data sources  
- Advanced cleaning and preprocessing  
- Cross-dataset integration and comparative analysis  
- Multiple visualization types and EDA  
- Feature engineering  
- At least two ML models with thorough evaluation  
- A comprehensive Streamlit application with interactive elements  
- GitHub documentation and reproducible structure  

Above-and-beyond components are supported through:

- Hyperparameter tuning  
- Rich, interactive visual diagnostics  
- Real-world recommendations and domain framing  

## Limitations and Future Work

### Current Limitations

1. Cross-dataset alignment is conceptual; datasets do not share true respondent identifiers.  
2. Survey responses are self-reported and may contain bias.  
3. Some demographic segments have limited representation.  
4. The data represents a snapshot in time.

### Future Enhancements

- Expand hyperparameter tuning and add additional ensemble methods.  
- Explore time-series or longitudinal datasets if available.  
- Deploy to a cloud host with automated CI checks.  
- Add stronger data validation rules and automated tests.

## Acknowledgments

Course: CMSE 830-001, Fall 2025  
Instructor: Dr. Silvestri  
Data sources: Kaggle (Finance, Salary, Trends surveys)
