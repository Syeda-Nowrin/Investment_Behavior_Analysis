# 🧾 Project Techniques Summary  
### *Investment & Salary Analysis — Data Cleaning, Preparation, and Preprocessing Concepts*

This project demonstrates a full **data preprocessing pipeline** — from raw survey and salary data cleaning to feature engineering and transformation — reflecting core **Data Science & Machine Learning course concepts** like **imputation**, **scaling**, **encoding**, **data transformation**, **sampling**, and **EDA**.

---

## 1. Data Cleaning & Preparation

**Techniques used:**
- **Whitespace trimming:**  
  Removing extra spaces in column names and string values (`str.strip()`).
- **Text normalization / standardization:**  
  Converting categories like `"Master'S Degree"` → `"Master's"` (case normalization and punctuation cleanup).
- **Handling inconsistent column naming:**  
  Renaming variants such as `"Years Of Experience"` → `"Years of Experience"`.
- **Duplicate removal:**  
  Using `drop_duplicates()` to ensure dataset uniqueness.
- **Dealing with inconsistent capitalization:**  
  Using `.str.title()` to make text values uniform.

---

## 2. Data Type Conversion & Parsing

**Techniques used:**
- **Type coercion:**  
  Converting numeric-like text fields to actual numeric (`pd.to_numeric(errors='coerce')`).
- **Regex-based parsing:**  
  Using `re.compile()` and `re.match()` to extract structured information, e.g. renaming survey columns like:  
  `"What do you think are the best options ... [Gold]"` → `"Preference rank for GOLD investment"`.
- **String transformations:**  
  Replacing curly apostrophes and backticks with standard `'` for consistency.

---

## 3. Feature Engineering

**Techniques used:**
- **Derived categorical features:**  
  Creating **Age Group** and **Experience Band** with custom logic — e.g., `25–34`, `10–14 yrs`.
- **Feature standardization:**  
  Merging semantically similar entries under unified labels (e.g., `"Master’s Degree"` → `"Master’s"`).
- **Adding interpretable categorical ranges:**  
  Enables more meaningful grouping and visualization.

---

## 4. Missing Value Handling (Imputation)

**Techniques used:**
- **Numeric imputation using median:**  
  `df[c].fillna(df[c].median())` preserves data distribution and minimizes outlier bias.
- **Categorical imputation with constants:**  
  Replacing text missing values with `"Unknown"` to retain all records for analysis.

---

## 5. Data Transformation & Normalization

**Techniques used:**
- **Category normalization:**  
  Title-casing and punctuation correction for consistent labels.
- **Value mapping & recoding:**  
  Logical replacements like `"Master'S Degree"` → `"Master's"`.
- **Feature renaming for clarity:**  
  Making complex headers concise (e.g., preference rankings renamed to human-readable names).

---

## 6. Encoding & Representation Concepts

Although **explicit encoding** (like one-hot encoding) isn’t performed yet, the cleaning steps prepare categorical fields for it by:
- Converting free-text categories into **finite, uniform sets** (e.g., `Education Level`, `Age Group`).
- Creating categorical bands (`Experience Band`) that are **ready for encoding** in ML pipelines.

---

## 7. Scaling & Numeric Readiness

**Techniques used:**
- Cleaning prepares numeric fields (**Age**, **Salary**, **Years of Experience**) for downstream scaling.
- Median imputation ensures these columns are free from NaN before applying scaling methods.

**Future-ready transformations supported:**
- **Z-score scaling** (StandardScaler)  
- **Min-Max normalization**

---

## 8. Data Validation & Quality Assurance

**Techniques used:**
- Printing dataset **shape and column summaries** for validation.  
- Checking column existence (`if col in df.columns:`) before applying transformations.  
- Safe defaults for coercion (`errors='coerce'`), fallback to `"Unknown"` for resilience.

---

## 9. Data Export & Reproducibility

**Techniques used:**
- Exporting standardized outputs with consistent schema (`*_Cleaned.csv`).  
- Organized directory management using `Pathlib`.  
- Dropping duplicates and saving deterministic files for repeatable results.

---

## 10. Broader Concepts Applied

| **Concept** | **Example from Project** |
|--------------|---------------------------|
| **ETL (Extract–Transform–Load)** | Read raw CSV → Clean → Transform → Save cleaned CSV |
| **Data Pipeline Design** | Sequential steps: Cleaning → Transformation → Feature Engineering → Export |
| **Regex & Pattern Matching** | Renaming complex survey headers dynamically |
| **Outlier Resilience** | Median imputation for numeric stability |
| **Human-Readable Features** | Derived `Age Group` and `Experience Band` |
| **Error-Tolerant Parsing** | `pd.to_numeric(errors="coerce")` prevents failures on bad data |

---

## 11. Data Quality, Processing, and Visualization (EDA & IDA)

**Where applied:**
- After cleaning, data were **ready for Exploratory Data Analysis (EDA)** and **Initial Data Analysis (IDA)** inside the Streamlit dashboard.
- Visuals include:
  - **Heatmaps** → correlation between Age, Experience, Salary, Expected Return.  
  - **Bar and Pie charts** → categorical distributions (Gender, Education).  
  - **Box/Violin plots** → spread and variance of numeric data.  
  - **Treemaps & stacked bars** → interactive pattern exploration.
- Focus on **data quality**, ensuring no duplicates or invalid numeric types before visualization.

## Data Analysis Coverage: Univariate, Bivariate, and Multivariate Insights

### **Univariate Analysis**

Univariate analysis focuses on **single variables**, helping identify distribution, central tendency, frequency, and data quality patterns.

**Implemented in the Dashboard:**

- **Overview → KPIs & Summary Stats**
  - `st.metric` cards for **Average Salary** and **Average Expected Return** — indicators of central tendency.
  - **Summary Statistics (Numeric Fields)** — mean, median, min, and max per numeric column.

- **Overview → Most Popular Investment Type (Pie Chart)**
  - Frequency of a single categorical variable (top rank-1 or fallback column).

- **Interactive Data Explorer → Finance Tab**
  - **Investment Type Distribution (Pie)** — frequency of one categorical variable.  
  - **Monitoring Frequency (Bar)** — count of monitoring frequency categories.  
  - **Missing Values — Finance (Table/Bar/Heatmap)** — column-wise missingness visualization.

- **Interactive Data Explorer → Salary Tab**
  - **Salary Distribution by Experience Band (Histogram)** — univariate salary distribution (color = secondary dimension).  
  - **Missing Values — Salary** — identical data-quality visualization style.

- **Demographics & Behavioral Patterns**
  - **Gender Ratio (Pie)** — categorical frequency for gender.  
  - **Age Group Distribution (Bar)** — categorical frequency for age group.

---

### **Bivariate Analysis**

Bivariate analysis explores **relationships between two variables**, such as numeric-vs-numeric or numeric-vs-categorical patterns.

**Implemented in the Dashboard:**

- **Overview**
  - **Average Salary by Education Level (Bar)** — numeric vs categorical.

- **Investment Behavior (Page 2)**
  - **Risk Appetite by Age Group (Stacked Bar)** — derived risk band vs age group.  
  - **Tenure Preference by Age Group (Grouped Bar)** — categorical vs categorical.  
  - **Monitoring Frequency by Gender (Line)** — two categoricals across ordered frequency.  
  - **Expected Return (%) by Education Level (Violin + Box)** — numeric vs categorical.

- **Salary & Education Insights (Page 3)**
  - **Salary Distribution by Experience Band (Histogram + Color)** — salary vs band.  
  - **Average Salary by Education (Bar)** — numeric vs categorical.  
  - **Salary by Age Group (Box)** — numeric vs categorical.  
  - **Salary vs Years of Experience (Scatter + Trendline)** — numeric vs numeric.

- **Relationship Between Income & Investment (Page 4)**
  - **Salary vs Expected Return (%) (Scatter + Trendline)** — numeric vs numeric.  
  - **Education Level vs Most-Preferred Investment (Grouped Bar)** — two categoricals.

- **Demographics & Behavioral Patterns (Page 5)**
  - **Education Level by Gender (Stacked Bar)** — categorical vs categorical.

---

### **Multivariate Analysis**

Multivariate analysis examines **three or more variables** to reveal deeper, high-dimensional interactions.

**Implemented in the Dashboard:**

- **Correlation Heatmaps (Overview, Page 3, Page 4)**  
  Visualizes pairwise correlations across numeric variables (Age, Salary, Experience, Expected Return).

- **Bubble Chart (Page 4)**  
  **Experience vs Salary** with bubble size and color representing **Expected Return (%)** — three variables simultaneously.

- **Treemap (Page 5)**  
  Hierarchical visualization: **Age → Education → Goal**, showing layered categorical relationships.

---

### **Summary Table**

| **Analysis Type** | **Example Visuals** | **Key Concepts** |
|--------------------|---------------------|------------------|
| **Univariate** | KPI Cards, Pie Charts, Histograms, Summary Tables | Frequency · Spread · Central Tendency |
| **Bivariate** | Bar, Line, Violin, Scatter, Box | Correlation · Categorical Comparison · Trendlines |
| **Multivariate** | Heatmaps, Bubble, Treemap | Inter-Variable Relationships · Multi-Dimensional Analysis |

---

## 12. Sampling, Data Imbalance, and Missingness

**Where applied:**
- **Balanced analysis preparation:**  
  The cleaned datasets ensure equal opportunity for demographic segments (`Age Group`, `Gender`, `Education`) without severe null dominance.
- **Missingness addressed through imputation:**  
  Median replacement keeps numeric stability, while `"Unknown"` prevents dropping valuable categorical samples.
- **Sampling readiness:**  
  Data now suitable for stratified sampling or train-test splits in future modeling.

---

## Summary of Concepts Applied

| **Category** | **Techniques Used** |
|---------------|--------------------|
| **Cleaning** | Trimming, case normalization, deduplication |
| **Imputation** | Median (numeric), “Unknown” (categorical) |
| **Transformation** | Mapping, binning, normalization |
| **Encoding Prep** | Consistent categories ready for one-hot encoding |
| **Scaling Prep** | Numeric coercion, imputation for standardization |
| **Validation** | Error handling, sanity checks, print summaries |
| **EDA & Visualization** | Correlation heatmaps, distribution plots, trend insights |
| **Data Quality & Processing** | Cleaning + assurance for balanced, analyzable data |
| **ETL Workflow** | Load → Clean → Transform → Save |

---

✅ **Overall Description:**

This project integrates the **full lifecycle of preprocessing** taught in your course:
- **Data Quality & Processing:** Cleaned inconsistent raw survey/salary data to a uniform, analysis-ready state.  
- **EDA & Visualization:** Applied interpretive, visual-based exploration to uncover correlations and behavioral patterns.  
- **Imputation & Missingness Handling:** Used statistical and categorical imputations to fill gaps and maintain structure.  
- **Transformation & Encoding:** Normalized categories and created new features ready for machine learning pipelines.  
- **Scaling & Numeric Readiness:** Ensured numeric data integrity for standardization and modeling.  
- **Reproducibility:** Built an automated, maintainable pipeline for repeatable, scalable analysis.
