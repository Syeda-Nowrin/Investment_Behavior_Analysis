import pandas as pd

finance = pd.read_csv("Finance_Dataset_Cleaned.csv")
salary = pd.read_csv("Salary_Dataset_Cleaned.csv")

# View types
print("Finance Dataset types:\n", finance.dtypes)
print("\nSalary Dataset types:\n", salary.dtypes)

# Count how many categorical vs numeric columns
finance_cats = finance.select_dtypes('object').columns
finance_nums = finance.select_dtypes('number').columns

print(f"\nFinance: {len(finance_cats)} categorical, {len(finance_nums)} numeric")
