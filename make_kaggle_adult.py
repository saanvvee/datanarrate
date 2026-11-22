import pandas as pd

df = pd.read_csv("adult_income.csv")
for c in df.select_dtypes(exclude="number").columns:
    df[c] = df[c].astype(object).where(df[c].notna(), "?")
df.to_csv("adult_kaggle_style.csv", index=False)
print("saved adult_kaggle_style.csv")