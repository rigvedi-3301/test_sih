import pandas as pd

df = pd.read_csv("csic_database.csv")

columns_to_keep = ["Method", "User-Agent", "URL", "content", "classification"]

columns_to_keep = [col for col in columns_to_keep if col in df.columns]

df_clean = df[columns_to_keep].copy()

df_clean.fillna("", inplace=True)

for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col] = df_clean[col].str.strip()

df_clean.to_csv("csic_cleaned.csv", index=False)