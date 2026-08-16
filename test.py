import pandas as pd

df = pd.read_csv(r'df_merged.csv')
print(df['code'].nunique())  # ← 去掉 *