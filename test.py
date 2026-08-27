import pandas as pd
df = pd.read_csv('df_merged.csv', nrows=1)
print(df.iloc[0]["low"])
print(df.iloc[0]["high"])
print(df.iloc[0]["vwap"])