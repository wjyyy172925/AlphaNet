import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm


# df_merged: raw data
# df_merged_fe: data with ratio features
# df_merged_fe1: data with ratio and mean features

file_name = "df_merged_fe.csv"
df_merged = pd.read_csv(file_name)
df_merged = df_merged.sort_values(["code", "date"]).reset_index(drop=True)

# 标签：未来10个交易日收益率（t -> t+10）
df_merged["target"] = df_merged.groupby("code")["close"].shift(-10) / df_merged["close"] - 1

codes = df_merged["code"].unique()
date_counts = {}
empty = []
total_samples = 0


# 第一遍：只统计每个标签日期的样本数，避免一次性拼接爆内存
for code in tqdm(codes, desc="count"):
    df = df_merged[df_merged["code"] == code]
    i = 0
    has_sample = False
    while i + 39 < len(df):
        date = df.iloc[i + 29]["date"]
        date_counts[date] = date_counts.get(date, 0) + 1
        total_samples += 1
        has_sample = True
        i += 10
    if not has_sample:
        empty.append(code)
    del df
    gc.collect()

if not date_counts:
    raise ValueError("没有生成任何样本，请检查原始数据和窗口长度。")

sorted_dates = sorted(date_counts.keys())
date_offsets = {}
cursor = 0
for date in sorted_dates:
    date_offsets[date] = cursor
    cursor += date_counts[date]

# 假设列顺序为：code, date, 特征..., target
feature_dim = df_merged.shape[1] - 3
X = np.empty((total_samples, feature_dim, 30), dtype=np.float32)
Y = np.empty(total_samples, dtype=np.float32)
Y_dates = np.empty(total_samples, dtype="U10")
Y_codes = np.empty(total_samples, dtype="U32")


# 第二遍：按日期直接写入预分配数组
write_cursor = date_offsets.copy()
for code in tqdm(codes, desc="write"):
    df = df_merged[df_merged["code"] == code]
    i = 0
    while i + 39 < len(df):
        date = df.iloc[i + 29]["date"]
        pos = write_cursor[date]

        window = df.iloc[i : i + 30, 1:-1].set_index("date").transpose()
        X[pos] = window.to_numpy(dtype=np.float32, copy=False)
        Y[pos] = np.float32(df.iloc[i + 29]["target"])
        Y_dates[pos] = str(date)
        Y_codes[pos] = str(code)

        write_cursor[date] += 1
        i += 10
    del df
    gc.collect()


# 数据已经按日期写入，无需再做全量排序
np.save("X_fe.npy", X)
np.save("Y_fe.npy", Y)
np.save("Y_dates.npy", Y_dates)
np.save("Y_codes.npy", Y_codes)

pd.DataFrame(
    {
        "date": Y_dates,
        "code": Y_codes,
        "target": Y,
    }
).to_csv("sample_meta.csv", index=False)

