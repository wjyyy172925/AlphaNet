import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import latest_run_dir, make_run_output_dir


OUTPUT_DIR = make_run_output_dir("Dataset_Results")
SOURCE_DIR = latest_run_dir("Baostock_Results")
if SOURCE_DIR is not None and (SOURCE_DIR / "df_merged_fe.csv").exists():
    file_name = SOURCE_DIR / "df_merged_fe.csv"
else:
    file_name = Path("df_merged_fe.csv")

print("Input file:", file_name)
print("Results dir:", OUTPUT_DIR)

df_merged = pd.read_csv(file_name)
df_merged = df_merged.sort_values(["code", "date"]).reset_index(drop=True)

required_columns = {"code", "date", "open", "close"}
missing_columns = required_columns - set(df_merged.columns)
if missing_columns:
    raise ValueError(f"缺少必要字段: {sorted(missing_columns)}")

volume_column = "volume" if "volume" in df_merged.columns else "volumn"
if volume_column not in df_merged.columns:
    raise ValueError("缺少成交量字段: volume 或 volumn")

vwap_column = "vwap" if "vwap" in df_merged.columns else None
can_buy_column = "can_buy" if "can_buy" in df_merged.columns else None
can_sell_column = "can_sell" if "can_sell" in df_merged.columns else None

# 训练时只使用指定特征列。
feature_columns = [
    "open",
    "close",
    "high",
    "low",
    "volume",
    "vwap",
    "return",
    "turn",
    "close_turn",
    "open_turn",
    "volume_low",
    "vwap_high",
    "low_high",
    "vwap_close",
    "turn_volume",
]
missing_features = [c for c in feature_columns if c not in df_merged.columns]
if missing_features:
    raise ValueError(f"缺少训练特征字段: {missing_features}")

# t 日收盘发信号，t+1 日开盘买入，t+10 日收盘卖出。
grouped = df_merged.groupby("code", sort=False)
df_merged["signal_date"] = df_merged["date"]
df_merged["entry_date"] = grouped["date"].shift(-1)
df_merged["exit_date"] = grouped["date"].shift(-10)
df_merged["signal_price"] = df_merged["close"]
df_merged["entry_price"] = grouped["open"].shift(-1)
df_merged["exit_price"] = grouped["close"].shift(-10)
df_merged["entry_volume"] = grouped[volume_column].shift(-1)
df_merged["exit_volume"] = grouped[volume_column].shift(-10)

if vwap_column is not None:
    df_merged["entry_vwap"] = grouped[vwap_column].shift(-1)
    df_merged["exit_vwap"] = grouped[vwap_column].shift(-10)
else:
    df_merged["entry_vwap"] = df_merged["entry_price"]
    df_merged["exit_vwap"] = df_merged["exit_price"]

df_merged["target"] = df_merged["exit_price"] / df_merged["entry_price"] - 1


def valid_positive_series(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & np.isfinite(values) & (values > 0)


# entry_tradable: t+1 是否可买；exit_tradable: t+10 是否可卖。
entry_tradable = (
    valid_positive_series(df_merged["entry_price"])
    & valid_positive_series(df_merged["entry_volume"])
)
exit_tradable = (
    valid_positive_series(df_merged["exit_price"])
    & valid_positive_series(df_merged["exit_volume"])
)

if can_buy_column is not None:
    entry_tradable = (
        entry_tradable
        & grouped[can_buy_column].shift(-1).fillna(0).astype(bool)
    )
if can_sell_column is not None:
    exit_tradable = (
        exit_tradable
        & grouped[can_sell_column].shift(-10).fillna(0).astype(bool)
    )

df_merged["entry_tradable"] = entry_tradable
df_merged["exit_tradable"] = exit_tradable
df_merged["sample_tradeable"] = (
    df_merged["entry_tradable"] & df_merged["exit_tradable"]
)

# 成交额代理，用于后续容量或流动性过滤。
df_merged["entry_amount_proxy"] = (
    df_merged["entry_volume"] * df_merged["entry_vwap"]
)
df_merged["exit_amount_proxy"] = (
    df_merged["exit_volume"] * df_merged["exit_vwap"]
)

codes = df_merged["code"].unique()
date_counts = {}
empty = []
total_samples = 0


def is_valid_target(row):
    target = row["target"]
    return pd.notna(target) and np.isfinite(float(target))


def build_window_values(df, start_idx):
    window = df.iloc[start_idx : start_idx + 30][["date"] + feature_columns]
    window = window.set_index("date").transpose()
    window_values = window.to_numpy(dtype=np.float64, copy=True)

    if not np.isfinite(window_values).all():
        return None

    row_mean = np.mean(window_values, axis=1, keepdims=True)
    row_std = np.std(window_values, axis=1, keepdims=True)
    valid_rows = row_std > 1e-10
    np.divide(
        window_values - row_mean,
        row_std,
        out=window_values,
        where=valid_rows,
    )

    return window_values.astype(np.float32, copy=False)


# 第一遍只统计样本数，避免先拼接超大列表导致内存峰值过高。
for code in tqdm(codes, desc="count"):
    df = df_merged[df_merged["code"] == code]
    i = 0
    has_sample = False

    while i + 39 < len(df):
        row = df.iloc[i + 29]
        window_values = build_window_values(df, i)
        if is_valid_target(row) and window_values is not None:
            date = row["signal_date"]
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

feature_dim = len(feature_columns)
X = np.empty((total_samples, feature_dim, 30), dtype=np.float32)
Y = np.empty(total_samples, dtype=np.float32)
Y_dates = np.empty(total_samples, dtype="U10")
Y_codes = np.empty(total_samples, dtype="U32")

entry_dates = np.empty(total_samples, dtype="U10")
exit_dates = np.empty(total_samples, dtype="U10")
signal_prices = np.empty(total_samples, dtype=np.float32)
entry_prices = np.empty(total_samples, dtype=np.float32)
exit_prices = np.empty(total_samples, dtype=np.float32)
entry_volumes = np.empty(total_samples, dtype=np.float32)
exit_volumes = np.empty(total_samples, dtype=np.float32)
entry_vwaps = np.empty(total_samples, dtype=np.float32)
exit_vwaps = np.empty(total_samples, dtype=np.float32)
entry_tradable_arr = np.empty(total_samples, dtype=bool)
exit_tradable_arr = np.empty(total_samples, dtype=bool)
sample_tradeable_arr = np.empty(total_samples, dtype=bool)
entry_amount_proxy = np.empty(total_samples, dtype=np.float32)
exit_amount_proxy = np.empty(total_samples, dtype=np.float32)


# 第二遍按信号日期直接写入预分配数组，结果天然按日期排序。
write_cursor = date_offsets.copy()
for code in tqdm(codes, desc="write"):
    df = df_merged[df_merged["code"] == code]
    i = 0

    while i + 39 < len(df):
        row = df.iloc[i + 29]
        if not is_valid_target(row):
            i += 10
            continue

        date = row["signal_date"]
        pos = write_cursor[date]

        window_values = build_window_values(df, i)
        if window_values is None:
            i += 10
            continue

        X[pos] = window_values

        Y[pos] = np.float32(row["target"])
        Y_dates[pos] = str(row["signal_date"])
        Y_codes[pos] = str(code)

        entry_dates[pos] = str(row["entry_date"])
        exit_dates[pos] = str(row["exit_date"])
        signal_prices[pos] = np.float32(row["signal_price"])
        entry_prices[pos] = np.float32(row["entry_price"])
        exit_prices[pos] = np.float32(row["exit_price"])
        entry_volumes[pos] = np.float32(row["entry_volume"])
        exit_volumes[pos] = np.float32(row["exit_volume"])
        entry_vwaps[pos] = np.float32(row["entry_vwap"])
        exit_vwaps[pos] = np.float32(row["exit_vwap"])
        entry_tradable_arr[pos] = bool(row["entry_tradable"])
        exit_tradable_arr[pos] = bool(row["exit_tradable"])
        sample_tradeable_arr[pos] = bool(row["sample_tradeable"])
        entry_amount_proxy[pos] = np.float32(row["entry_amount_proxy"])
        exit_amount_proxy[pos] = np.float32(row["exit_amount_proxy"])

        write_cursor[date] += 1
        i += 10

    del df
    gc.collect()

# X：每个样本的 30 日历史特征，形状大致是 (样本数, 特征数, 30)，样本数是调仓天数
# Y：对应样本的未来收益率标签，也就是 t+1 买入到 t+10 卖出的收益率
np.save(OUTPUT_DIR / "X_fe.npy", X)
np.save(OUTPUT_DIR / "Y_fe.npy", Y)
np.save(OUTPUT_DIR / "Y_dates.npy", Y_dates)
np.save(OUTPUT_DIR / "Y_codes.npy", Y_codes)

# sample_meta.csv的长度是调仓天数
pd.DataFrame(
    {
        "date": Y_dates,
        "signal_date": Y_dates,
        "code": Y_codes,
        "entry_date": entry_dates,
        "exit_date": exit_dates,
        "signal_price": signal_prices,
        "entry_price": entry_prices,
        "exit_price": exit_prices,
        "entry_volume": entry_volumes,
        "exit_volume": exit_volumes,
        "entry_vwap": entry_vwaps,
        "exit_vwap": exit_vwaps,
        "entry_tradable": entry_tradable_arr,
        "exit_tradable": exit_tradable_arr,
        "sample_tradeable": sample_tradeable_arr,
        "entry_amount_proxy": entry_amount_proxy,
        "exit_amount_proxy": exit_amount_proxy,
        "target": Y,
    }
).to_csv(OUTPUT_DIR / "sample_meta.csv", index=False)
