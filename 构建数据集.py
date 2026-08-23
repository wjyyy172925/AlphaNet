import gc

import numpy as np
import pandas as pd
from tqdm import tqdm


file_name = "df_merged_fe.csv"
df_merged = pd.read_csv(file_name)
df_merged = df_merged.sort_values(["code", "date"]).reset_index(drop=True)

required_columns = {"code", "date", "open", "close"}
missing_columns = required_columns - set(df_merged.columns)
if missing_columns:
    raise ValueError(f"缺少必要字段: {sorted(missing_columns)}")

volume_column = "volumn" if "volumn" in df_merged.columns else "volume"
if volume_column not in df_merged.columns:
    raise ValueError("缺少成交量字段: volumn 或 volume")

vwap_column = "vwap" if "vwap" in df_merged.columns else None

# X 只使用原始特征列，避免把标签和交易元数据混入输入特征。
feature_columns = [
    column for column in df_merged.columns if column not in {"code", "date"}
]

# 标签和交易信息：t 日收盘生成信号，t+1 日开盘买入，t+10 日收盘卖出。
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

# 未来 10 个交易日的可执行收益率，避免使用 t 日收盘价直接买入。
df_merged["target"] = df_merged["exit_price"] / df_merged["entry_price"] - 1


def valid_positive_series(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & np.isfinite(values) & (values > 0)


# 当前原始文件没有停牌/ST/涨跌停字段，这里先生成基于价格和成交量的可交易性代理。
df_merged["entry_tradable"] = (
    valid_positive_series(df_merged["entry_price"])
    & valid_positive_series(df_merged["entry_volume"])
)
df_merged["exit_tradable"] = (
    valid_positive_series(df_merged["exit_price"])
    & valid_positive_series(df_merged["exit_volume"])
)

# 成交额代理值 = 成交量 * VWAP，用于后续流动性和容量过滤。
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


# 第一遍只统计样本数量，避免先拼接超大列表导致内存峰值过高。
for code in tqdm(codes, desc="count"):
    df = df_merged[df_merged["code"] == code]
    i = 0
    has_sample = False

    while i + 39 < len(df):
        row = df.iloc[i + 29]
        if is_valid_target(row):
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

# 与 Y 对齐的交易元数据数组。
entry_dates = np.empty(total_samples, dtype="U10")
exit_dates = np.empty(total_samples, dtype="U10")
signal_prices = np.empty(total_samples, dtype=np.float32)
entry_prices = np.empty(total_samples, dtype=np.float32)
exit_prices = np.empty(total_samples, dtype=np.float32)
entry_volumes = np.empty(total_samples, dtype=np.float32)
exit_volumes = np.empty(total_samples, dtype=np.float32)
entry_vwaps = np.empty(total_samples, dtype=np.float32)
exit_vwaps = np.empty(total_samples, dtype=np.float32)
entry_tradable = np.empty(total_samples, dtype=bool)
exit_tradable = np.empty(total_samples, dtype=bool)
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

        window = df.iloc[i : i + 30][["date"] + feature_columns]
        window = window.set_index("date").transpose()
        X[pos] = window.to_numpy(dtype=np.float32, copy=False)

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
        entry_tradable[pos] = bool(row["entry_tradable"])
        exit_tradable[pos] = bool(row["exit_tradable"])
        entry_amount_proxy[pos] = np.float32(row["entry_amount_proxy"])
        exit_amount_proxy[pos] = np.float32(row["exit_amount_proxy"])

        write_cursor[date] += 1
        i += 10

    del df
    gc.collect()


# 保存训练数据。
np.save("X_fe.npy", X)
np.save("Y_fe.npy", Y)
np.save("Y_dates.npy", Y_dates)
np.save("Y_codes.npy", Y_codes)

# 保存回测需要的交易元数据。
np.save("entry_dates.npy", entry_dates)
np.save("exit_dates.npy", exit_dates)
np.save("signal_prices.npy", signal_prices)
np.save("entry_prices.npy", entry_prices)
np.save("exit_prices.npy", exit_prices)
np.save("entry_volumes.npy", entry_volumes)
np.save("exit_volumes.npy", exit_volumes)
np.save("entry_vwaps.npy", entry_vwaps)
np.save("exit_vwaps.npy", exit_vwaps)
np.save("entry_tradable.npy", entry_tradable)
np.save("exit_tradable.npy", exit_tradable)
np.save("entry_amount_proxy.npy", entry_amount_proxy)
np.save("exit_amount_proxy.npy", exit_amount_proxy)

# 生成一张便于回测直接读取的样本级元数据表。
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
        "entry_tradable": entry_tradable,
        "exit_tradable": exit_tradable,
        "entry_amount_proxy": entry_amount_proxy,
        "exit_amount_proxy": exit_amount_proxy,
        "target": Y,
    }
).to_csv("sample_meta.csv", index=False)
