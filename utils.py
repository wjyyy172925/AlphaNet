import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def load_dataset(data_dir="."):
    x = np.load(os.path.join(data_dir, "X_fe.npy"))
    y = np.load(os.path.join(data_dir, "Y_fe.npy"))
    dates = np.load(os.path.join(data_dir, "Y_dates.npy"))
    codes = np.load(os.path.join(data_dir, "Y_codes.npy"), allow_pickle=True)
    return x, y, dates, codes


def load_sample_meta(data_dir="."):
    path = os.path.join(data_dir, "sample_meta.csv")
    if not os.path.exists(path):
        return None

    meta = pd.read_csv(path)
    for col in ("date", "signal_date", "entry_date", "exit_date"):
        if col in meta.columns:
            meta[col] = pd.to_datetime(meta[col], errors="coerce")
    for col in ("entry_tradable", "exit_tradable", "sample_tradeable"):
        if col in meta.columns:
            meta[col] = meta[col].fillna(False).astype(bool)
    return meta


def to_date_array(dates):
    return np.array([pd.Timestamp(date).date() for date in dates])


class myDataset(Dataset):
    def __init__(self, X, y, scaler=None, is_train=True):
        super().__init__()
        X = np.asarray(X)
        y = np.asarray(y)
        self.origin_shape = X.shape

        X_2d = X.transpose(0, 2, 1).reshape(-1, self.origin_shape[1])
        if is_train:
            self.scaler = StandardScaler()
            X_trans = self.scaler.fit_transform(X_2d)
        else:
            if scaler is None:
                raise ValueError("scaler is required when is_train=False")
            self.scaler = scaler
            X_trans = self.scaler.transform(X_2d)

        self.X = torch.as_tensor(X_trans.reshape(self.origin_shape), dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_scaler(self):
        return self.scaler


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device=None):
    if device is None:
        device = torch.device("cpu")
    weights = torch.load(path, map_location=device)
    model.load_state_dict(weights)
    return model


def predict_model(model, loader, device=None):
    if device is None:
        device = torch.device("cpu")
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            batch_pred = model(x).detach().cpu().reshape(-1)
            preds.append(batch_pred)
    return torch.cat(preds).numpy() if preds else np.array([])


def build_signal_frame(preds, y_true, target_dates, codes):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(target_dates),
            "code": np.asarray(codes),
            "pred": np.asarray(preds).reshape(-1),
            "target": np.asarray(y_true).reshape(-1),
        }
    )


def apply_tradeability_filter(signal_df):
    df = signal_df.copy()
    mask = df["sample_tradeable"].fillna(False).astype(bool)
    if "target" in df.columns:
        df.loc[~mask, "target"] = np.nan
    return df


def compute_daily_ic(signal_df):
    daily_ic = []
    for _, day_df in signal_df.groupby("date"):
        if len(day_df) < 20:
            continue
        ic, _ = stats.spearmanr(day_df["pred"], day_df["target"])
        daily_ic.append(ic)
    return np.asarray(daily_ic)


def calc_performance(ret_series, periods_per_year=252 / 10):
    ret_series = pd.Series(ret_series).dropna()
    if ret_series.empty:
        return {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
        }
    nav = (1 + ret_series).cumprod()
    annual_return = nav.iloc[-1] ** (periods_per_year / len(ret_series)) - 1
    annual_volatility = ret_series.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
    max_drawdown = (nav / nav.cummax() - 1).min()
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def backtest_topk(signal_df, topk=50):
    rows = []
    signal_df = signal_df.sort_values(["date", "pred"], ascending=[True, False]).copy()
    for date, day_df in signal_df.groupby("date"):
        top_df = day_df.sort_values("pred", ascending=False).head(min(topk, len(day_df)))
        rows.append(
            {
                "date": date,
                "strategy_ret": top_df["target"].mean(),
            }
        )
    daily_df = pd.DataFrame(rows).sort_values("date").set_index("date")
    daily_df["strategy_nav"] = (1 + daily_df["strategy_ret"]).cumprod()
    return daily_df, calc_performance(daily_df["strategy_ret"])


def backtest_group_strategy(signal_df, group_num=10):
    signal_df = signal_df.copy().dropna(subset=["pred", "target"])
    group_rows = []

    for date, day_df in signal_df.groupby("date"):
        day_df = day_df.sort_values("pred", ascending=True).copy()
        size = len(day_df)
        if size == 0:
            continue

        # 将每只股票按预测值从低到高等分成N组，组号越大表示预测值越高
        day_df["group"] = np.ceil(
            day_df["pred"].rank(method="first") / (size / group_num)
        ).astype(int)
        day_df["group"] = day_df["group"].clip(1, group_num)

        # 某一天每个分组（1到5组）的平均收益率(group，date，group_ret)
        day_group = day_df.groupby("group", as_index=False)["target"].mean().rename(
            columns={"target": "group_ret"}
        )
        day_group["date"] = date
        group_rows.append(day_group)

    if not group_rows:
        empty_group_curve = pd.DataFrame(columns=["date", "group", "group_ret", "group_nav"])
        empty_strategy_curve = pd.DataFrame(columns=["strategy_ret", "strategy_nav"])
        return empty_group_curve, empty_strategy_curve, np.nan, calc_performance([])

    group_curve = pd.concat(group_rows, ignore_index=True)
    group_curve = group_curve.sort_values(["group", "date"]).reset_index(drop=True)
    group_curve["group_nav"] = group_curve.groupby("group")["group_ret"].transform(
        lambda x: (1 + x).cumprod()
    )

    strategy_group = int(group_num)
    strategy_curve = (
        group_curve.query("group == @strategy_group")
        .sort_values("date")
        .set_index("date")
        .rename(columns={"group_ret": "strategy_ret"})
    )
    strategy_curve["strategy_nav"] = (1 + strategy_curve["strategy_ret"]).cumprod()
    strategy_stats = calc_performance(strategy_curve["strategy_ret"])
    return group_curve, strategy_curve, strategy_group, strategy_stats


def build_rolling_splits(target_dates, train_window=1500, valid_ratio=0.8, test_window=126, step=126):
    target_dates = np.asarray(target_dates)
    unique_dates = sorted(np.unique(target_dates))
    splits = []
    i = 0
    k = int(train_window * valid_ratio)
    while i + train_window + test_window <= len(unique_dates):
        splits.append(
            (
                sum(target_dates < unique_dates[i]),
                sum(target_dates < unique_dates[i + k]),
                sum(target_dates < unique_dates[i + train_window]),
                sum(target_dates < unique_dates[i + train_window + test_window]),
            )
        )
        i += step
    return splits
