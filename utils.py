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
            "next_rn": np.asarray(y_true).reshape(-1),
        }
    )


def compute_daily_ic(signal_df):
    daily_ic = []
    for _, day_df in signal_df.groupby("date"):
        if len(day_df) < 20:
            continue
        ic, _ = stats.spearmanr(day_df["pred"], day_df["next_rn"])
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
                "strategy_ret": top_df["next_rn"].mean(),
            }
        )
    daily_df = pd.DataFrame(rows).sort_values("date").set_index("date")
    daily_df["strategy_nav"] = (1 + daily_df["strategy_ret"]).cumprod()
    return daily_df, calc_performance(daily_df["strategy_ret"])


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
