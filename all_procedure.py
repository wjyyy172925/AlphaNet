# ============================================================
# Cell 1: Import Libraries & Global Configuration
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import warnings
import time
import os
from itertools import combinations
from datetime import datetime, timedelta

# JoinQuant Data API
from jqdata import *

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, BatchNormalization,
    Concatenate, Lambda, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import backend as K

# Settings
warnings.filterwarnings('ignore')
# FIX 1: Removed Chinese fonts to prevent garbled text in plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-whitegrid')

# Random seed
SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)

# ==================== Global Parameters ====================
CONFIG = {
    'start_date':          '2015-01-01',
    'end_date':            '2025-12-31',
    'backtest_start':      '2020-01-01',
    'lookback_days':       750,
    'n_features':          9,
    'seq_len':             30,
    'stride_extract':      10,
    'stride_pool':         3,
    'd_extract':           10,
    'd_pool':              3,
    'hidden_units':        30,
    'dropout_rate':        0.5,
    'learning_rate':       0.0001,
    'batch_size':          1000,
    'early_stop_patience': 10,
    'n_epochs':            100,
    'retrain_months':      6,
    'n_repeats':           3,
    'rebalance_days':      10,
    'n_layers':            5,
    'transaction_cost':    0.001,
}

print("Library import complete, global config set.")
print(f"   Backtest period : {CONFIG['backtest_start']} ~ {CONFIG['end_date']}")
print(f"   Rebalance days  : {CONFIG['rebalance_days']}")
print(f"   Repeat training : {CONFIG['n_repeats']}")


# ============================================================
# Cell 2: Data Fetching & Preprocessing
# ============================================================

def get_stock_pool(date):
    """Get stock pool: all A-shares, excluding ST/PT and suspended stocks."""
    all_stocks = get_all_securities('stock', date=date)
    stocks = all_stocks[all_stocks.index.str.startswith(('0', '3', '6'))].index.tolist()

    # Exclude ST stocks
    st_info = get_extras('is_st', stocks, start_date=date, end_date=date, df=True)
    if len(st_info) > 0:
        st_stocks = st_info.columns[st_info.iloc[0] == True].tolist()
        stocks = [s for s in stocks if s not in st_stocks]

    # Exclude suspended stocks
    paused = get_extras('is_trading', stocks, start_date=date, end_date=date, df=True)
    if len(paused) > 0:
        paused_stocks = paused.columns[paused.iloc[0] == False].tolist()
        stocks = [s for s in stocks if s not in paused_stocks]

    return stocks


def fetch_raw_data(stocks, end_date, n_days=60):
    """
    Fetch raw OHLCV data for a list of stocks.
    Returns: dict, key=stock_code, value=DataFrame(9 features, indexed by date)
    """
    trade_days = get_trade_days(end_date=end_date, count=n_days)
    start = str(trade_days[0])

    price_df = get_price(
        stocks, start_date=start, end_date=end_date,
        frequency='daily',
        fields=['open', 'close', 'high', 'low', 'volume'],
        skip_paused=False, fq='post',
        panel=False
    )

    # Approximate vwap using money/volume
    money_df = get_price(
        stocks, start_date=start, end_date=end_date,
        frequency='daily',
        fields=['money'],
        skip_paused=False, fq='post',
        panel=False
    )

    # FIX 2: Removed bare 'set_seed' statement — it was a NameError bug
    # that caused fetch_raw_data() to crash silently, leaving current_model=None.
    q = query(
        valuation.code,
        valuation.turnover_ratio,
    ).filter(
        valuation.code.in_(stocks)
    )

    all_data = {}
    for stock in stocks:
        try:
            stk_price = price_df[price_df['code'] == stock].set_index('time')
            stk_money = money_df[money_df['code'] == stock].set_index('time')

            if len(stk_price) < n_days * 0.8:
                continue

            df = pd.DataFrame(index=stk_price.index)
            df['open']  = stk_price['open']
            df['high']  = stk_price['high']
            df['low']   = stk_price['low']
            df['close'] = stk_price['close']

            # vwap
            df['vwap'] = stk_money['money'] / stk_price['volume']
            df['vwap'] = df['vwap'].replace([np.inf, -np.inf], np.nan).fillna(df['close'])

            # volume
            df['volume'] = stk_price['volume']

            # return1
            df['return1'] = df['close'].pct_change()

            # Turnover (simplified using volume)
            df['turn']      = stk_price['volume']
            df['free_turn'] = stk_price['volume']

            df = df.dropna()
            if len(df) >= CONFIG['seq_len']:
                all_data[stock] = df
        except:
            continue

    return all_data


def prepare_dataset_for_date(date, stocks, lookback=30, forward_days=10):
    """
    Prepare dataset for a given cross-section date.
    Returns: X (n_stocks, 9, 30), Y (n_stocks,), valid_stocks list
    """
    trade_days  = get_trade_days(end_date=date, count=lookback + 5)
    future_days = get_trade_days(start_date=date, count=forward_days + 2)

    if len(future_days) < forward_days + 1:
        return None, None, None

    future_end = str(future_days[-1])
    hist_start = str(trade_days[0])

    price_data = get_price(
        stocks, start_date=hist_start, end_date=future_end,
        frequency='daily',
        fields=['open', 'close', 'high', 'low', 'volume', 'money'],
        skip_paused=False, fq='post', panel=False
    )

    X_list, Y_list, valid_stocks = [], [], []

    for stock in stocks:
        try:
            stk = price_data[price_data['code'] == stock].set_index('time').sort_index()

            hist_end_idx = stk.index.get_indexer(
                [pd.Timestamp(date)], method='ffill'
            )[0]
            if hist_end_idx < lookback - 1:
                continue

            hist = stk.iloc[hist_end_idx - lookback + 1: hist_end_idx + 1]
            if len(hist) < lookback:
                continue

            features    = np.zeros((9, lookback))
            features[0] = hist['open'].values
            features[1] = hist['high'].values
            features[2] = hist['low'].values
            features[3] = hist['close'].values
            vwap = (hist['money'] / hist['volume']).replace([np.inf, -np.inf], np.nan)
            features[4] = vwap.fillna(hist['close']).values
            features[5] = hist['volume'].values
            features[6] = hist['close'].pct_change().fillna(0).values
            features[7] = hist['volume'].values
            features[8] = hist['volume'].values

            if np.isnan(features).any():
                continue

            future = stk.iloc[hist_end_idx + 1: hist_end_idx + 1 + forward_days]
            if len(future) < forward_days:
                continue

            ret = future['close'].iloc[-1] / stk['close'].iloc[hist_end_idx] - 1
            if np.isnan(ret) or np.isinf(ret):
                continue

            X_list.append(features)
            Y_list.append(ret)
            valid_stocks.append(stock)

        except:
            continue

    if len(X_list) == 0:
        return None, None, None

    X = np.array(X_list)
    Y = np.array(Y_list)

    Y_mean = np.mean(Y)
    Y_std  = np.std(Y)
    if Y_std > 1e-8:
        Y = (Y - Y_mean) / Y_std

    return X, Y, valid_stocks

print("Data fetching functions defined.")


# ============================================================
# Cell 3: AlphaNet Feature Computation
# ============================================================

def ts_corr(x, y, d):
    """Pearson correlation of x and y over the last d days."""
    if len(x) < d or len(y) < d:
        return np.nan
    return np.corrcoef(x[-d:], y[-d:])[0, 1]


def ts_cov(x, y, d):
    """Covariance of x and y over the last d days."""
    if len(x) < d or len(y) < d:
        return np.nan
    return np.cov(x[-d:], y[-d:])[0, 1]


def ts_stddev(x, d):
    """Standard deviation of x over the last d days."""
    if len(x) < d:
        return np.nan
    return np.std(x[-d:], ddof=1)


def ts_zscore(x, d):
    """Mean / std of x over the last d days."""
    if len(x) < d:
        return np.nan
    std = np.std(x[-d:], ddof=1)
    if std < 1e-10:
        return 0.0
    return np.mean(x[-d:]) / std


def ts_return(x, d):
    """Return from d days ago to now."""
    if len(x) < d + 1:
        return np.nan
    if abs(x[-d - 1]) < 1e-10:
        return 0.0
    return x[-1] / x[-d - 1] - 1


def ts_decaylinear(x, d):
    """Linearly decayed weighted average over the last d days."""
    if len(x) < d:
        return np.nan
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    return np.sum(x[-d:] * weights)


def ts_mean(x, d):
    """Mean of x over the last d days."""
    if len(x) < d:
        return np.nan
    return np.mean(x[-d:])


def ts_max(x, d):
    """Max of x over the last d days."""
    if len(x) < d:
        return np.nan
    return np.max(x[-d:])


def ts_min(x, d):
    """Min of x over the last d days."""
    if len(x) < d:
        return np.nan
    return np.min(x[-d:])


def compute_alphanet_features(data_image, d=10, stride=10, d_pool=3, stride_pool=3):
    """
    Compute AlphaNet features for a single stock's data image.
    Input : data_image shape (9, 30)
    Output: 1-D feature vector
    """
    n_features, n_days = data_image.shape
    n_steps = (n_days - d) // stride + 1

    feature_names = ['open', 'high', 'low', 'close', 'vwap',
                     'volume', 'return1', 'turn', 'free_turn']

    all_extract_features = []

    # ===== 1. ts_corr and ts_cov (pairwise) =====
    pairs = list(combinations(range(n_features), 2))

    corr_features = []
    cov_features  = []

    for step in range(n_steps):
        end_idx   = d + step * stride
        start_idx = end_idx - d

        for i, j in pairs:
            x = data_image[i, start_idx:end_idx]
            y = data_image[j, start_idx:end_idx]

            corr_val = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr_val):
                corr_val = 0.0
            corr_features.append(corr_val)

            cov_val = np.cov(x, y)[0, 1]
            if np.isnan(cov_val):
                cov_val = 0.0
            cov_features.append(cov_val)

    n_pairs       = len(pairs)
    corr_features = np.array(corr_features).reshape(n_steps, n_pairs).T  # (36, n_steps)
    cov_features  = np.array(cov_features).reshape(n_steps, n_pairs).T

    # ===== 2. Single-variable operators =====
    single_ops = {
        'stddev':      lambda x, d: np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0,
        'zscore':      lambda x, d: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 1e-10 else 0,
        'return':      lambda x, d: (x[-1] / x[0] - 1) if abs(x[0]) > 1e-10 else 0,
        'decaylinear': lambda x, d: np.sum(x * np.arange(1, len(x)+1) / np.sum(np.arange(1, len(x)+1))),
        'mean':        lambda x, d: np.mean(x),
    }

    single_features_dict = {op: [] for op in single_ops}

    for step in range(n_steps):
        end_idx   = d + step * stride
        start_idx = end_idx - d

        for feat_idx in range(n_features):
            x = data_image[feat_idx, start_idx:end_idx]
            for op_name, op_func in single_ops.items():
                val = op_func(x, d)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                single_features_dict[op_name].append(val)

    single_features_arrays = {}
    for op_name in single_ops:
        arr = np.array(single_features_dict[op_name]).reshape(n_steps, n_features).T  # (9, n_steps)
        single_features_arrays[op_name] = arr

    # ===== 3. Collect extraction layer outputs =====
    extract_outputs = {
        'corr':        corr_features,
        'cov':         cov_features,
        'stddev':      single_features_arrays['stddev'],
        'zscore':      single_features_arrays['zscore'],
        'return':      single_features_arrays['return'],
        'decaylinear': single_features_arrays['decaylinear'],
        'mean':        single_features_arrays['mean'],
    }

    # ===== 4. Flatten extraction layer =====
    extract_flat = []
    for name, feat in extract_outputs.items():
        extract_flat.append(feat.flatten())

    # ===== 5. Pooling layer =====
    pool_flat = []
    for name, feat in extract_outputs.items():
        n_dim, n_t       = feat.shape
        n_pool_steps     = (n_t - d_pool) // stride_pool + 1

        for ps in range(n_pool_steps):
            end_p   = d_pool + ps * stride_pool
            start_p = end_p - d_pool
            window  = feat[:, start_p:end_p]

            pool_flat.append(np.mean(window, axis=1))
            pool_flat.append(np.max(window,  axis=1))
            pool_flat.append(np.min(window,  axis=1))

    # ===== 6. Concatenate all features =====
    all_features = np.concatenate(extract_flat + pool_flat)
    all_features = np.nan_to_num(all_features)
    all_features = np.clip(all_features, -1e10, 1e10)

    return all_features


def compute_features_batch(X_batch):
    """
    Compute features for a batch of stocks.
    Input : X_batch shape (n_stocks, 9, 30)
    Output: features shape (n_stocks, n_feat)
    """
    features_list = []
    for i in range(X_batch.shape[0]):
        feat = compute_alphanet_features(
            X_batch[i],
            d=CONFIG['d_extract'],
            stride=CONFIG['stride_extract'],
            d_pool=CONFIG['d_pool'],
            stride_pool=CONFIG['stride_pool']
        )
        features_list.append(feat)
    return np.array(features_list)


# Dimension check
test_input    = np.random.randn(9, 30)
test_features = compute_alphanet_features(test_input)
print(f"Feature computation functions defined.")
print(f"   Single-stock feature dimension: {test_features.shape[0]}")


# ============================================================
# Cell 4: Build AlphaNet-v1 Keras Model
# ============================================================

def build_alphanet_v1(input_dim):
    """
    Build AlphaNet-v1:
      Input -> BN -> Dense(30, ReLU) -> Dropout -> Dense(1, linear)
    """
    inputs = Input(shape=(input_dim,), name='features_input')

    x = BatchNormalization(name='bn_input')(inputs)

    x = Dense(
        CONFIG['hidden_units'],
        activation='relu',
        kernel_initializer=TruncatedNormal(stddev=0.02),
        name='fc_hidden'
    )(x)
    x = Dropout(CONFIG['dropout_rate'], name='dropout')(x)

    outputs = Dense(
        1,
        activation='linear',
        kernel_initializer=TruncatedNormal(stddev=0.02),
        name='output'
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name='AlphaNet_v1')
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=CONFIG['learning_rate']),
        loss='mse'
    )
    return model


# Build once to show summary
test_model = build_alphanet_v1(test_features.shape[0])
test_model.summary()
print("\nAlphaNet-v1 model built successfully.")


# ============================================================
# Cell 5: Fast Data Preparation
# ============================================================

def get_all_trade_dates(start_date, end_date):
    """Get all trading days in the given range."""
    return get_trade_days(start_date=start_date, end_date=end_date)


def get_rebalance_dates(start_date, end_date, interval_days=10):
    """Get rebalance date list."""
    all_days    = get_all_trade_dates(start_date, end_date)
    rebal_dates = all_days[::interval_days]
    return rebal_dates


def batch_prepare_data(rebal_date, forward_days=10, lookback=30):
    """Efficiently prepare cross-section data for a single date."""
    all_hist_days   = get_trade_days(end_date=rebal_date,   count=lookback + 10)
    all_future_days = get_trade_days(start_date=rebal_date, count=forward_days + 5)

    if len(all_future_days) < forward_days + 1:
        return None, None, None, None

    data_start = str(all_hist_days[0])
    data_end   = str(all_future_days[min(forward_days + 2, len(all_future_days) - 1)])

    all_securities = get_all_securities('stock', date=rebal_date)
    stocks = [s for s in all_securities.index if s.startswith(('0', '3', '6'))]

    try:
        st_info = get_extras('is_st', stocks, start_date=rebal_date,
                             end_date=rebal_date, df=True)
        if len(st_info) > 0:
            non_st = st_info.columns[st_info.iloc[0] != True].tolist()
            stocks = [s for s in stocks if s in non_st]
    except:
        pass

    if len(stocks) > 3000:
        stocks = stocks[:3000]

    try:
        price_df = get_price(
            stocks, start_date=data_start, end_date=data_end,
            frequency='daily',
            fields=['open', 'close', 'high', 'low', 'volume', 'money'],
            skip_paused=False, fq='post', panel=False
        )
    except:
        return None, None, None, None

    rebal_ts = pd.Timestamp(rebal_date)

    X_list, Y_list, Y_raw_list, valid_stocks = [], [], [], []

    grouped = price_df.groupby('code')

    for stock, stk_df in grouped:
        try:
            stk = stk_df.set_index('time').sort_index()

            if len(stk) < lookback + forward_days:
                continue

            dates_before = stk.index[stk.index <= rebal_ts]
            if len(dates_before) < lookback:
                continue

            hist_end_loc = len(dates_before) - 1
            hist = stk.iloc[hist_end_loc - lookback + 1: hist_end_loc + 1]

            if len(hist) < lookback:
                continue

            img    = np.zeros((9, lookback))
            img[0] = hist['open'].values
            img[1] = hist['high'].values
            img[2] = hist['low'].values
            img[3] = hist['close'].values

            vwap   = hist['money'] / hist['volume']
            vwap   = vwap.replace([np.inf, -np.inf], np.nan).fillna(hist['close'])
            img[4] = vwap.values
            img[5] = hist['volume'].values
            img[6] = hist['close'].pct_change().fillna(0).values
            img[7] = hist['volume'].rolling(5).mean().fillna(hist['volume']).values
            img[8] = hist['volume'].rolling(10).mean().fillna(hist['volume']).values

            if np.isnan(img).any() or np.isinf(img).any():
                continue

            for row in range(9):
                row_std = np.std(img[row])
                if row_std > 1e-10:
                    img[row] = (img[row] - np.mean(img[row])) / row_std

            future = stk.iloc[hist_end_loc + 1: hist_end_loc + 1 + forward_days]
            if len(future) < max(1, forward_days - 2):
                continue

            base_price = stk['close'].iloc[hist_end_loc]
            if base_price < 0.01:
                continue

            future_ret = future['close'].iloc[-1] / base_price - 1
            if np.isnan(future_ret) or np.isinf(future_ret):
                continue

            X_list.append(img)
            Y_raw_list.append(future_ret)
            valid_stocks.append(stock)

        except:
            continue

    if len(X_list) < 100:
        return None, None, None, None

    X     = np.array(X_list)
    Y_raw = np.array(Y_raw_list)

    y_mean = np.mean(Y_raw)
    y_std  = np.std(Y_raw)
    Y      = (Y_raw - y_mean) / (y_std + 1e-8)

    return X, Y, Y_raw, valid_stocks

print("Fast data preparation function defined.")


# ============================================================
# Cell 6: Rolling Training & Prediction
# ============================================================
import time

def rolling_train_predict(rebalance_days=10, forward_days=10):
    backtest_start = CONFIG['backtest_start']
    end_date       = CONFIG['end_date']

    rebal_dates = get_rebalance_dates(backtest_start, end_date, rebalance_days)
    print(f"Total rebalance periods: {len(rebal_dates)}")

    train_months = CONFIG['retrain_months']

    all_predictions = []
    current_model   = None
    last_train_date = None

    for idx, rebal_date in enumerate(rebal_dates):
        rebal_str = str(rebal_date)

        need_train = False
        if current_model is None:
            need_train = True
        elif last_train_date is not None:
            months_diff = (
                (rebal_date.year  - last_train_date.year) * 12 +
                (rebal_date.month - last_train_date.month)
            )
            if months_diff >= train_months:
                need_train = True

        if need_train:
            print(f"\n[RETRAIN] [{idx+1}/{len(rebal_dates)}] @ {rebal_str}")

            t_total = time.time()

            t0 = time.time()
            print("   Fetching training dates ...", flush=True)
            train_days = get_trade_days(
                end_date=rebal_date, count=CONFIG['lookback_days']
            )
            print(f"   Training dates fetched, elapsed {time.time()-t0:.1f}s",
                  flush=True)

            sample_dates   = train_days[::5]
            n_train        = len(sample_dates) // 2
            train_sample_dates = sample_dates[:n_train]
            val_sample_dates   = sample_dates[n_train:]

            print(f"   Train cross-sections: {len(train_sample_dates)}, "
                  f"Val cross-sections: {len(val_sample_dates)}")

            # ===== Collect training data =====
            X_train_all, Y_train_all = [], []

            t0 = time.time()
            print(f"   Collecting training data "
                  f"({len(train_sample_dates[-20:])} cross-sections) ...",
                  flush=True)
            for si, sd in enumerate(train_sample_dates[-20:]):
                t1 = time.time()
                X, Y, _, _ = batch_prepare_data(sd, forward_days, lookback=30)
                t_data = time.time() - t1

                if X is not None:
                    t2       = time.time()
                    features = compute_features_batch(X)
                    t_feat   = time.time() - t2
                    X_train_all.append(features)
                    Y_train_all.append(Y)
                    print(f"      CS {si+1}: data {t_data:.1f}s, "
                          f"feat {t_feat:.1f}s, n_stocks {X.shape[0]}", flush=True)
                else:
                    print(f"      CS {si+1}: empty, skipped", flush=True)

            print(f"   Training data collected, elapsed {time.time()-t0:.1f}s",
                  flush=True)

            if len(X_train_all) == 0:
                print("   [WARN] Insufficient training data, skipping retrain.")
                continue

            X_train = np.vstack(X_train_all)
            Y_train = np.concatenate(Y_train_all)

            # ===== Collect validation data =====
            X_val_all, Y_val_all = [], []

            t0 = time.time()
            print(f"   Collecting validation data "
                  f"({len(val_sample_dates[-10:])} cross-sections) ...",
                  flush=True)
            for si, sd in enumerate(val_sample_dates[-10:]):
                t1 = time.time()
                X, Y, _, _ = batch_prepare_data(sd, forward_days, lookback=30)
                t_data = time.time() - t1

                if X is not None:
                    t2       = time.time()
                    features = compute_features_batch(X)
                    t_feat   = time.time() - t2
                    X_val_all.append(features)
                    Y_val_all.append(Y)
                    print(f"      CS {si+1}: data {t_data:.1f}s, "
                          f"feat {t_feat:.1f}s, n_stocks {X.shape[0]}", flush=True)
                else:
                    print(f"      CS {si+1}: empty, skipped", flush=True)

            print(f"   Validation data collected, elapsed {time.time()-t0:.1f}s",
                  flush=True)

            if len(X_val_all) == 0:
                X_val = X_train[-1000:]
                Y_val = Y_train[-1000:]
            else:
                X_val = np.vstack(X_val_all)
                Y_val = np.concatenate(Y_val_all)

            print(f"   Train samples: {X_train.shape[0]}, "
                  f"Val samples: {X_val.shape[0]}, "
                  f"Feature dim: {X_train.shape[1]}")

            # ===== Train models =====
            best_models = []
            for rep in range(CONFIG['n_repeats']):
                tf.set_random_seed(SEED + rep * 100)
                np.random.seed(SEED + rep * 100)

                model = build_alphanet_v1(X_train.shape[1])
                es    = EarlyStopping(
                    monitor='val_loss',
                    patience=CONFIG['early_stop_patience'],
                    verbose=0
                )

                t0 = time.time()
                print(f"   Training model {rep+1}/{CONFIG['n_repeats']} ...",
                      flush=True)

                model.fit(
                    X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=CONFIG['n_epochs'],
                    batch_size=CONFIG['batch_size'],
                    callbacks=[es],
                    verbose=0
                )

                print(f"   Model {rep+1} done, elapsed {time.time()-t0:.1f}s",
                      flush=True)
                best_models.append(model)

            current_model   = best_models
            last_train_date = rebal_date
            print(f"   All models trained, total elapsed "
                  f"{time.time()-t_total:.1f}s")

        # ========== Predict ==========
        # FIX 3: Guard against None model — prevents silent skip and
        # surfaces the root cause clearly in the log.
        if current_model is None:
            print(f"   [WARN] No trained model available at {rebal_str}, skipping.")
            continue

        t0 = time.time()
        X_pred, _, Y_raw, valid_stocks = batch_prepare_data(
            rebal_str, forward_days, lookback=30
        )

        if X_pred is None:
            continue

        features_pred = compute_features_batch(X_pred)

        preds = []
        for m in current_model:
            pred = m.predict(features_pred, verbose=0).flatten()
            preds.append(pred)

        avg_pred = np.mean(preds, axis=0)

        for i, stock in enumerate(valid_stocks):
            all_predictions.append({
                'date':       rebal_date,
                'stock':      stock,
                'pred':       avg_pred[i],
                'actual_ret': Y_raw[i]
            })

        print(f"   [OK] [{idx+1}/{len(rebal_dates)}] Predicted @ {rebal_str}, "
              f"stocks: {len(valid_stocks)}, elapsed: {time.time()-t0:.1f}s",
              flush=True)

    results_df = pd.DataFrame(all_predictions)
    print(f"\nRolling prediction complete. Total records: {len(results_df)}")
    return results_df, current_model          # ← 同时返回模型


# Run
print("=" * 60)
print("Starting AlphaNet-v1 Rolling Training & Prediction")
print("=" * 60)
results, current_model = rolling_train_predict(
    rebalance_days=CONFIG['rebalance_days'],
    forward_days=CONFIG['rebalance_days']
)


# ============================================================
# Cell 7: Factor IC Analysis
# ============================================================

from scipy import stats

def compute_rank_ic(results_df):
    """Compute cross-sectional RankIC for each rebalance date."""
    ic_records = []

    for date, group in results_df.groupby('date'):
        if len(group) < 30:
            continue

        ic, _ = stats.spearmanr(group['pred'], group['actual_ret'])

        ic_records.append({
            'date':     date,
            'RankIC':   ic,
            'n_stocks': len(group)
        })

    ic_df = pd.DataFrame(ic_records).set_index('date')
    return ic_df


def ic_analysis(ic_df):
    """Summarise IC statistics."""
    stats_dict = {
        'RankIC Mean':  ic_df['RankIC'].mean(),
        'RankIC Std':   ic_df['RankIC'].std(),
        'IC_IR':        (ic_df['RankIC'].mean() / ic_df['RankIC'].std()
                         if ic_df['RankIC'].std() > 0 else 0),
        'IC>0 Ratio':   (ic_df['RankIC'] > 0).mean(),
        'N Periods':    len(ic_df),
    }
    return stats_dict


# Compute IC
ic_df    = compute_rank_ic(results)
ic_stats = ic_analysis(ic_df)

print("\n" + "=" * 60)
print("AlphaNet-v1  Synthetic Factor  IC Analysis")
print("=" * 60)
print(f"  RankIC Mean  : {ic_stats['RankIC Mean']:.4f}  "
      f"({ic_stats['RankIC Mean']*100:.2f}%)")
print(f"  RankIC Std   : {ic_stats['RankIC Std']:.4f}")
print(f"  IC_IR        : {ic_stats['IC_IR']:.4f}")
print(f"  IC>0 Ratio   : {ic_stats['IC>0 Ratio']:.4f}  "
      f"({ic_stats['IC>0 Ratio']*100:.2f}%)")
print(f"  N Periods    : {ic_stats['N Periods']}")

# ---- Plot 1: IC time series + cumulative IC ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                         gridspec_kw={'height_ratios': [2, 1]})

ax1 = axes[0]
bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in ic_df['RankIC']]
ax1.bar(ic_df.index, ic_df['RankIC'], color=bar_colors, alpha=0.7, width=5)
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.axhline(
    y=ic_df['RankIC'].mean(), color='red', linestyle='--', linewidth=1.5,
    label=f'Mean = {ic_df["RankIC"].mean():.4f}'
)
ax1.set_title('AlphaNet-v1  Synthetic Factor  RankIC Time Series',
              fontsize=14, fontweight='bold')
ax1.set_ylabel('RankIC', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
cum_ic = ic_df['RankIC'].cumsum()
ax2.plot(cum_ic.index, cum_ic.values, color='#e74c3c', linewidth=2,
         label='Cumulative RankIC')
ax2.fill_between(cum_ic.index, 0, cum_ic.values, alpha=0.15, color='#e74c3c')
ax2.set_title('AlphaNet-v1  Synthetic Factor  Cumulative RankIC',
              fontsize=14, fontweight='bold')
ax2.set_ylabel('Cumulative RankIC', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ic_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Plot 2: IC distribution ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(ic_df['RankIC'], bins=30, color='#3498db', alpha=0.7, edgecolor='white')
ax.axvline(x=ic_df['RankIC'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {ic_df["RankIC"].mean():.4f}')
ax.axvline(x=0, color='black', linewidth=1)
ax.set_title('AlphaNet-v1  Synthetic Factor  RankIC Distribution',
             fontsize=14, fontweight='bold')
ax.set_xlabel('RankIC', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('ic_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Plot 3: IC summary table ----
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
table_data = [
    ['RankIC Mean', 'RankIC Std', 'IC_IR', 'IC>0 Ratio'],
    [f"{ic_stats['RankIC Mean']*100:.2f}%",
     f"{ic_stats['RankIC Std']*100:.2f}%",
     f"{ic_stats['IC_IR']:.2f}",
     f"{ic_stats['IC>0 Ratio']*100:.2f}%"]
]
tbl = ax.table(cellText=table_data, loc='center', cellLoc='center',
               colWidths=[0.2] * 4)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2)
for j in range(4):
    tbl[0, j].set_facecolor('#e74c3c')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
    tbl[1, j].set_facecolor('#f8f9fa')
ax.set_title('AlphaNet-v1  Synthetic Factor  IC Analysis Summary',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('ic_table.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nIC analysis plots saved.")


# ============================================================
# Cell 8: Factor Layered Test
# ============================================================

def layered_test(results_df, n_layers=5, cost=0.0):
    """
    Layered back-test.
    Returns: nav_df, ret_df, long_short_nav, long_short_ret
    """
    layer_returns = {f'Layer {i+1}': [] for i in range(n_layers)}
    layer_dates   = []

    dates = sorted(results_df['date'].unique())

    for i, date in enumerate(dates):
        group = results_df[results_df['date'] == date].copy()

        if len(group) < n_layers * 10:
            continue

        group      = group.sort_values('pred', ascending=False)
        n          = len(group)
        layer_size = n // n_layers

        for layer_idx in range(n_layers):
            start_idx   = layer_idx * layer_size
            end_idx     = start_idx + layer_size if layer_idx < n_layers - 1 else n
            layer_ret   = group.iloc[start_idx:end_idx]['actual_ret'].mean()

            if cost > 0 and i > 0:
                layer_ret -= cost * 2

            layer_returns[f'Layer {layer_idx+1}'].append(layer_ret)

        layer_dates.append(date)

    ret_df         = pd.DataFrame(layer_returns, index=layer_dates)
    nav_df         = (1 + ret_df).cumprod()
    long_short_ret = ret_df['Layer 1'] - ret_df[f'Layer {n_layers}']
    long_short_nav = (1 + long_short_ret).cumprod()

    return nav_df, ret_df, long_short_nav, long_short_ret


def compute_layer_stats(ret_df, long_short_ret, n_layers=5):
    """Compute annualised statistics for each layer."""
    stats_out    = {}
    rebal_days   = CONFIG['rebalance_days']
    annual_factor = 252 / rebal_days

    for layer_name in ret_df.columns:
        r       = ret_df[layer_name]
        ann_ret = (1 + r.mean()) ** annual_factor - 1
        ann_vol = r.std() * np.sqrt(annual_factor)
        stats_out[layer_name] = {
            'Ann. Excess Return': ann_ret,
            'Ann. Volatility':    ann_vol,
        }

    ls         = long_short_ret
    ls_ann_ret = (1 + ls.mean()) ** annual_factor - 1
    ls_ann_vol = ls.std() * np.sqrt(annual_factor)
    ls_sharpe  = ls_ann_ret / ls_ann_vol if ls_ann_vol > 0 else 0

    top_ret     = ret_df['Layer 1']
    top_ann_ret = (1 + top_ret.mean()) ** annual_factor - 1
    top_ann_vol = top_ret.std() * np.sqrt(annual_factor)
    top_ir      = top_ann_ret / top_ann_vol if top_ann_vol > 0 else 0
    top_win     = (top_ret > 0).mean()

    return {
        'Layer Stats':          stats_out,
        'L/S Ann. Return':      ls_ann_ret,
        'L/S Sharpe':           ls_sharpe,
        'TOP IR':               top_ir,
        'TOP Win Rate':         top_win,
        'TOP Ann. Excess Ret':  top_ann_ret,
    }


# Run both cost scenarios
nav_no_cost,   ret_no_cost,   ls_nav_no,   ls_ret_no   = layered_test(results, cost=0.0)
nav_with_cost, ret_with_cost, ls_nav_cost, ls_ret_cost = layered_test(results, cost=0.002)
stats_no_cost   = compute_layer_stats(ret_no_cost,   ls_ret_no)
stats_with_cost = compute_layer_stats(ret_with_cost, ls_ret_cost)

# ---- Plot 4: Layered NAV ----
layer_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
for i, col in enumerate(nav_no_cost.columns):
    ax.plot(nav_no_cost.index, nav_no_cost[col],
            color=layer_colors[i], linewidth=2, label=col)
ax.set_title('Layered NAV  (No Transaction Cost)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Net Asset Value', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

ax = axes[1]
for i, col in enumerate(nav_with_cost.columns):
    ax.plot(nav_with_cost.index, nav_with_cost[col],
            color=layer_colors[i], linewidth=2, label=col)
ax.set_title('Layered NAV  (Transaction Cost 0.2%)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Net Asset Value', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('layered_test.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Plot 5: Long-Short NAV ----
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ls_nav_no.index, ls_nav_no.values, color='#e74c3c', linewidth=2,
        label=f'No Cost  Ann.Ret={stats_no_cost["L/S Ann. Return"]*100:.1f}%')
ax.plot(ls_nav_cost.index, ls_nav_cost.values, color='#3498db', linewidth=2,
        label=f'Cost 0.2%  Ann.Ret={stats_with_cost["L/S Ann. Return"]*100:.1f}%')
ax.set_title('AlphaNet-v1  Long-Short Portfolio NAV',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Net Asset Value', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.savefig('long_short.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Print stats ----
for label, st in [('No Cost', stats_no_cost), ('Cost 0.2%', stats_with_cost)]:
    print(f"\n{'='*70}")
    print(f"Layered Test Results  ({label})")
    print(f"{'='*70}")
    for ln, ls in st['Layer Stats'].items():
        print(f"  {ln}: Ann. Excess Return = {ls['Ann. Excess Return']*100:.2f}%")
    print(f"  L/S Ann. Return : {st['L/S Ann. Return']*100:.2f}%")
    print(f"  L/S Sharpe      : {st['L/S Sharpe']:.2f}")
    print(f"  TOP IR          : {st['TOP IR']:.2f}")
    print(f"  TOP Win Rate    : {st['TOP Win Rate']*100:.2f}%")

# ---- Plot 6: Layer bar chart ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, st, title in [
    (axes[0], stats_no_cost,   'Ann. Excess Return by Layer  (No Cost)'),
    (axes[1], stats_with_cost, 'Ann. Excess Return by Layer  (Cost 0.2%)'),
]:
    lnames = list(st['Layer Stats'].keys())
    lrets  = [st['Layer Stats'][k]['Ann. Excess Return'] * 100 for k in lnames]
    bcolors = ['#e74c3c' if r > 0 else '#3498db' for r in lrets]
    bars   = ax.bar(lnames, lrets, color=bcolors, alpha=0.8, edgecolor='white')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Ann. Excess Return (%)', fontsize=11)
    for bar, val in zip(bars, lrets):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('layer_returns.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLayered test plots saved.")


# ============================================================
# Cell 9: TOP Portfolio Excess Return Analysis
# ============================================================

def compute_excess_returns(results_df, n_layers=5):
    """Compute TOP portfolio excess return vs equal-weight benchmark."""
    dates = sorted(results_df['date'].unique())

    top_returns = []

    for date in dates:
        group = results_df[results_df['date'] == date].copy()
        if len(group) < n_layers * 10:
            continue

        group      = group.sort_values('pred', ascending=False)
        n          = len(group)
        layer_size = n // n_layers

        top_ret   = group.iloc[:layer_size]['actual_ret'].mean()
        bench_ret = group['actual_ret'].mean()

        top_returns.append({
            'date':      date,
            'top_ret':   top_ret,
            'bench_ret': bench_ret
        })

    df = pd.DataFrame(top_returns).set_index('date')
    df['excess_ret'] = df['top_ret'] - df['bench_ret']
    df['cum_top']    = (1 + df['top_ret']).cumprod()
    df['cum_bench']  = (1 + df['bench_ret']).cumprod()
    df['cum_excess'] = (1 + df['excess_ret']).cumprod()

    cum_excess   = df['cum_excess']
    rolling_max  = cum_excess.expanding().max()
    df['excess_drawdown'] = (cum_excess - rolling_max) / rolling_max

    return df


excess_df = compute_excess_returns(results)

# ---- Plot 7: TOP vs Benchmark NAV ----
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(excess_df.index, excess_df['cum_top'],   color='#e74c3c',
        linewidth=2, label='TOP Portfolio')
ax.plot(excess_df.index, excess_df['cum_bench'], color='#95a5a6',
        linewidth=2, label='Benchmark (Equal Weight)')
ax.set_title('AlphaNet-v1  TOP Portfolio vs Benchmark NAV',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Net Asset Value', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.savefig('top_vs_benchmark.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Plot 8: Excess return & drawdown ----
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(excess_df.index, (excess_df['cum_excess'] - 1) * 100,
         color='#e74c3c', linewidth=2, label='Cumulative Excess Return')
ax1.set_ylabel('Cumulative Excess Return (%)', fontsize=12, color='#e74c3c')
ax1.tick_params(axis='y', labelcolor='#e74c3c')

ax2 = ax1.twinx()
ax2.fill_between(excess_df.index,
                 excess_df['excess_drawdown'] * 100, 0,
                 color='#3498db', alpha=0.3, label='Excess Return Drawdown')
ax2.set_ylabel('Excess Return Drawdown (%)', fontsize=12, color='#3498db')
ax2.tick_params(axis='y', labelcolor='#3498db')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
ax1.set_title('AlphaNet-v1  Excess Return & Drawdown',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.savefig('excess_return_drawdown.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Excess return stats ----
ann_factor    = 252 / CONFIG['rebalance_days']
excess_ann_ret = (1 + excess_df['excess_ret'].mean()) ** ann_factor - 1
excess_ann_vol = excess_df['excess_ret'].std() * np.sqrt(ann_factor)
excess_ir      = excess_ann_ret / excess_ann_vol if excess_ann_vol > 0 else 0
max_dd         = excess_df['excess_drawdown'].min()

print("\n" + "=" * 60)
print("TOP Portfolio Excess Return Statistics")
print("=" * 60)
print(f"  Ann. Excess Return   : {excess_ann_ret*100:.2f}%")
print(f"  Ann. Tracking Error  : {excess_ann_vol*100:.2f}%")
print(f"  Information Ratio    : {excess_ir:.2f}")
print(f"  Max Excess Drawdown  : {max_dd*100:.2f}%")
print(f"  Win Rate             : {(excess_df['excess_ret']>0).mean()*100:.2f}%")

print("\nExcess return analysis complete.")


# ============================================================
# Cell 10: Yearly Analysis & Dashboard
# ============================================================

def yearly_analysis(ic_df, excess_df, ret_df):
    """Compute per-year statistics."""
    ic_df.index     = pd.to_datetime(ic_df.index)
    excess_df.index = pd.to_datetime(excess_df.index)
    ret_df.index    = pd.to_datetime(ret_df.index)

    yearly_stats = []

    for year in sorted(set(ic_df.index.year)):
        year_ic     = ic_df[ic_df.index.year == year]
        year_excess = excess_df[excess_df.index.year == year]
        year_ret    = ret_df[ret_df.index.year == year]

        ic_mean = year_ic['RankIC'].mean() if len(year_ic) > 0 else np.nan
        ic_std  = year_ic['RankIC'].std()  if len(year_ic) > 1 else np.nan
        ic_ir   = (ic_mean / ic_std
                   if (ic_std is not None and ic_std > 0) else np.nan)

        row = {
            'Year':         year,
            'RankIC Mean':  ic_mean,
            'IC_IR':        ic_ir,
            'IC>0 Ratio':   (year_ic['RankIC'] > 0).mean() if len(year_ic) > 0 else np.nan,
        }

        if len(year_excess) > 0:
            af = 252 / CONFIG['rebalance_days']
            row['TOP Ann. Excess'] = (
                (1 + year_excess['excess_ret'].mean()) ** af - 1
            )

        if len(year_ret) > 0:
            af    = 252 / CONFIG['rebalance_days']
            top_r = year_ret['Layer 1']
            row['TOP Ann. Return'] = (1 + top_r.mean()) ** af - 1

        yearly_stats.append(row)

    return pd.DataFrame(yearly_stats)


yearly_df = yearly_analysis(ic_df, excess_df, ret_no_cost)

# ---- Plot 9: Yearly IC analysis ----
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax = axes[0, 0]
bars = ax.bar(
    yearly_df['Year'].astype(str),
    yearly_df['RankIC Mean'] * 100,
    color=['#e74c3c' if v > 0 else '#3498db'
           for v in yearly_df['RankIC Mean']],
    alpha=0.8, edgecolor='white'
)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_title('Yearly RankIC Mean', fontsize=13, fontweight='bold')
ax.set_ylabel('RankIC (%)', fontsize=11)
for bar, val in zip(bars, yearly_df['RankIC Mean']):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
ax.bar(
    yearly_df['Year'].astype(str),
    yearly_df['IC_IR'].fillna(0),
    color=['#e74c3c' if v > 0 else '#3498db'
           for v in yearly_df['IC_IR'].fillna(0)],
    alpha=0.8, edgecolor='white'
)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_title('Yearly IC_IR', fontsize=13, fontweight='bold')
ax.set_ylabel('IC_IR', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 0]
ax.bar(
    yearly_df['Year'].astype(str),
    yearly_df['IC>0 Ratio'] * 100,
    color='#2ecc71', alpha=0.8, edgecolor='white'
)
ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50%')
ax.set_title('Yearly IC>0 Ratio', fontsize=13, fontweight='bold')
ax.set_ylabel('IC>0 Ratio (%)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
if 'TOP Ann. Excess' in yearly_df.columns:
    ax.bar(
        yearly_df['Year'].astype(str),
        yearly_df['TOP Ann. Excess'].fillna(0) * 100,
        color=['#e74c3c' if v > 0 else '#3498db'
               for v in yearly_df['TOP Ann. Excess'].fillna(0)],
        alpha=0.8, edgecolor='white'
    )
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_title('Yearly TOP Portfolio Ann. Excess Return',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Ann. Excess Return (%)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('AlphaNet-v1  Yearly Performance Analysis',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('yearly_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Print yearly table ----
print("\n" + "=" * 80)
print("AlphaNet-v1  Yearly Statistics")
print("=" * 80)
print(yearly_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print("\nYearly analysis complete.")


# ============================================================
# Cell 11 替代方案：Permutation Feature Importance
# ============================================================
if current_model is None or len(current_model) == 0:
    raise RuntimeError(
        "current_model is None — 请确认 Cell 6 已正确运行并返回模型。"
    )
print(f"Model ensemble ready: {len(current_model)} models loaded.")

def permutation_importance(models, X_data, feature_names,
                           n_repeats=3, sample_size=500):
    """
    逐特征打乱输入，计算预测变化量作为重要性得分。
    models      : list of Keras models (ensemble)
    X_data      : np.ndarray, shape (n_samples, n_features)
    feature_names: list of str
    """
    np.random.seed(SEED)
    idx = np.random.choice(X_data.shape[0],
                           size=min(sample_size, X_data.shape[0]),
                           replace=False)
    X_sub = X_data[idx].copy()

    # 基准预测
    def ensemble_pred(X):
        preds = [m.predict(X, verbose=0).flatten() for m in models]
        return np.mean(preds, axis=0)

    baseline = ensemble_pred(X_sub)
    baseline_mse = np.mean(baseline ** 2)

    importances = np.zeros(X_sub.shape[1])

    for fi in range(X_sub.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_sub.copy()
            np.random.shuffle(X_perm[:, fi])          # 打乱第 fi 列
            perm_pred = ensemble_pred(X_perm)
            perm_mse  = np.mean(perm_pred ** 2)
            scores.append(abs(perm_mse - baseline_mse))
        importances[fi] = np.mean(scores)

        if fi % 100 == 0:
            print(f"   Progress: {fi}/{X_sub.shape[1]}", flush=True)

    # 整理结果
    perm_df = pd.DataFrame({
        'feature':    feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return perm_df


# ---- 获取最新截面数据 ----
print("Collecting sample data for Permutation Importance ...")
latest_date  = str(results['date'].max())
X_perm_raw, _, _, _ = batch_prepare_data(
    latest_date,
    forward_days=CONFIG['rebalance_days'],
    lookback=30
)

if X_perm_raw is not None:
    X_perm_feat = compute_features_batch(X_perm_raw)
    print(f"Sample shape: {X_perm_feat.shape}")

    print("Computing Permutation Importance ...")
    t0 = time.time()
    perm_df = permutation_importance(
        current_model,
        X_perm_feat,
        feature_names_list,
        n_repeats=3,
        sample_size=500
    )
    print(f"Done. Elapsed: {time.time()-t0:.1f}s")

    print("\nTop 20 features by Permutation Importance:")
    print(perm_df[['feature', 'importance']].head(20).to_string(index=False))

    # ---- 绘图：Top 20 重要特征 ----
    TOP_N  = 20
    top_df = perm_df.head(TOP_N).copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(TOP_N), top_df['importance'].values[::-1],
            color='#e74c3c', alpha=0.8)
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(top_df['feature'].values[::-1], fontsize=9)
    ax.set_xlabel('Permutation Importance Score', fontsize=12)
    ax.set_title(f'AlphaNet-v1  Top {TOP_N} Features by Permutation Importance',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('perm_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ---- 绘图：按算子类型聚合 ----
    perm_df['operator'] = perm_df['feature'].apply(
        lambda x: x.split('(')[0].split('[')[0]
    )
    op_importance = (perm_df.groupby('operator')['importance']
                     .sum()
                     .sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(op_importance.index, op_importance.values,
           color='#8e44ad', alpha=0.8, edgecolor='white')
    ax.set_title('AlphaNet-v1  Total Importance by Operator Type',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Operator', fontsize=12)
    ax.set_ylabel('Total Permutation Importance', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('perm_by_operator.png', dpi=150, bbox_inches='tight')
    plt.show()

    perm_df.to_csv('alphanet_factor_importance.csv', index=False)
    print("Factor importance saved to alphanet_factor_importance.csv")

    # 供 Cell 12 仪表盘使用（统一接口）
    shap_df = perm_df.rename(columns={'importance': 'mean_abs_shap'})
    shap_df['mean_shap'] = shap_df['mean_abs_shap']

else:
    shap_df = pd.DataFrame(columns=['feature', 'mean_abs_shap', 'mean_shap'])
    print("No data available, skipping importance analysis.")


# ============================================================
# Cell 12: Comprehensive Performance Dashboard
# ============================================================

fig = plt.figure(figsize=(20, 24))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# ---- Panel 1: RankIC time series ----
ax = fig.add_subplot(gs[0, :])
bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in ic_df['RankIC']]
ax.bar(ic_df.index, ic_df['RankIC'], color=bar_colors, alpha=0.7, width=5)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(
    y=ic_df['RankIC'].mean(), color='red', linestyle='--', linewidth=1.5,
    label=f'Mean = {ic_df["RankIC"].mean():.4f}'
)
ax.set_title('AlphaNet-v1  RankIC Time Series',
             fontsize=14, fontweight='bold')
ax.set_ylabel('RankIC', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ---- Panel 2: Layered NAV ----
ax = fig.add_subplot(gs[1, 0])
for i, col in enumerate(nav_no_cost.columns):
    ax.plot(nav_no_cost.index, nav_no_cost[col],
            color=layer_colors[i], linewidth=1.8, label=col)
ax.set_title('Layered NAV  (No Cost)', fontsize=13, fontweight='bold')
ax.set_ylabel('Net Asset Value', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ---- Panel 3: Long-Short NAV ----
ax = fig.add_subplot(gs[1, 1])
ax.plot(ls_nav_no.index, ls_nav_no.values,
        color='#8e44ad', linewidth=2, label='Long-Short (No Cost)')
ax.plot(ls_nav_cost.index, ls_nav_cost.values,
        color='#2980b9', linewidth=2, linestyle='--',
        label='Long-Short (Cost 0.2%)')
ax.axhline(y=1, color='black', linewidth=0.8, linestyle=':')
ax.set_title('Long-Short Portfolio NAV', fontsize=13, fontweight='bold')
ax.set_ylabel('Net Asset Value', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ---- Panel 4: TOP vs Benchmark ----
ax = fig.add_subplot(gs[2, 0])
ax.plot(excess_df.index, excess_df['cum_top'],
        color='#e74c3c', linewidth=2, label='TOP Portfolio')
ax.plot(excess_df.index, excess_df['cum_bench'],
        color='#95a5a6', linewidth=2, label='Benchmark')
ax.set_title('TOP Portfolio vs Benchmark', fontsize=13, fontweight='bold')
ax.set_ylabel('Net Asset Value', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ---- Panel 5: Cumulative excess return ----
ax = fig.add_subplot(gs[2, 1])
ax.plot(excess_df.index, (excess_df['cum_excess'] - 1) * 100,
        color='#e74c3c', linewidth=2, label='Cumulative Excess Return')
ax2 = ax.twinx()
ax2.fill_between(excess_df.index,
                 excess_df['excess_drawdown'] * 100, 0,
                 color='#3498db', alpha=0.25, label='Drawdown')
ax.set_title('Excess Return & Drawdown', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Excess Return (%)', fontsize=10, color='#e74c3c')
ax2.set_ylabel('Drawdown (%)', fontsize=10, color='#3498db')
ax.tick_params(axis='y', labelcolor='#e74c3c')
ax2.tick_params(axis='y', labelcolor='#3498db')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ---- Panel 6: Feature Importance (SHAP / Permutation / Gradient) ----
ax = fig.add_subplot(gs[3, 0])

if not shap_df.empty and 'mean_abs_shap' in shap_df.columns:
    top10 = shap_df.head(10)
    ax.barh(range(10), top10['mean_abs_shap'].values[::-1],
            color='#e74c3c', alpha=0.8)
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10['feature'].values[::-1], fontsize=8)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title('Top 10 Features by Importance', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Feature importance\nnot available',
            ha='center', va='center', fontsize=12,
            color='gray', transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('Top 10 Features by Importance', fontsize=13, fontweight='bold')


# ---- Panel 7: Yearly RankIC ----
ax = fig.add_subplot(gs[3, 1])
ax.bar(
    yearly_df['Year'].astype(str),
    yearly_df['RankIC Mean'] * 100,
    color=['#e74c3c' if v > 0 else '#3498db'
           for v in yearly_df['RankIC Mean']],
    alpha=0.8, edgecolor='white'
)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_title('Yearly RankIC Mean', fontsize=13, fontweight='bold')
ax.set_ylabel('RankIC (%)', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ---- Summary text box ----
summary_text = (
    f"AlphaNet-v1  Performance Summary\n"
    f"{'─'*38}\n"
    f"RankIC Mean     : {ic_stats['RankIC Mean']*100:.2f}%\n"
    f"IC_IR           : {ic_stats['IC_IR']:.2f}\n"
    f"IC>0 Ratio      : {ic_stats['IC>0 Ratio']*100:.1f}%\n"
    f"TOP Ann. Excess : {excess_ann_ret*100:.2f}%\n"
    f"Info. Ratio     : {excess_ir:.2f}\n"
    f"L/S Ann. Return : {stats_no_cost['L/S Ann. Return']*100:.2f}%\n"
    f"L/S Sharpe      : {stats_no_cost['L/S Sharpe']:.2f}\n"
    f"Max Excess DD   : {max_dd*100:.2f}%"
)
fig.text(
    0.5, -0.01, summary_text,
    ha='center', va='top', fontsize=11,
    bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa',
              edgecolor='#dee2e6', linewidth=1.5),
    family='monospace'
)

plt.suptitle('AlphaNet-v1  Comprehensive Performance Dashboard',
             fontsize=17, fontweight='bold', y=1.01)
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nComprehensive dashboard saved to dashboard.png")
print("\n" + "=" * 60)
print("All cells complete.")
print("=" * 60)
print(f"  Output files:")
print(f"    ic_analysis.png")
print(f"    ic_distribution.png")
print(f"    ic_table.png")
print(f"    layered_test.png")
print(f"    long_short.png")
print(f"    layer_returns.png")
print(f"    top_vs_benchmark.png")
print(f"    excess_return_drawdown.png")
print(f"    yearly_analysis.png")
print(f"    shap_abs.png")
print(f"    shap_signed.png")
print(f"    shap_by_operator.png")
print(f"    dashboard.png")
print(f"    alphanet_factor_importance.csv")
