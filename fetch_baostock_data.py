from pathlib import Path
import socket
import time

import baostock as bs
import baostock.common.context as bs_context
import numpy as np
import pandas as pd
from tqdm import tqdm


START_DATE = '2011-01-31'
END_DATE = '2026-05-31'
RAW_DIR = Path('data/raw/baostock_daily')
MERGED_PATH = Path('df_merged.csv')
FEATURE_PATH = Path('df_merged_fe.csv')
FAILED_PATH = Path('failed_codes.csv')
RETRY_TIMES = 3
SLEEP_SECONDS = 0.5
RETRY_SLEEP_SECONDS = 3
LOGIN_RETRY_TIMES = 6
MAX_RETRY_SLEEP_SECONDS = 60
SOCKET_TIMEOUT_SECONDS = 30
LIMIT_TOLERANCE_PCT = 0.5

# BaoStock adjustflag: 1=后复权, 2=前复权, 3=不复权

RAW_REQUIRED_COLUMNS = {
    'date',
    'code',
    'open',
    'high',
    'low',
    'close',
    'preclose',
    'volume',
    'amount',
    'turn',
    'tradestatus',
    'pctChg',
    'isST',
}


class BaoStockQueryError(RuntimeError):
    """Raised when BaoStock returns a query or network error."""


def is_no_data_error(error):
    return 'returned no data' in str(error).lower()


def get_missing_columns(columns, required_columns):
    return sorted(set(required_columns) - set(columns))


def ensure_required_columns(columns, required_columns, source_name):
    missing_columns = get_missing_columns(columns, required_columns)
    if missing_columns:
        raise ValueError(f'{source_name} missing columns: {missing_columns}')


def close_baostock_socket():
    """Close and clear BaoStock's shared socket after a broken connection."""
    current_socket = getattr(bs_context, 'default_socket', None)
    if current_socket is not None:
        try:
            current_socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            current_socket.close()
        except Exception:
            pass
    bs_context.default_socket = None


def retry_sleep(attempt):
    delay = min(RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_SLEEP_SECONDS)
    print(f'wait {delay}s before retry')
    time.sleep(delay)


def login_baostock():
    last_error = 'unknown login error'
    for attempt in range(1, LOGIN_RETRY_TIMES + 1):
        close_baostock_socket()
        try:
            lg = bs.login()
            if str(lg.error_code) == '0':
                current_socket = getattr(bs_context, 'default_socket', None)
                if current_socket is not None:
                    current_socket.settimeout(SOCKET_TIMEOUT_SECONDS)
                print('baostock login success')
                return
            last_error = lg.error_msg
        except Exception as e:
            last_error = f'{type(e).__name__}: {e}'

        print(f'baostock login failed ({attempt}/{LOGIN_RETRY_TIMES}): {last_error}')
        close_baostock_socket()
        if attempt < LOGIN_RETRY_TIMES:
            retry_sleep(attempt)

    raise RuntimeError(
        f'BaoStock login failed after {LOGIN_RETRY_TIMES} attempts: {last_error}'
    )


def logout_baostock():
    try:
        if getattr(bs_context, 'default_socket', None) is not None:
            bs.logout()
    except Exception as e:
        print('baostock logout warning:', e)
    finally:
        close_baostock_socket()


def reconnect_baostock():
    close_baostock_socket()
    login_baostock()


def get_all_a_stocks():
    print('get A stock pool...')
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            rs = bs.query_stock_basic()
            if str(rs.error_code) != '0':
                raise BaoStockQueryError(rs.error_msg)
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            df = pd.DataFrame(data_list, columns=rs.fields)
            df = df[df['type'] == '1']
            df = df[df['code'].str.startswith(('sh.6', 'sz.0', 'sz.3'))]
            return df['code'].tolist()
        except Exception as e:
            print(f'get stock pool failed ({attempt}/{RETRY_TIMES}): {e}')
            if attempt == RETRY_TIMES:
                raise
            retry_sleep(attempt)
            reconnect_baostock()

# preclose前一日收盘价，amount成交金额，turn换手率（当日成交量占流通股本的比例），tradestatus交易状态(1正常交易 0停牌)，pctChg涨跌幅
def query_stock_daily(code, adjustflag):
    fields = (
        'date,code,open,high,low,close,preclose,volume,amount,turn,'
        'tradestatus,pctChg,isST'
    )
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=START_DATE,
        end_date=END_DATE,
        frequency='d',
        adjustflag=adjustflag,
    )
    if str(rs.error_code) != '0':
        raise BaoStockQueryError(
            f'{code} adjustflag={adjustflag}: {rs.error_msg}'
        )
    data_list = []
    try:
        while rs.next():
            data_list.append(rs.get_row_data())
    except Exception as e:
        raise BaoStockQueryError(
            f'{code} adjustflag={adjustflag}: receive loop failed: '
            f'{type(e).__name__}: {e}'
        ) from e
    if not data_list:
        return None
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['code'] = df['code'].str.split('.').str[1] + '.' + df['code'].str.split('.').str[0].str.upper()
    df['date'] = pd.to_datetime(df['date'])
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['preclose'] = pd.to_numeric(df['preclose'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
    df['tradestatus'] = pd.to_numeric(df['tradestatus'], errors='coerce')
    df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
    df['isST'] = pd.to_numeric(df['isST'], errors='coerce')
    return df


def get_stock_daily(code):
    # 后复权
    adj_df = query_stock_daily(code, adjustflag='1')
    # 不复权
    raw_df = query_stock_daily(code, adjustflag='3')
    if adj_df is None or raw_df is None:
        return None
    adj_df = adj_df[['code', 'date', 'open', 'high', 'low', 'close', 'preclose', 'isST']]
    raw_df = raw_df[['code', 'date', 'close', 'preclose', 'volume', 'turn', 'amount', 'tradestatus', 'pctChg', 'isST']]
    df = adj_df.merge(raw_df, on=['code', 'date'], how='inner', suffixes=('', '_raw'))
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df['return'] = (df.groupby('code')['close'].pct_change() * 100)
    adjust_factor = safe_divide(df['close'], df['close_raw'])
    df['vwap'] = safe_divide(df['amount'], df['volume']) * adjust_factor
    df = df[['code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount','vwap', 'return', 'turn', 'preclose', 'tradestatus', 'pctChg', 'isST']]
    return df


def raw_file_has_required_columns(path):
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return RAW_REQUIRED_COLUMNS.issubset(columns)


def download_raw_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stocks = get_all_a_stocks()
    failed_codes = []
    print('A stock count:', len(stocks))
    for code in tqdm(stocks, desc='download stock daily'):
        output_path = RAW_DIR / f'{code}.csv'
        if output_path.exists() and raw_file_has_required_columns(output_path):
            continue
        success = False
        temp_path = output_path.with_name(f'{output_path.name}.part')
        for attempt in range(1, RETRY_TIMES + 1):
            try:
                df = get_stock_daily(code)
                if df is None:
                    raise BaoStockQueryError(f'{code} returned no data')
                ensure_required_columns(df.columns, RAW_REQUIRED_COLUMNS, f'{code} dataframe')
                df.to_csv(temp_path, index=False)
                if not raw_file_has_required_columns(temp_path):
                    raise ValueError(f'{temp_path} missing required columns after save')
                temp_path.replace(output_path)
                success = True
                break
            except Exception as e:
                print(f'{code} attempt {attempt}/{RETRY_TIMES} failed:', e)
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
                if is_no_data_error(e):
                    print(f'{code} has no historical daily data, skip remaining retries')
                    break
                if attempt < RETRY_TIMES:
                    retry_sleep(attempt)
                    try:
                        print(f'{code} reconnecting...')
                        reconnect_baostock()
                    except Exception as reconnect_error:
                        print('reconnect failed:', reconnect_error)
        if not success:
            print(code, f'failed after {RETRY_TIMES} retries')
            failed_codes.append(code)
        time.sleep(SLEEP_SECONDS)
    pd.DataFrame({'code': failed_codes}).to_csv(FAILED_PATH, index=False)
    print('failed stock count:', len(failed_codes))
    print('save done:', FAILED_PATH)


def merge_raw_data():
    all_files = sorted(RAW_DIR.glob('*.csv'))
    all_data = []
    invalid_files = []
    for file in tqdm(all_files, desc='merge raw data'):
        if not raw_file_has_required_columns(file):
            invalid_files.append(str(file))
            continue
        df = pd.read_csv(file)
        all_data.append(df)
    if invalid_files:
        preview = invalid_files[:5]
        raise ValueError(
            f'{len(invalid_files)} raw files are missing required columns (including amount). '
            f'Examples: {preview}'
        )
    if not all_data:
        raise Exception('no data downloaded')
    result = pd.concat(all_data, ignore_index=True)
    ensure_required_columns(result.columns, RAW_REQUIRED_COLUMNS, 'merged raw data')
    return result


def get_limit_threshold(code, date, is_st):
    threshold = pd.Series(10.0, index=code.index, dtype=float)
    threshold.loc[is_st == 1] = 5.0
    chinext = code.str.startswith(('300', '301')) & (date >= pd.Timestamp('2020-08-24'))
    star = code.str.startswith('688')
    beijing = code.str.endswith('.BJ') & (date >= pd.Timestamp('2021-11-15'))
    threshold.loc[chinext | star] = 20.0
    threshold.loc[beijing] = 30.0
    return threshold


def clean_data(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(['code', 'date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    numeric_cols = [
        'open', 'high', 'low', 'close', 'preclose',
        'volume', 'amount', 'turn', 'tradestatus', 'pctChg', 'isST'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['volume'] = df['volume'].fillna(0.0)
    df['amount'] = df['amount'].fillna(0.0)
    df['turn'] = df['turn'].fillna(0.0)
    df['pctChg'] = df['pctChg'].fillna(0.0)
    df['tradestatus'] = df['tradestatus'].fillna(0).astype('int8')
    df['isST'] = df['isST'].fillna(0).astype('int8')

    df['return'] = df.groupby('code')['close'].pct_change().mul(100).fillna(0.0)
    df['vwap'] = safe_divide(df['amount'], df['volume'])
    df['vwap'] = pd.Series(df['vwap'], index=df.index).replace([np.inf, -np.inf], np.nan)
    df['vwap'] = df['vwap'].fillna(df['close'])
    df.loc[df['tradestatus'] == 0, 'vwap'] = df.loc[df['tradestatus'] == 0, 'close']

    df['limit_pct'] = get_limit_threshold(
        df['code'],
        df['date'],
        df['isST'],
    )
    pct_chg = pd.to_numeric(df['pctChg'], errors='coerce')
    df['is_limit_up'] = (
        pct_chg >= df['limit_pct'] - LIMIT_TOLERANCE_PCT
    ).astype('int8')
    df['is_limit_down'] = (
        pct_chg <= -df['limit_pct'] + LIMIT_TOLERANCE_PCT
    ).astype('int8')
    df['is_suspended'] = (df['tradestatus'] == 0).astype('int8')
    df['is_tradable'] = (df['tradestatus'] == 1).astype('int8')

    # can_buy: 可买入条件：可交易、非ST、非涨停
    # can_sell: 可卖出条件：可交易、非跌停
    df['can_buy'] = (
        (df['is_tradable'] == 1) &
        (df['isST'] != 1) &
        (df['is_limit_up'] == 0)
    ).astype('int8')
    df['can_sell'] = (
        (df['is_tradable'] == 1) &
        (df['is_limit_down'] == 0)
    ).astype('int8')
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


def safe_divide(a, b):
    return np.where(b == 0, np.nan, a / b)


def add_ratio_features(df):
    df = df.copy()
    df['close_turn'] = safe_divide(df['close'], df['turn'])
    df['open_turn'] = safe_divide(df['open'], df['turn'])
    df['volume_low'] = safe_divide(df['volume'], df['low'])
    df['vwap_high'] = safe_divide(df['vwap'], df['high'])
    df['low_high'] = safe_divide(df['low'], df['high'])
    df['vwap_close'] = safe_divide(df['vwap'], df['close'])
    df['turn_volume'] = safe_divide(df['turn'], df['volume'])
    ratio_cols = [
        'close_turn', 'open_turn', 'volume_low', 'vwap_high',
        'low_high', 'vwap_close', 'turn_volume'
    ]
    df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.reset_index(drop=True)
    return df


def main():
    login_baostock()
    try:
        download_raw_data()
        df_merged = merge_raw_data()
        df_merged = clean_data(df_merged)
        df_merged.to_csv(MERGED_PATH, index=False)
        df_merged_fe = add_ratio_features(df_merged)
        df_merged_fe.to_csv(FEATURE_PATH, index=False)
        print(df_merged.head())
        print('raw data shape:', df_merged.shape)
        print('feature data shape:', df_merged_fe.shape)
        print('save done:', MERGED_PATH)
        print('save done:', FEATURE_PATH)
    finally:
        logout_baostock()


if __name__ == '__main__':
    main()
