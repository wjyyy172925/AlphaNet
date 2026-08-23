from pathlib import Path
import time

import baostock as bs
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
LIMIT_TOLERANCE_PCT = 0.2

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


def login_baostock():
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(lg.error_msg)
    print('baostock login success')


def logout_baostock():
    bs.logout()


def reconnect_baostock():
    try:
        logout_baostock()
    except Exception:
        pass
    time.sleep(RETRY_SLEEP_SECONDS)
    login_baostock()


def get_all_a_stocks():
    print('get A stock pool...')
    rs = bs.query_stock_basic()
    if rs.error_code != '0':
        raise Exception(rs.error_msg)
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)
    df = df[df['type'] == '1']
    df = df[df['code'].str.startswith(('sh.6', 'sz.0', 'sz.3'))]
    return df['code'].tolist()


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
    if rs.error_code != '0':
        print(code, adjustflag, 'failed:', rs.error_msg)
        return None
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
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
    adj_df = query_stock_daily(code, adjustflag='2')
    raw_df = query_stock_daily(code, adjustflag='3')
    if adj_df is None or raw_df is None:
        return None
    adj_df = adj_df[['code', 'date', 'open', 'high', 'low', 'close', 'isST']]
    raw_df = raw_df[
        [
            'code',
            'date',
            'close',
            'preclose',
            'volume',
            'turn',
            'amount',
            'tradestatus',
            'pctChg',
            'isST',
        ]
    ]
    df = adj_df.merge(raw_df, on=['code', 'date'], how='inner', suffixes=('', '_raw'))
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df['return'] = df['close'].pct_change() * 100
    adjust_factor = safe_divide(df['close'], df['close_raw'])
    df['volumn'] = df['volume']
    df['vwap'] = safe_divide(df['amount'], df['volumn']) * adjust_factor
    df = df[
        [
            'code',
            'date',
            'open',
            'close',
            'high',
            'low',
            'volumn',
            'vwap',
            'return',
            'turn',
            'preclose',
            'tradestatus',
            'pctChg',
            'isST',
        ]
    ]
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
        for i in range(RETRY_TIMES):
            try:
                df = get_stock_daily(code)
                if df is not None:
                    df.to_csv(output_path, index=False)
                    success = True
                    break
            except Exception as e:
                print(code, e)
            print(code, 'retry', i + 1)
            reconnect_baostock()
        if not success:
            print(code, 'failed after 3 retries')
            failed_codes.append(code)
        time.sleep(SLEEP_SECONDS)
    pd.DataFrame({'code': failed_codes}).to_csv(FAILED_PATH, index=False)
    print('failed stock count:', len(failed_codes))
    print('save done:', FAILED_PATH)


def merge_raw_data():
    all_files = sorted(RAW_DIR.glob('*.csv'))
    all_data = []
    for file in tqdm(all_files, desc='merge raw data'):
        df = pd.read_csv(file)
        all_data.append(df)
    if not all_data:
        raise Exception('no data downloaded')
    result = pd.concat(all_data, ignore_index=True)
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

    # 买入端避开 ST、停牌和涨停；卖出端保留跌停标记，供回测处理无法卖出的持仓。
    df['can_buy'] = (
        (df['is_tradable'] == 1) &
        (df['isST'] != 1) &
        (df['is_limit_up'] == 0)
    ).astype('int8')
    df['can_sell'] = (
        (df['is_tradable'] == 1) &
        (df['is_limit_down'] == 0)
    ).astype('int8')

    df = df.dropna().reset_index(drop=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


def safe_divide(a, b):
    return np.where(b == 0, np.nan, a / b)


def add_ratio_features(df):
    df = df.copy()
    df['close_turn'] = safe_divide(df['close'], df['turn'])
    df['open_turn'] = safe_divide(df['open'], df['turn'])
    df['volumn_low'] = safe_divide(df['volumn'], df['low'])
    df['vwap_high'] = safe_divide(df['vwap'], df['high'])
    df['low_high'] = safe_divide(df['low'], df['high'])
    df['vwap_close'] = safe_divide(df['vwap'], df['close'])
    df['turn_volumn'] = safe_divide(df['turn'], df['volumn'])
    df = df.dropna().reset_index(drop=True)
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
