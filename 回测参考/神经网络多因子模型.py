"""
神经网络多因子模型 - 研报完整复现
来源：江海证券《股票多因子系列（四）：神经网络多因子模型初探》

修正清单：
① 滚动训练：2年训练 + 6月验证 + 1年测试，严格时序隔离（内存优化）
② 全A股票池：剔除ST/*ST、停牌、上市不足6个月次新股
③ 因子提取：基础/情绪/成长/动量/每股指标/质量/风险/风格/技术指标 9大类
④ 三步因子预处理：MAD去极值 → 行业市值中性化 → 对称正交化
⑤ 10分组标签：收益率截面10等分位，标签0~9
⑥ 内存优化：mini-batch训练 / 分块推理 / 主动GC / LSTM-GRU隐层减半
⑦ 【新修复】列对齐：训练/验证/测试集统一对齐到训练集公共列，缺失补0
"""

# ============================================================
# 0. 依赖导入与全局配置
# ============================================================
import gc
import pandas as pd
import numpy as np
from jqdata import *
from jqfactor import get_factor_values
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# ---------- 全局超参数 ----------
SEED         = 666
NUM_CLASSES  = 10
TRAIN_MONTHS = 36
VAL_MONTHS   = 12
TEST_MONTHS  = 24
EPOCHS       = 100
PATIENCE     = 10
LR           = 0.001
BATCH_SIZE   = 128
INFER_BATCH  = 512
COST         = 0.002

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _hide_spines(ax, spines=('top', 'right')):
    """兼容旧版 matplotlib，逐个隐藏边框。"""
    for sp in spines:
        ax.spines[sp].set_visible(False)

# ============================================================
# PART 1 ── 股票池
# ============================================================

def get_universe(date_str: str) -> list:
    all_info   = get_all_securities('stock', date=date_str)
    all_stocks = all_info.index.tolist()

    cutoff     = pd.Timestamp(date_str) - pd.DateOffset(months=6)
    all_stocks = [
        s for s in all_stocks
        if pd.Timestamp(all_info.loc[s, 'start_date']) <= cutoff
    ]

    try:
        st_mask = get_extras('is_st', all_stocks,
                             start_date=date_str, end_date=date_str, df=True)
        if not st_mask.empty:
            st_list    = st_mask.columns[st_mask.iloc[0].astype(bool)].tolist()
            all_stocks = [s for s in all_stocks if s not in st_list]
    except Exception:
        pass

    try:
        paused_mask = get_extras('paused_days', all_stocks,
                                 start_date=date_str, end_date=date_str, df=True)
        if not paused_mask.empty:
            paused_list = paused_mask.columns[paused_mask.iloc[0] > 0].tolist()
            all_stocks  = [s for s in all_stocks if s not in paused_list]
    except Exception:
        pass

    return all_stocks


# ============================================================
# PART 2 ── 因子提取（9大类）
# ============================================================

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _safe_query(q, date_str):
    try:
        df = get_fundamentals(q, date=date_str)
        if 'code' in df.columns:
            df = df.set_index('code')
        return df
    except Exception:
        return pd.DataFrame()


# ---- A. 基础类 ----------------------------------------
def _factor_basic(stocks, date_str):
    out = pd.DataFrame(index=stocks, dtype=float)
    try:
        q = query(
            income.code,
            income.net_profit,
            income.total_profit,
            income.operating_profit,
            income.operating_revenue,
            income.operating_cost,
            income.np_parent_company_owners,
            income.financial_expense,
            balance.surplus_reserve_fund,
            balance.retained_profit,
            valuation.ps_ratio,
        ).filter(income.code.in_(stocks))
        fd = _safe_query(q, date_str)

        gross_profit = fd['operating_revenue'] - fd['operating_cost']
        ebit         = fd['total_profit'] + fd['financial_expense'].fillna(0)

        out['EBIT']                         = ebit
        out['EBITDA']                       = ebit
        out['gross_profit_ttm']             = gross_profit
        out['net_profit_ttm']               = fd['net_profit']
        out['np_parent_company_owners_ttm'] = fd['np_parent_company_owners']
        out['OperateNetIncome']             = fd['operating_profit']
        out['operating_profit_ttm']         = fd['operating_profit']
        out['retained_earnings']            = (
            fd['surplus_reserve_fund'].fillna(0)
            + fd['retained_profit'].fillna(0)
        )
        out['sales_to_price_ratio']         = 1.0 / fd['ps_ratio'].replace(0, np.nan)
        out['total_profit_ttm']             = fd['total_profit']

    except Exception as e:
        print(f"    [基础类] {e}")
    return out


# ---- B. 情绪类 ----------------------------------------
def _factor_sentiment(stocks, date_str):
    out = pd.DataFrame(index=stocks, dtype=float)
    try:
        price_panel = get_price(
            stocks, end_date=date_str, frequency='daily',
            fields=['open', 'high', 'low', 'close', 'volume'],
            count=250, panel=False)

        q_circ    = query(valuation.code, valuation.circulating_cap).filter(
            valuation.code.in_(stocks))
        circ_data = _safe_query(q_circ, date_str)

        for stk in stocks:
            try:
                d   = price_panel[price_panel['code'] == stk].reset_index(drop=True)
                if len(d) < 30:
                    continue
                c   = d['close'].values
                h   = d['high'].values
                lo  = d['low'].values
                op  = d['open'].values
                vol = d['volume'].values.astype(float)
                n   = len(c)

                tr = np.maximum(h[1:] - lo[1:],
                     np.maximum(np.abs(h[1:] - c[:-1]),
                                np.abs(lo[1:] - c[:-1])))
                out.loc[stk, 'ATR14'] = float(np.mean(tr[-14:])) if len(tr) >= 14 else np.nan

                if n >= 27:
                    ar_d = float(np.sum(op[-26:] - lo[-26:]))
                    br_d = float(np.sum(np.maximum(c[-27:-1] - lo[-26:], 0)))
                    ar   = float(np.sum(h[-26:] - op[-26:])) / ar_d if ar_d != 0 else np.nan
                    br   = float(np.sum(np.maximum(h[-26:] - c[-27:-1], 0))) / br_d if br_d != 0 else np.nan
                    out.loc[stk, 'BR']   = br
                    out.loc[stk, 'ARBR'] = ar - br if not (np.isnan(ar) or np.isnan(br)) else np.nan

                if n >= 13:
                    out.loc[stk, 'PSY'] = float(np.sum(c[-12:] > c[-13:-1])) / 12 * 100

                if n >= 27:
                    up_v = float(np.sum(vol[-26:][c[-26:] > c[-27:-1]]))
                    dn_v = float(np.sum(vol[-26:][c[-26:] < c[-27:-1]]))
                    out.loc[stk, 'VR'] = up_v / dn_v if dn_v != 0 else np.nan

                vol_s  = pd.Series(vol)
                vema12 = _ema(vol_s, 12)
                vema26 = _ema(vol_s, 26)
                out.loc[stk, 'VEMA26'] = float(vema26.iloc[-1])
                out.loc[stk, 'VDEA']   = float(_ema(vema12 - vema26, 9).iloc[-1])

                hl   = h - lo
                wvad = np.where(hl > 0, (c - op) / hl * vol, 0.0)
                out.loc[stk, 'MAWVAD'] = float(np.mean(wvad[-6:])) if n >= 6 else np.nan

                if stk in circ_data.index:
                    circ = float(circ_data.loc[stk, 'circulating_cap']) * 1e4
                    if circ > 0:
                        to = vol / circ
                        if n >= 120:
                            out.loc[stk, 'VOL120'] = float(np.mean(to[-120:]))
                        if n >= 240:
                            out.loc[stk, 'VOL240'] = float(np.mean(to[-240:]))
            except Exception:
                pass
    except Exception as e:
        print(f"    [情绪类] {e}")
    return out


# ---- C. 成长类 ----------------------------------------
_GROWTH_NAMES = [
    'net_profit_growth_rate',
    'operating_revenue_growth_rate',
    'total_asset_growth_rate',
    'net_asset_growth_rate',
    'net_operate_cashflow_growth_rate',
    'total_profit_growth_rate',
    'np_parent_company_owners_growth_rate',
    'financing_cash_growth_rate',
]

def _factor_growth(stocks, date_str):
    out = pd.DataFrame(index=stocks, dtype=float)
    try:
        vals = get_factor_values(stocks, _GROWTH_NAMES, end_date=date_str, count=1)
        for fn in _GROWTH_NAMES:
            if fn in vals and not vals[fn].empty:
                out[fn] = vals[fn].iloc[0]
    except Exception as e:
        print(f"    [成长类] {e}")

    try:
        q  = query(valuation.code, valuation.pe_ratio).filter(
            valuation.code.in_(stocks))
        pe = _safe_query(q, date_str)['pe_ratio']
        if 'net_profit_growth_rate' in out.columns:
            out['PEG'] = pe / (out['net_profit_growth_rate'] * 100).replace(0, np.nan)
    except Exception:
        pass
    return out


# ---- D. 动量类 ----------------------------------------
def _factor_momentum(stocks, date_str):
    out = pd.DataFrame(index=stocks, dtype=float)
    try:
        pp  = get_price(stocks, end_date=date_str, frequency='daily',
                        fields=['close', 'high', 'low', 'volume'], count=260, panel=True)
        cls = pp['close']
        his = pp['high']
        los = pp['low']

        for stk in stocks:
            if stk not in cls.columns:
                continue
            try:
                p  = cls[stk].dropna()
                hi = his[stk].dropna()
                lo = los[stk].dropna()
                n  = len(p)
                cv = float(p.iloc[-1])

                if n >= 22:
                    out.loc[stk, 'Rank1M'] = cv / float(p.iloc[-22]) - 1
                if n >= 250:
                    out.loc[stk, 'Price1Y'] = cv / float(p.iloc[-250:].mean()) - 1
                if n >= 121:
                    out.loc[stk, 'ROC120'] = (cv - float(p.iloc[-121])) / float(p.iloc[-121]) * 100
                if n >= 250:
                    w250 = p.iloc[-250:].values
                    out.loc[stk, 'fifty_two_week_close_rank'] = (
                        float(np.searchsorted(np.sort(w250), cv)) / 250
                    )
                if n >= 24:
                    bbi = (p.rolling(3).mean() + p.rolling(6).mean() +
                           p.rolling(12).mean() + p.rolling(24).mean()) / 4
                    out.loc[stk, 'BBIC'] = float(bbi.iloc[-1]) / cv
                if n >= 26:
                    wh = hi.iloc[-26:].values
                    wl = lo.iloc[-26:].values
                    out.loc[stk, 'arron_up_25']  = (25 - int(np.argmax(wh[::-1]))) / 25 * 100
                    out.loc[stk, 'arron_down_25'] = (25 - int(np.argmax(wl[::-1]))) / 25 * 100
                if n >= 14:
                    ema13 = _ema(p, 13)
                    out.loc[stk, 'bear_power'] = (float(lo.iloc[-1]) - float(ema13.iloc[-1])) / cv
                if n >= 34:
                    hl   = hi - lo
                    e9   = _ema(hl, 9)
                    e9e9 = _ema(e9, 9)
                    out.loc[stk, 'MASS'] = float(
                        (e9 / e9e9.replace(0, np.nan)).rolling(25).sum().iloc[-1]
                    )
                if n >= 2 and stk in pp['volume'].columns:
                    vv  = float(pp['volume'][stk].dropna().iloc[-1])
                    ret = (cv - float(p.iloc[-2])) / float(p.iloc[-2])
                    out.loc[stk, 'single_day_VPT'] = ret * vv
            except Exception:
                pass
    except Exception as e:
        print(f"    [动量类] {e}")
    return out


# ---- E. 每股指标类 ----------------------------------------
def _factor_pershare(stocks, date_str):
    out        = pd.DataFrame(index=stocks, dtype=float)
    cap_shares = None

    try:
        q = query(
            income.code,
            balance.total_owner_equities,
            balance.surplus_reserve_fund,
            balance.retained_profit,
            valuation.capitalization,
            income.net_profit,
            income.operating_profit,
            income.operating_revenue,
            cash_flow.net_operate_cash_flow,
        ).filter(income.code.in_(stocks))
        fd = _safe_query(q, date_str)

        cap_shares = (fd['capitalization'] * 10000).replace(0, np.nan)

        out['net_asset_per_share']             = fd['total_owner_equities'] / cap_shares
        out['retained_profit_per_share']       = fd['retained_profit'].fillna(0) / cap_shares
        out['surplus_reserve_fund_per_share']  = fd['surplus_reserve_fund'].fillna(0) / cap_shares
        out['retained_earnings_per_share']     = (
            (fd['surplus_reserve_fund'].fillna(0)
             + fd['retained_profit'].fillna(0)) / cap_shares
        )
        out['eps_ttm']                         = fd['net_profit'] / cap_shares
        out['operating_profit_per_share']      = fd['operating_profit'] / cap_shares
        out['operating_revenue_per_share']     = fd['operating_revenue'] / cap_shares
        out['net_operate_cash_flow_per_share'] = fd['net_operate_cash_flow'] / cap_shares

    except Exception as e:
        print(f"    [每股指标-主] {e}")
        return out

    _CASH_CANDIDATES = [
        ('balance',    'cash_equivalents'),
        ('balance',    'money_funds'),
        ('cash_flow',  'cash_and_cash_equivalents'),
    ]
    for module_name, field_name in _CASH_CANDIDATES:
        try:
            module  = balance if module_name == 'balance' else cash_flow
            field   = getattr(module, field_name)
            q_cash  = query(income.code, field).filter(income.code.in_(stocks))
            fd_cash = _safe_query(q_cash, date_str)
            if field_name in fd_cash.columns and cap_shares is not None:
                out['cash_and_equivalents_per_share'] = (
                    fd_cash[field_name].fillna(0) / cap_shares
                )
                break
        except Exception:
            continue

    return out


# ---- F. 质量类 ----------------------------------------
def _factor_quality(stocks, date_str):
    out = pd.DataFrame(index=stocks, dtype=float)
    try:
        q = query(
            income.code,
            income.operating_profit,
            income.total_profit,
            income.financial_expense,
            income.operating_revenue,
            income.operating_cost,
            balance.total_owner_equities,
            balance.total_assets,
            balance.total_liability,
            balance.shortterm_loan,
            balance.longterm_loan,
            balance.bonds_payable,
            balance.total_current_assets,
            balance.total_current_liability,
            balance.intangible_assets,
            cash_flow.net_operate_cash_flow,
        ).filter(income.code.in_(stocks))
        fd = _safe_query(q, date_str)

        ebit = fd['total_profit'] + fd['financial_expense'].fillna(0)
        interest_bearing_debt = (
            fd['shortterm_loan'].fillna(0)
            + fd['longterm_loan'].fillna(0)
            + fd['bonds_payable'].fillna(0)
        )
        invested_capital = (
            fd['total_owner_equities'].fillna(0) + interest_bearing_debt
        ).replace(0, np.nan)

        total_assets = fd['total_assets'].replace(0, np.nan)
        equity       = fd['total_owner_equities'].replace(0, np.nan)
        rev          = fd['operating_revenue'].replace(0, np.nan)
        cur_liab     = fd['total_current_liability'].replace(0, np.nan)
        gross        = fd['operating_revenue'] - fd['operating_cost']

        out['roic_ttm']                       = ebit / invested_capital
        out['roa_ttm']                        = fd['operating_profit'] / total_assets
        out['roe_ttm']                        = fd['operating_profit'] / equity
        out['debt_to_asset_ratio']            = fd['total_liability'] / total_assets
        out['gross_income_ratio']             = gross / rev
        out['net_operate_cash_flow_to_asset'] = fd['net_operate_cash_flow'] / total_assets
        out['current_ratio']                  = fd['total_current_assets'] / cur_liab
        out['intangible_asset_ratio']         = fd['intangible_assets'].fillna(0) / total_assets

    except Exception as e:
        print(f"    [质量类] {e}")
    return out


# ---- G. 风险类 ----------------------------------------
def _factor_risk(stocks, date_str):
    out = pd.DataFrame(index=stocks, dtype=float)
    try:
        price_mat = get_price(stocks, end_date=date_str, frequency='daily',
                              fields=['close'], count=130, panel=True)['close']
        rets_mat  = price_mat.pct_change().dropna()

        for stk in stocks:
            if stk not in rets_mat.columns:
                continue
            r = rets_mat[stk].dropna()
            for w, suf in [(20, '20'), (60, '60'), (120, '120')]:
                if len(r) >= w:
                    rw  = r.iloc[-w:]
                    std = float(rw.std())
                    out.loc[stk, f'Kurtosis{suf}']      = float(rw.kurtosis())
                    out.loc[stk, f'Skewness{suf}']      = float(rw.skew())
                    if std > 0:
                        out.loc[stk, f'sharpe_ratio_{suf}'] = (
                            float(rw.mean()) / std * np.sqrt(252)
                        )
            if len(r) >= 120:
                out.loc[stk, 'Variance120'] = float(r.iloc[-120:].var()) * 252
    except Exception as e:
        print(f"    [风险类] {e}")
    return out


# ---- H. 风格类 ----------------------------------------
_STYLE_NAMES = [
    'book_to_price_ratio', 'cash_earnings_to_price_ratio',
    'earnings_to_price_ratio', 'earnings_yield', 'growth',
    'earnings_growth', 'market_leverage',
    'predicted_earnings_to_price_ratio',
    'sales_growth', 'long_term_predicted_earnings_growth',
]

def _factor_style(stocks, date_str):
    out = pd.DataFrame(index=stocks, dtype=float)
    try:
        vals = get_factor_values(stocks, _STYLE_NAMES, end_date=date_str, count=1)
        for fn in _STYLE_NAMES:
            if fn in vals and not vals[fn].empty:
                out[fn] = vals[fn].iloc[0]
    except Exception as e:
        print(f"    [风格类] {e}")
    return out


# ---- I. 技术指标类 ----------------------------------------
def _factor_technical(stocks, date_str):
    out = pd.DataFrame(index=stocks, dtype=float)
    try:
        pp = get_price(stocks, end_date=date_str, frequency='daily',
                       fields=['close'], count=70, panel=True)['close']
        for stk in stocks:
            if stk not in pp.columns:
                continue
            p  = pp[stk].dropna()
            n  = len(p)
            cv = float(p.iloc[-1])
            if cv == 0 or n < 5:
                continue
            for sp, fn in [(5, 'EMA5'), (10, 'EMAC10'), (12, 'EMAC12'),
                           (20, 'EMAC20'), (26, 'EMAC26')]:
                if n >= sp:
                    out.loc[stk, fn] = float(_ema(p, sp).iloc[-1]) / cv
            for w, fn in [(5, 'MAC5'), (10, 'MAC10'), (20, 'MAC20'), (60, 'MAC60')]:
                if n >= w:
                    out.loc[stk, fn] = float(p.rolling(w).mean().iloc[-1]) / cv
            if n >= 20:
                ma20  = p.rolling(20).mean()
                std20 = p.rolling(20).std()
                out.loc[stk, 'boll_down'] = float((ma20 - 2 * std20).iloc[-1]) / cv
    except Exception as e:
        print(f"    [技术指标] {e}")
    return out


# ---- 合并全部因子 --------------------------------------------
def extract_all_factors(stocks: list, date_str: str) -> pd.DataFrame:
    parts = [
        _factor_basic(stocks, date_str),
        _factor_sentiment(stocks, date_str),
        _factor_growth(stocks, date_str),
        _factor_momentum(stocks, date_str),
        _factor_pershare(stocks, date_str),
        _factor_quality(stocks, date_str),
        _factor_risk(stocks, date_str),
        _factor_style(stocks, date_str),
        _factor_technical(stocks, date_str),
    ]
    merged = pd.concat(parts, axis=1)
    merged = merged.loc[stocks]
    merged = merged.dropna(axis=1, how='all')
    return merged.astype(float)


# ============================================================
# PART 3 ── 三步因子预处理
# ============================================================

def step1_mad_zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        s   = df[col].astype(float).copy()
        med = s.median()
        mad = (s - med).abs().median()
        s   = s.clip(med - 3 * mad, med + 3 * mad)
        s   = s.fillna(s.mean())
        std = s.std()
        out[col] = (s - s.mean()) / std if std > 1e-8 else s - s.mean()
    return out


def step2_neutralize(df: pd.DataFrame, stocks: list, date_str: str) -> pd.DataFrame:
    out = df.copy()
    try:
        q_cap   = query(valuation.code, valuation.market_cap).filter(
            valuation.code.in_(stocks))
        cap_df  = _safe_query(q_cap, date_str)
        log_cap = np.log(cap_df['market_cap'].replace(0, np.nan)).dropna()

        ind_raw = get_industry(stocks, date=date_str)
        ind_map = {}
        for stk in stocks:
            try:
                ind_map[stk] = (ind_raw.get(stk, {})
                                .get('sw_l1', {})
                                .get('industry_code', 'unknown'))
            except Exception:
                ind_map[stk] = 'unknown'
        ind_series  = pd.Series(ind_map)
        ind_dummies = pd.get_dummies(ind_series, prefix='ind',
                                     drop_first=True).astype(float)

        for col in out.columns:
            y     = out[col].copy()
            valid = [i for i in y.dropna().index
                     if i in log_cap.index and i in ind_dummies.index]
            if len(valid) < 100:
                continue
            y_v      = y[valid].values
            X_v      = np.column_stack([
                np.ones(len(valid)),
                log_cap[valid].values,
                ind_dummies.loc[valid].fillna(0).values
            ])
            beta, *_ = np.linalg.lstsq(X_v, y_v, rcond=None)
            resid    = y_v - X_v @ beta
            y[valid] = resid
            out[col] = y

        for col in out.columns:
            s   = out[col]
            std = s.std()
            if std > 1e-8:
                out[col] = (s - s.mean()) / std

    except Exception as e:
        print(f"    [中性化] {e}")

    return out.fillna(0)


def step3_sym_orthogonalize(df: pd.DataFrame) -> pd.DataFrame:
    try:
        F = df.fillna(0).values.astype(float)
        n, p = F.shape
        if n < p:
            return df
        cov              = F.T @ F / n
        eigvals, eigvecs = eigh(cov)
        eigvals          = np.maximum(eigvals, 1e-10)
        inv_sqrt         = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        F_orth           = F @ inv_sqrt
        out              = pd.DataFrame(F_orth, index=df.index, columns=df.columns)
        for col in out.columns:
            s   = out[col]
            std = s.std()
            if std > 1e-8:
                out[col] = (s - s.mean()) / std
        return out
    except Exception as e:
        print(f"    [正交化] {e}")
        return df


def full_preprocess(df: pd.DataFrame, stocks: list, date_str: str) -> pd.DataFrame:
    df1 = step1_mad_zscore(df)
    df2 = step2_neutralize(df1, stocks, date_str)
    df3 = step3_sym_orthogonalize(df2)
    return df3.fillna(0)


# ============================================================
# PART 4 ── 神经网络模型定义
# ============================================================

class FCNN(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64,  32),        nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, n_classes),  nn.Softmax(dim=1),
        )
    def forward(self, x):
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes), nn.Softmax(dim=1),
        )
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


class GRUModel(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 10):
        super().__init__()
        self.gru  = nn.GRU(input_dim, 64, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes), nn.Softmax(dim=1),
        )
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :]))


def fit_model(model, X_tr, y_tr, X_val, y_val,
              epochs=EPOCHS, patience=PATIENCE,
              device=DEVICE, batch_size=BATCH_SIZE):
    model     = model.to(device)
    opt       = optim.RMSprop(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    tr_dataset = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.LongTensor(y_tr)
    )
    loader = DataLoader(tr_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=False)

    best_loss  = float('inf')
    best_state = None
    no_imp     = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            loss = loss.detach()
            del xb, yb

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    del X_val_t, y_val_t
    free_memory()
    return model


# ============================================================
# PART 5 ── 神经网络因子得分
# ============================================================

def nn_factor_score(probs: np.ndarray, n_classes: int = 10) -> np.ndarray:
    S = np.arange(1, n_classes + 1, dtype=float)
    return (probs * S).sum(axis=1)


def predict_scores(model, X: np.ndarray, device=DEVICE) -> np.ndarray:
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(X), INFER_BATCH):
            xb    = torch.FloatTensor(X[i: i + INFER_BATCH]).to(device)
            probs = model(xb).cpu().numpy()
            results.append(nn_factor_score(probs, NUM_CLASSES))
            del xb
    return np.concatenate(results)


# ============================================================
# PART 6 ── 月度数据构建
# ============================================================

def get_month_end_dates(start_date: str, end_date: str) -> list:
    all_tds = get_trade_days(start_date=start_date, end_date=end_date)
    ends, cur, prev = [], None, None
    for td in all_tds:
        m = td.strftime('%Y-%m')
        if m != cur:
            if cur is not None:
                ends.append(prev)
            cur = m
        prev = td
    if prev is not None:
        ends.append(prev)
    return ends


def build_monthly_dataset(start_date='2021-01-01', end_date='2025-12-31'):
    dates      = get_month_end_dates(start_date, end_date)
    mf, ml, mr = {}, {}, {}

    for i, date in enumerate(dates[:-1]):
        ndate = dates[i + 1]
        ds    = date.strftime('%Y-%m-%d')
        nds   = ndate.strftime('%Y-%m-%d')
        print(f"\n[月度数据] {ds}")

        try:
            stocks = get_universe(ds)
            if len(stocks) < 200:
                print(f"  股票数不足({len(stocks)})，跳过")
                continue

            raw = extract_all_factors(stocks, ds)
            print(f"  原始因子: {raw.shape[1]} 个，股票: {len(stocks)} 只")

            clean = full_preprocess(raw, stocks, ds)

            p_cur = get_price(stocks, start_date=ds, end_date=ds,
                              fields=['close'], panel=True)['close']
            p_nxt = get_price(stocks, start_date=nds, end_date=nds,
                              fields=['close'], panel=True)['close']
            if p_cur.empty or p_nxt.empty:
                continue

            rets   = (p_nxt.iloc[0] / p_cur.iloc[0] - 1).dropna()
            labels = pd.qcut(rets, q=NUM_CLASSES,
                             labels=False, duplicates='drop').dropna()

            common = clean.index.intersection(labels.index).intersection(rets.index)
            if len(common) < 100:
                continue

            mf[ds] = clean.loc[common]
            ml[ds] = labels[common].astype(int)
            mr[ds] = rets[common]
            print(f"  入库: {len(common)} 只股票，因子: {clean.shape[1]} 个")

        except Exception as e:
            print(f"  {ds} 失败: {e}")
            import traceback; traceback.print_exc()

        free_memory()

    print(f"\n月度数据集构建完成：{len(mf)} 个截面")
    return mf, ml, mr


# ============================================================
# PART 7 ── 滚动训练与预测（核心修复：列对齐）
# ============================================================

def _align_to_cols(dfs_dict: dict, label_dict: dict,
                   dates: list, cols: list) -> tuple:
    """
    将 dates 中各月的因子矩阵对齐到指定列集合 cols：
      · cols 中存在的列直接取值
      · cols 中不存在的列填 0
    返回 (X: np.ndarray, y: np.ndarray)，若无有效数据返回 (None, None)
    """
    Xs, ys = [], []
    for d in dates:
        if d not in dfs_dict:
            continue
        df  = dfs_dict[d]
        # 构造对齐后的矩阵，缺失列补0
        arr = np.zeros((len(df), len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            if c in df.columns:
                arr[:, j] = df[c].values.astype(np.float32)
        Xs.append(arr)
        ys.append(label_dict[d].values)
    if not Xs:
        return None, None
    return np.vstack(Xs), np.concatenate(ys)


def rolling_predict(monthly_factors, monthly_labels, monthly_returns,
                    model_cls, model_name='FCNN'):
    """
    严格时序滚动（无数据泄露）：
      训练集 = [t-30, t-7] 共24月
      验证集 = [t-6,  t-1] 共6月
      测试集 = [t,  t+11]  共12月
    每年重训一次。

    列对齐策略（修复 ValueError）：
      1. 收集训练集所有月的公共列 → train_cols（保证训练集内部列一致）
      2. 验证集、测试集均对齐到 train_cols（缺失列补0，多余列忽略）
      3. 模型 input_dim = len(train_cols)，始终固定
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dates  = sorted(monthly_factors.keys())
    N      = len(dates)
    WINDOW = TRAIN_MONTHS + VAL_MONTHS

    all_scores = {}

    for test_start in range(WINDOW, N, TEST_MONTHS):
        val_start   = test_start - VAL_MONTHS
        train_start = val_start  - TRAIN_MONTHS

        if train_start < 0:
            continue

        tr_dates  = dates[train_start : val_start]
        val_dates = dates[val_start   : test_start]
        tst_dates = dates[test_start  : min(test_start + TEST_MONTHS, N)]

        print(f"\n[{model_name}] "
              f"Train {tr_dates[0][:7]}~{tr_dates[-1][:7]} | "
              f"Val {val_dates[0][:7]}~{val_dates[-1][:7]} | "
              f"Test {tst_dates[0][:7]}~{tst_dates[-1][:7]}")

        # ── Step A：确定训练集公共列 ────────────────────────────
        # 取训练集各月列名的交集，确保 np.vstack 时列数完全一致
        train_col_sets = [
            set(monthly_factors[d].columns)
            for d in tr_dates if d in monthly_factors
        ]
        if not train_col_sets:
            print(f"  训练集为空，跳过")
            continue

        # 交集：只保留所有训练月均存在的列
        train_cols = sorted(
            train_col_sets[0].intersection(*train_col_sets[1:])
        )
        if len(train_cols) == 0:
            print(f"  训练集公共列为空，跳过")
            continue
        print(f"  公共因子列: {len(train_cols)} 个")

        # ── Step B：对齐并收集训练/验证数据 ────────────────────
        X_tr,  y_tr  = _align_to_cols(monthly_factors, monthly_labels,
                                       tr_dates,  train_cols)
        X_val, y_val = _align_to_cols(monthly_factors, monthly_labels,
                                       val_dates, train_cols)
        if X_tr is None or X_val is None:
            print(f"  训练/验证数据不足，跳过")
            continue

        # ── Step C：训练模型 ────────────────────────────────────
        input_dim = len(train_cols)   # 固定 = 公共列数
        model     = model_cls(input_dim, n_classes=NUM_CLASSES)
        model     = fit_model(model, X_tr, y_tr, X_val, y_val)

        del X_tr, y_tr, X_val, y_val
        free_memory()

        # ── Step D：对测试集做预测（同样对齐到 train_cols）──────
        for d in tst_dates:
            if d not in monthly_factors:
                continue
            X_test, _ = _align_to_cols(monthly_factors, monthly_labels,
                                        [d], train_cols)
            if X_test is None:
                continue
            idx            = monthly_factors[d].index
            scores         = predict_scores(model, X_test)
            all_scores[d]  = pd.Series(scores, index=idx, name=d)
            print(f"  预测 {d}: {len(idx)} 只，列: {input_dim}")

        del model
        free_memory()

    return all_scores


# ============================================================
# PART 8 ── 回测评估
# ============================================================

def calc_rankic(all_scores, monthly_returns):
    ic = {}
    for d, sc in all_scores.items():
        if d not in monthly_returns:
            continue
        rt  = monthly_returns[d]
        com = sc.index.intersection(rt.index)
        if len(com) < 50:
            continue
        r, _ = spearmanr(sc[com].values, rt[com].values)
        ic[d] = r
    return pd.Series(ic).sort_index()


def layer_backtest(all_scores, monthly_returns,
                   n_groups=NUM_CLASSES, cost=COST):
    g_rets = {g: [] for g in range(n_groups)}
    dates  = []

    for d in sorted(all_scores.keys()):
        if d not in monthly_returns:
            continue
        sc  = all_scores[d]
        rt  = monthly_returns[d]
        com = sc.index.intersection(rt.index)
        if len(com) < n_groups * 5:
            continue

        try:
            grp = pd.qcut(sc[com], q=n_groups,
                          labels=False, duplicates='drop')
        except Exception:
            continue

        for g in range(n_groups):
            stks = grp[grp == g].index
            avg  = float(rt[stks].mean()) - cost * 2 if len(stks) > 0 else 0.0
            g_rets[g].append(avg)
        dates.append(d)

    nav_df = pd.DataFrame(
        {f'G{g+1}': (1 + np.array(g_rets[g])).cumprod()
         for g in range(n_groups)},
        index=dates)

    ls_arr = np.array(g_rets[n_groups - 1]) - np.array(g_rets[0])
    ls_nav = pd.Series((1 + ls_arr).cumprod(), index=dates)

    return nav_df, ls_nav, ls_arr


def compute_monotonicity(nav_df):
    years = len(nav_df) / 12
    if years <= 0:
        return np.nan
    ann_rets = []
    for col in nav_df.columns:
        final = float(nav_df[col].iloc[-1])
        ann_rets.append(final ** (1 / years) - 1)
    groups = np.arange(1, len(ann_rets) + 1)
    r, _   = spearmanr(groups, ann_rets)
    return abs(r)


def print_report(name, ic_s, nav_df, ls_nav, ls_arr):
    ic_mean = ic_s.mean()
    ic_std  = ic_s.std()
    ic_ir   = ic_mean / (ic_std + 1e-8)
    t_val   = ic_mean / (ic_std / np.sqrt(len(ic_s)) + 1e-8)

    years     = len(ls_arr) / 12
    ls_ann    = (ls_nav.iloc[-1] ** (1 / years) - 1) if years > 0 else 0
    ls_s      = pd.Series(ls_arr)
    ls_sharpe = (ls_s.mean() / (ls_s.std() + 1e-8)) * np.sqrt(12)
    cum_max   = ls_nav.cummax()
    max_dd    = float(((ls_nav - cum_max) / cum_max).min())
    calmar    = ls_ann / abs(max_dd) if max_dd != 0 else 0

    long_ann = (float(nav_df['G10'].iloc[-1]) ** (1 / years) - 1) if years > 0 else 0
    long_s   = nav_df['G10'].pct_change().dropna()
    long_sh  = (long_s.mean() / (long_s.std() + 1e-8)) * np.sqrt(12)

    mono = compute_monotonicity(nav_df)

    print(f"\n{'='*58}")
    print(f"  {name} 分类因子回测报告")
    print(f"{'='*58}")
    print(f"  RankIC 均值      : {ic_mean:+.4f}  ({ic_mean*100:.2f}%)")
    print(f"  RankIC_IR        : {ic_ir:.4f}")
    print(f"  t 值             : {t_val:.2f}")
    print(f"  分层单调性       : {mono:.4f}")
    print(f"  多头年化收益     : {long_ann:.2%}   夏普: {long_sh:.2f}")
    print(f"  多空年化收益     : {ls_ann:.2%}")
    print(f"  多空夏普率       : {ls_sharpe:.2f}")
    print(f"  多空最大回撤     : {abs(max_dd):.2%}")
    print(f"  多空 Calmar      : {calmar:.2f}")
    print(f"{'='*58}")


# ============================================================
# PART 9 ── 可视化模块（兼容旧版 matplotlib）
# ============================================================

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches
from matplotlib.gridspec import GridSpec
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC',
                                           'WenQuanYi Micro Hei', 'Arial Unicode MS',
                                           'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

GROUP_COLORS = [
    '#003F87', '#1565C0', '#1E88E5', '#42A5F5', '#90CAF9',
    '#FFCC80', '#FFA726', '#FB8C00', '#E64A19', '#B71C1C',
]
LS_COLOR   = '#C62828'
NAV_COLOR  = '#1565C0'
BM_COLOR   = '#9E9E9E'
DD_COLOR   = '#CFD8DC'
IC_POS     = '#1565C0'
IC_NEG     = '#C62828'
IC_CUM_CLR = '#E65100'


def _hide_spines(ax, spines=('top', 'right')):
    """兼容旧版 matplotlib，逐个隐藏边框线。"""
    for sp in spines:
        ax.spines[sp].set_visible(False)


def _annual_returns(ret_series: pd.Series) -> pd.Series:
    idx = pd.to_datetime(ret_series.index)
    df  = pd.DataFrame({'ret': ret_series.values}, index=idx)
    ann = (1 + df['ret']).groupby(df.index.year).prod() - 1
    return ann


# ── 图1：分层回测净值 ──────────────────────────────────────
def plot_layer_nav(nav_df: pd.DataFrame, model_name: str = 'FCNN',
                  save_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 5))
    idx = pd.to_datetime(nav_df.index)
    for i, col in enumerate(nav_df.columns):
        lw    = 1.8 if col in ('G1', 'G10') else 0.9
        alpha = 1.0 if col in ('G1', 'G10') else 0.75
        ax.plot(idx, nav_df[col].values,
                color=GROUP_COLORS[i], lw=lw, alpha=alpha,
                label=f'第{i+1}组')
    ax.set_title(f'图 · {model_name} 分类因子分层回测净值',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('净值', fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m'))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    _hide_spines(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=5, fontsize=8,
              loc='upper left', framealpha=0.6,
              columnspacing=0.8, handlelength=1.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── 图2：多空净值 + 回撤 ───────────────────────────────────
def plot_ls_nav(ls_nav: pd.Series, ls_arr: np.ndarray,
                benchmark_rets: pd.Series = None,
                model_name: str = 'FCNN',
                save_path: str = None):
    idx = pd.to_datetime(ls_nav.index)
    dd  = (ls_nav - ls_nav.cummax()) / ls_nav.cummax() * 100

    if benchmark_rets is not None:
        bm_nav = (1 + benchmark_rets.reindex(ls_nav.index).fillna(0)).cumprod()
    else:
        bm_nav = None

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax2.bar(idx, -dd.values, color=DD_COLOR, alpha=0.6,
            width=20, label='多空组合回撤（右轴）')
    ax2.set_ylim(0, max(-dd.min() * 2.5, 15))
    ax2.set_ylabel('回撤 %', fontsize=9, color='#546E7A')
    ax2.tick_params(axis='y', labelsize=8, colors='#546E7A')
    ax2.invert_yaxis()
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    _hide_spines(ax2, ('top',))

    ax1.plot(idx, ls_nav.values, color=LS_COLOR, lw=2.0,
             zorder=3, label='多空组合净值')
    if bm_nav is not None:
        excess = ls_nav.values / bm_nav.values
        ax1.plot(idx, excess, color=NAV_COLOR, lw=1.4,
                 linestyle='--', zorder=2, label='多空超额净值')
        ax1.plot(idx, bm_nav.values, color=BM_COLOR, lw=1.2,
                 zorder=1, alpha=0.7, label='中证全指')

    ax1.set_title(f'图 · {model_name} 因子多空组合净值',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('净值', fontsize=10)
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m'))
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax1.tick_params(axis='x', rotation=30, labelsize=8)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    _hide_spines(ax1, ('top',))
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2 + lines1, labels2 + labels1,
               fontsize=8, loc='upper left', framealpha=0.6, ncol=2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── 图3：多空年度收益 ──────────────────────────────────────
def plot_annual_returns(ls_arr: np.ndarray, dates: list,
                        model_name: str = 'FCNN',
                        save_path: str = None):
    idx      = pd.to_datetime(dates)
    ls_s     = pd.Series(ls_arr, index=idx)
    ann_rets = _annual_returns(ls_s)
    colors   = [LS_COLOR if r >= 0 else '#42A5F5' for r in ann_rets.values]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(ann_rets.index.astype(str), ann_rets.values * 100,
                  color=colors, width=0.6, zorder=3)
    for bar, val in zip(bars, ann_rets.values * 100):
        offset = 0.4 if val >= 0 else -0.8
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=8, color='#37474F')
    ax.axhline(0, color='#757575', lw=0.8)
    ax.set_title(f'图 · {model_name} 分类因子多空组合年度收益',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel('年化收益率 (%)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    _hide_spines(ax)
    ax.legend(handles=[
        matplotlib.patches.Patch(color=LS_COLOR,  label='多空组合（正收益）'),
        matplotlib.patches.Patch(color='#42A5F5', label='多空组合（负收益）'),
    ], fontsize=8, loc='upper right', framealpha=0.6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── 图4：RankIC 走势 ───────────────────────────────────────
def plot_rankic(ic_series: pd.Series, model_name: str = 'FCNN',
                save_path: str = None):
    idx     = pd.to_datetime(ic_series.index)
    ic_vals = ic_series.values
    ic_cum  = ic_series.cumsum().values
    colors  = [IC_POS if v >= 0 else IC_NEG for v in ic_vals]

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    ax1.bar(idx, ic_vals, color=colors, alpha=0.75, width=20,
            zorder=3, label='RankIC')
    ax1.axhline(0, color='#757575', lw=0.6)
    ax1.axhline(ic_series.mean(), color=NAV_COLOR,
                lw=1.0, linestyle='--', alpha=0.8,
                label=f'均值={ic_series.mean()*100:.2f}%')
    ax1.set_ylabel('RankIC', fontsize=10)
    ax1.set_ylim(min(ic_vals) * 1.6, max(ic_vals) * 1.6)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax1.tick_params(axis='y', labelsize=9)
    _hide_spines(ax1, ('top',))

    ax2.plot(idx, ic_cum, color=IC_CUM_CLR, lw=1.8,
             zorder=4, label='RankIC累计值（右轴）')
    ax2.set_ylabel('RankIC 累计值', fontsize=9, color=IC_CUM_CLR)
    ax2.tick_params(axis='y', labelsize=8, colors=IC_CUM_CLR)
    _hide_spines(ax2, ('top',))

    ax1.set_title(f'图 · {model_name} 分类因子 RankIC 走势',
                  fontsize=12, fontweight='bold', pad=8)
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m'))
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax1.tick_params(axis='x', rotation=30, labelsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, loc='upper left', framealpha=0.6, ncol=2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── 图5：多头净值 ──────────────────────────────────────────
def plot_long_nav(nav_df: pd.DataFrame, monthly_returns: dict,
                  model_name: str = 'FCNN',
                  save_path: str = None):
    idx      = pd.to_datetime(nav_df.index)
    long_nav = nav_df['G10'].values

    bm_dates = sorted(set(nav_df.index) & set(monthly_returns.keys()))
    bm_rets  = pd.Series(
        {d: float(monthly_returns[d].mean()) for d in bm_dates}
    ).sort_index()
    bm_rets  = bm_rets.reindex(nav_df.index).fillna(0)
    bm_nav   = (1 + bm_rets.values).cumprod()

    long_s = pd.Series(long_nav, index=idx)
    dd     = (long_s - long_s.cummax()) / long_s.cummax() * 100

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax2.bar(idx, -dd.values, color=DD_COLOR, alpha=0.55,
            width=20, label='多头回撤（右轴）')
    ax2.set_ylim(0, max(-dd.min() * 2.5, 60))
    ax2.set_ylabel('回撤 %', fontsize=9, color='#546E7A')
    ax2.tick_params(axis='y', labelsize=8, colors='#546E7A')
    ax2.invert_yaxis()
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    _hide_spines(ax2, ('top',))

    ax1.plot(idx, long_nav, color=NAV_COLOR, lw=2.0,
             zorder=3, label='多头净值（G10）')
    ax1.plot(idx, bm_nav,   color=BM_COLOR,  lw=1.2,
             linestyle='--', zorder=2, alpha=0.8, label='全A基准')

    ax1.set_title(f'图 · {model_name} 分类因子多头组合净值',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('净值', fontsize=10)
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y/%m'))
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax1.tick_params(axis='x', rotation=30, labelsize=8)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    _hide_spines(ax1, ('top',))
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2 + lines1, labels2 + labels1,
               fontsize=8, loc='upper left', framealpha=0.6, ncol=2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── 图6：多头年度收益 vs 基准 ─────────────────────────────
def plot_long_annual_returns(nav_df: pd.DataFrame, monthly_returns: dict,
                              model_name: str = 'FCNN',
                              save_path: str = None):
    idx    = pd.to_datetime(nav_df.index)
    long_s = nav_df['G10'].pct_change().dropna()
    long_s.index = pd.to_datetime(long_s.index)

    bm_dates = sorted(set(nav_df.index) & set(monthly_returns.keys()))
    bm_rets  = pd.Series(
        {d: float(monthly_returns[d].mean()) for d in bm_dates}
    ).sort_index()
    bm_rets.index = pd.to_datetime(bm_rets.index)
    bm_rets  = bm_rets.reindex(long_s.index).fillna(0)

    long_ann = _annual_returns(long_s)
    bm_ann   = _annual_returns(bm_rets)
    years    = long_ann.index.astype(str)
    x        = np.arange(len(years))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(11, 4))
    bars1 = ax.bar(x - width/2, long_ann.values * 100,
                   width, color=NAV_COLOR, alpha=0.85,
                   label='多头（G10）', zorder=3)
    bars2 = ax.bar(x + width/2,
                   bm_ann.reindex(long_ann.index).fillna(0).values * 100,
                   width, color=BM_COLOR, alpha=0.7,
                   label='全A基准', zorder=3)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + (0.5 if h >= 0 else -1.5),
                f'{h:.1f}%', ha='center', fontsize=7.5, color='#1A237E')

    ax.axhline(0, color='#757575', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=9)
    ax.set_title(f'图 · {model_name} 分类因子多头组合年度收益',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel('年化收益率 (%)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    _hide_spines(ax)
    ax.legend(fontsize=9, framealpha=0.6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── 图7：3×4 汇总仪表盘 ───────────────────────────────────
def plot_summary_dashboard(all_results: dict, monthly_returns: dict,
                            save_path: str = None):
    models = ['FCNN', 'LSTM', 'GRU']
    fig    = plt.figure(figsize=(22, 14))
    gs     = GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.36)

    for row, name in enumerate(models):
        if name not in all_results:
            continue
        scores = all_results[name]
        ic_s   = calc_rankic(scores, monthly_returns)
        nav_df, ls_nav, ls_arr = layer_backtest(scores, monthly_returns)
        dates  = sorted(scores.keys())

        # (A) 分层净值
        ax_a = fig.add_subplot(gs[row, 0])
        idx  = pd.to_datetime(nav_df.index)
        for i, col in enumerate(nav_df.columns):
            lw    = 1.6 if col in ('G1', 'G10') else 0.7
            alpha = 1.0 if col in ('G1', 'G10') else 0.65
            ax_a.plot(idx, nav_df[col].values,
                      color=GROUP_COLORS[i], lw=lw, alpha=alpha)
        ax_a.set_title(f'{name} 分层回测净值', fontsize=9, fontweight='bold')
        ax_a.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax_a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        ax_a.tick_params(labelsize=7)
        ax_a.grid(axis='y', ls='--', alpha=0.35)
        _hide_spines(ax_a)

        # (B) 多空净值 + 回撤
        ax_b  = fig.add_subplot(gs[row, 1])
        ax_b2 = ax_b.twinx()
        dd      = (ls_nav - ls_nav.cummax()) / ls_nav.cummax() * 100
        ls_idx  = pd.to_datetime(ls_nav.index)
        ax_b2.bar(ls_idx, -dd.values, color=DD_COLOR, alpha=0.5, width=20)
        ax_b2.set_ylim(0, max(-dd.min() * 2.5, 15))
        ax_b2.invert_yaxis()
        ax_b2.tick_params(labelsize=6, colors='#78909C')
        ax_b2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        _hide_spines(ax_b2, ('top',))
        ax_b.plot(ls_idx, ls_nav.values, color=LS_COLOR, lw=1.6, zorder=3)
        ax_b.set_title(f'{name} 多空组合净值', fontsize=9, fontweight='bold')
        ax_b.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax_b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        ax_b.tick_params(labelsize=7)
        ax_b.grid(axis='y', ls='--', alpha=0.3, zorder=0)
        _hide_spines(ax_b, ('top',))
        ax_b.set_zorder(ax_b2.get_zorder() + 1)
        ax_b.patch.set_visible(False)

        # (C) 年度收益
        ax_c = fig.add_subplot(gs[row, 2])
        ls_s = pd.Series(ls_arr,
                         index=pd.to_datetime(dates[:len(ls_arr)]))
        ann  = _annual_returns(ls_s)
        clrs = [LS_COLOR if r >= 0 else '#42A5F5' for r in ann.values]
        ax_c.bar(ann.index.astype(str), ann.values * 100,
                 color=clrs, width=0.6, zorder=3)
        ax_c.axhline(0, color='#757575', lw=0.6)
        ax_c.set_title(f'{name} 年度收益（多空）', fontsize=9, fontweight='bold')
        ax_c.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax_c.tick_params(axis='x', rotation=45, labelsize=6.5)
        ax_c.tick_params(axis='y', labelsize=7)
        ax_c.grid(axis='y', ls='--', alpha=0.4, zorder=0)
        _hide_spines(ax_c)

        # (D) RankIC
        ax_d  = fig.add_subplot(gs[row, 3])
        ax_d2 = ax_d.twinx()
        ic_idx    = pd.to_datetime(ic_s.index)
        ic_vals   = ic_s.values
        ic_cum    = ic_s.cumsum().values
        ic_colors = [IC_POS if v >= 0 else IC_NEG for v in ic_vals]
        ax_d.bar(ic_idx, ic_vals, color=ic_colors, alpha=0.75, width=20, zorder=3)
        ax_d.axhline(0, color='#757575', lw=0.5)
        ax_d2.plot(ic_idx, ic_cum, color=IC_CUM_CLR, lw=1.4, zorder=4)
        ax_d2.tick_params(labelsize=6, colors=IC_CUM_CLR)
        _hide_spines(ax_d2, ('top',))
        ax_d.set_title(
            f'{name} RankIC 走势\n'
            f'均值={ic_s.mean()*100:.2f}%  '
            f'IR={ic_s.mean()/(ic_s.std()+1e-8):.2f}',
            fontsize=8.5, fontweight='bold')
        ax_d.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax_d.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        ax_d.tick_params(labelsize=7)
        ax_d.grid(axis='y', ls='--', alpha=0.3, zorder=0)
        _hide_spines(ax_d, ('top',))
        ax_d.set_zorder(ax_d2.get_zorder() + 1)
        ax_d.patch.set_visible(False)

    fig.suptitle('神经网络多因子模型 | FCNN / LSTM / GRU 回测结果汇总',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ── 统一入口 ───────────────────────────────────────────────
def generate_all_charts(all_results: dict, monthly_returns: dict):
    print("\n[可视化] 开始生成图表……")
    for name, scores in all_results.items():
        print(f"  → {name}")
        ic_s                   = calc_rankic(scores, monthly_returns)
        nav_df, ls_nav, ls_arr = layer_backtest(scores, monthly_returns)
        dates                  = sorted(scores.keys())

        plot_layer_nav(nav_df, model_name=name)
        plot_long_nav(nav_df, monthly_returns, model_name=name)
        plot_long_annual_returns(nav_df, monthly_returns, model_name=name)
        plot_ls_nav(ls_nav, ls_arr, model_name=name)
        plot_annual_returns(ls_arr, dates[:len(ls_arr)], model_name=name)
        plot_rankic(ic_s, model_name=name)

    print("  → 汇总仪表盘")
    plot_summary_dashboard(all_results, monthly_returns)
    print("[可视化] 全部完成！")


# ============================================================
# PART 10 ── 主流程入口（含可视化）
# ============================================================

def main():
    print("=" * 58)
    print("  神经网络多因子模型 | 江海证券研报复现版")
    print(f"  数据区间: 2010-01-01 ~ 2025-12-31")
    print(f"  训练窗口: {TRAIN_MONTHS}月训练 / {VAL_MONTHS}月验证 / {TEST_MONTHS}月测试")
    print("=" * 58)

    print("\n[Step 1] 构建月度数据集（全A股，三步预处理，10分组标签）")
    mf, ml, mr = build_monthly_dataset(
        start_date='2010-01-01',
        end_date='2025-12-31'
    )

    all_results = {}

    for cls, name in [(FCNN, 'FCNN'), (LSTMModel, 'LSTM'), (GRUModel, 'GRU')]:
        print(f"\n[Step] 滚动训练 {name}"
              f"（{TRAIN_MONTHS}月训练 / {VAL_MONTHS}月验证 / {TEST_MONTHS}月测试）")
        scores = rolling_predict(mf, ml, mr,
                                 model_cls=cls, model_name=name)
        all_results[name] = scores

        ic_s                   = calc_rankic(scores, mr)
        nav_df, ls_nav, ls_arr = layer_backtest(scores, mr)
        print_report(name, ic_s, nav_df, ls_nav, ls_arr)
        free_memory()

    # ── 生成全套图表 ──────────────────────────────────────
    generate_all_charts(all_results, mr)

    print("\n[DONE] 全部完成！")
    return all_results, mf, ml, mr


# ── 执行 ──────────────────────────────────────────────────
results, monthly_factors, monthly_labels, monthly_returns = main()