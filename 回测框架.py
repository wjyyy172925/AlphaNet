import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from models import AlphaNet_v2
from utils import (
    apply_tradeability_filter,
    backtest_group_strategy,
    build_rolling_splits,
    build_signal_frame,
    calc_performance,
    load_dataset,
    load_model,
    load_sample_meta,
    myDataset,
    predict_model,
    make_run_output_dir,
    res_path,
    set_output_dir,
    to_date_array,
)


device = torch.device("cpu")
print("Using CPU.")
OUTPUT_DIR = make_run_output_dir("Backtest_Results")
set_output_dir(OUTPUT_DIR)
print(f"Results will be saved to: {OUTPUT_DIR}")
MODEL_DIR = Path(os.environ.get("ALPHANET_OUTPUT_ROOT", "Res")) / "Models"
print(f"Models will be loaded from: {MODEL_DIR}")

# ============================================================================
# 1. 加载数据与初始化回测参数
#    X/Y 是模型输入和目标收益，dates/codes 用于还原每个样本的交易日期和股票。
# ============================================================================
X, Y, dates, Y_codes = load_dataset(".")
sample_meta = load_sample_meta(".")
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

if sample_meta is not None and len(sample_meta) != len(X):
    raise ValueError("sample_meta.csv 与 X_fe.npy 样本数不一致")

target_dates = to_date_array(dates)
splits = build_rolling_splits(target_dates)

group_num = 10
model_name = "alphanet_v2"
group_results = []
group_curves = []
strategy_curves = []
ic_curves = []
layer_returns_with_cost = []
layer_round_results = []
top_benchmark_curves = []
cnt = 0

BUY_COMMISSION = 0.0003
SELL_COMMISSION = 0.0003
SELL_STAMP_TAX = 0.0005
BUY_SLIPPAGE = 0.0005
SELL_SLIPPAGE = 0.0005
REBALANCE_DAYS = 10

# 因子分层回测参数：Layer 1 为预测值最高组，Layer 5 为预测值最低组。
layer_num = 5
layer_min_stocks_per_layer = 10


def _calc_turnover_cost(
    current_weights,
    previous_weights,
    buy_commission,
    sell_commission,
    sell_stamp_tax,
    buy_slippage,
    sell_slippage,
):
    names = set(current_weights) | set(previous_weights)
    buy_turnover = 0.0
    sell_turnover = 0.0
    for name in names:
        diff = current_weights.get(name, 0.0) - previous_weights.get(name, 0.0)
        if diff > 0:
            buy_turnover += diff
        elif diff < 0:
            sell_turnover += -diff

    trade_cost = (
        buy_turnover * (buy_commission + buy_slippage)
        + sell_turnover * (sell_commission + sell_slippage + sell_stamp_tax)
    )
    return trade_cost, buy_turnover, sell_turnover

# ============================================================================
# 3. RankIC 分析
#    计算每个调仓日的截面 RankIC，并在回测结束后输出曲线、统计和图片。
# ============================================================================
def compute_rank_ic_by_date(signal_df, min_stocks=30):
    """计算每个调仓日的截面 RankIC 和股票数量。"""
    records = []

    # 只保留预测值和未来收益都有效的股票样本。
    valid_df = signal_df.dropna(subset=["pred", "target"])

    # RankIC 是逐调仓日计算的截面指标，不跨日期混合股票。
    for date, day_df in valid_df.groupby("date"):
        if len(day_df) < min_stocks:
            continue
        if day_df["pred"].nunique() < 2 or day_df["target"].nunique() < 2:
            continue

        # Spearman 相关系数只使用股票排序，因此适合衡量因子排序能力。
        rank_ic, _ = stats.spearmanr(day_df["pred"], day_df["target"])
        if np.isfinite(rank_ic):
            records.append(
                {
                    "date": date,
                    "RankIC": rank_ic,
                    "n_stocks": len(day_df),
                }
            )

    if not records:
        return pd.DataFrame(columns=["date", "RankIC", "n_stocks"])

    return pd.DataFrame(records).sort_values("date")


def save_rankic_plots(ic_curve_df, rankic_mean, rankic_std, rankic_ir, positive_ratio):
    """保存 RankIC 时间序列、累计曲线和汇总表图片。"""
    if plt is None:
        print("matplotlib is not installed; RankIC image outputs were skipped.")
        return

    rankic_series = ic_curve_df["RankIC"]

    # RankIC 时间序列和累计 RankIC 曲线。
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    bar_colors = ["#e74c3c" if value > 0 else "#2ecc71" for value in rankic_series]
    axes[0].bar(
        ic_curve_df["date"],
        rankic_series,
        color=bar_colors,
        alpha=0.7,
        width=5,
    )
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].axhline(
        y=rankic_mean,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {rankic_mean:.4f}",
    )
    axes[0].set_title("AlphaNet RankIC Time Series")
    axes[0].set_ylabel("RankIC")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        ic_curve_df["date"],
        ic_curve_df["cumulative_rankic"],
        color="#e74c3c",
        linewidth=2,
        label="Cumulative RankIC",
    )
    axes[1].fill_between(
        ic_curve_df["date"],
        0,
        ic_curve_df["cumulative_rankic"],
        alpha=0.15,
        color="#e74c3c",
    )
    axes[1].set_title("AlphaNet Cumulative RankIC")
    axes[1].set_ylabel("Cumulative RankIC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(res_path("rankic_analysis_v2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # RankIC 汇总表图片。
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    table_data = [
        ["RankIC Mean", "RankIC Std", "IC_IR", "IC>0 Ratio", "N Periods"],
        [
            f"{rankic_mean * 100:.2f}%",
            f"{rankic_std * 100:.2f}%",
            f"{rankic_ir:.2f}",
            f"{positive_ratio * 100:.2f}%",
            f"{len(rankic_series)}",
        ],
    ]
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.18] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for column in range(5):
        table[0, column].set_facecolor("#e74c3c")
        table[0, column].set_text_props(color="white", fontweight="bold")
        table[1, column].set_facecolor("#f8f9fa")
    ax.set_title("AlphaNet IC Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(res_path("ic_summary_v2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 4. TOP VS Benchmark
# ============================================================================
def compute_top_vs_benchmark(signal_df, n_layers=5, min_stocks_per_layer=10):
    """Compute TOP portfolio return vs equal-weight benchmark by date."""
    valid_df = signal_df.dropna(subset=["pred", "target"]).copy()
    records = []

    for date, day_df in valid_df.groupby("date", sort=True):
        if len(day_df) < n_layers * min_stocks_per_layer:
            continue

        day_df = day_df.sort_values("pred", ascending=False).reset_index(drop=True)
        layer_size = len(day_df) // n_layers
        if layer_size == 0:
            continue

        records.append(
            {
                "date": date,
                "top_ret": day_df.iloc[:layer_size]["target"].mean(),
                "bench_ret": day_df["target"].mean(),
            }
        )

    if not records:
        empty = pd.DataFrame(
            columns=[
                "top_ret",
                "bench_ret",
                "excess_ret",
                "cum_top",
                "cum_bench",
                "cum_excess",
                "excess_drawdown",
            ]
        )
        empty.index.name = "date"
        return empty

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["excess_ret"] = df["top_ret"] - df["bench_ret"]
    df["cum_top"] = (1 + df["top_ret"]).cumprod()
    df["cum_bench"] = (1 + df["bench_ret"]).cumprod()
    df["cum_excess"] = (1 + df["excess_ret"]).cumprod()
    rolling_max = df["cum_excess"].expanding().max()
    df["excess_drawdown"] = (df["cum_excess"] - rolling_max) / rolling_max
    return df


def save_top_vs_benchmark_plots(excess_df):
    """Save TOP vs Benchmark NAV and excess return drawdown plots."""
    if plt is None:
        print("matplotlib is not installed; TOP vs Benchmark image outputs were skipped.")
        return

    if excess_df.empty:
        print("No valid TOP vs Benchmark records; image outputs were skipped.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        excess_df.index,
        excess_df["cum_top"],
        color="#e74c3c",
        linewidth=2,
        label="TOP Portfolio",
    )
    ax.plot(
        excess_df.index,
        excess_df["cum_bench"],
        color="#95a5a6",
        linewidth=2,
        label="Benchmark (Equal Weight)",
    )
    ax.set_title("AlphaNet TOP Portfolio vs Benchmark NAV", fontsize=14, fontweight="bold")
    ax.set_ylabel("Net Asset Value", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(res_path("top_vs_benchmark_v2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(
        excess_df.index,
        (excess_df["cum_excess"] - 1) * 100,
        color="#e74c3c",
        linewidth=2,
        label="Cumulative Excess Return",
    )
    ax1.set_ylabel("Cumulative Excess Return (%)", fontsize=12, color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")

    ax2 = ax1.twinx()
    ax2.fill_between(
        excess_df.index,
        excess_df["excess_drawdown"] * 100,
        0,
        color="#3498db",
        alpha=0.3,
        label="Excess Return Drawdown",
    )
    ax2.set_ylabel("Excess Return Drawdown (%)", fontsize=12, color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)
    ax1.set_title("AlphaNet Excess Return & Drawdown", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(res_path("excess_return_drawdown_v2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# 4. 因子分层回测
#    按每个调仓日的预测值排序，计算各层收益、长短组合和交易成本情景。
# ============================================================================
def layered_test(
    signal_df,
    n_layers=5,
    min_stocks_per_layer=10,
    buy_commission=BUY_COMMISSION,
    sell_commission=SELL_COMMISSION,
    sell_stamp_tax=SELL_STAMP_TAX,
    buy_slippage=BUY_SLIPPAGE,
    sell_slippage=SELL_SLIPPAGE,
):
    """运行截面因子分层回测。

    Layer 1 是预测值最高组合，Layer ``n_layers`` 是预测值最低组合。
    cost 表示单边固定成本，按参考实现每次调仓扣除双边成本。
    """
    if n_layers < 2:
        raise ValueError("n_layers must be at least 2")

    # 分层回测同样只使用预测值和未来收益均有效的样本。
    valid_df = signal_df.dropna(subset=["pred", "target"]).copy()
    layer_gross_returns = {f"Layer {i + 1}": [] for i in range(n_layers)}
    layer_costs = {f"Layer {i + 1}": [] for i in range(n_layers)}
    layer_dates = []
    previous_weights_by_layer = {f"Layer {i + 1}": {} for i in range(n_layers)}

    # 每个调仓日独立排序和分层，避免不同日期之间相互影响。
    for date, day_df in valid_df.groupby("date", sort=True):
        if len(day_df) < n_layers * min_stocks_per_layer:
            continue

        # 预测值从高到低排序，因此 Layer 1 是最高预测值组合。
        day_df = day_df.sort_values("pred", ascending=False).reset_index(drop=True)
        layer_size = len(day_df) // n_layers
        if layer_size == 0:
            continue

        for layer_idx in range(n_layers):
            start_idx = layer_idx * layer_size
            end_idx = start_idx + layer_size if layer_idx < n_layers - 1 else len(day_df)
            layer_df = day_df.iloc[start_idx:end_idx].copy()
            layer_gross_ret = layer_df["target"].mean()
            current_weights = {
                str(code): 1.0 / len(layer_df)
                for code in layer_df["code"].astype(str).tolist()
            }
            trade_cost, _, _ = _calc_turnover_cost(
                current_weights,
                previous_weights_by_layer[f"Layer {layer_idx + 1}"],
                buy_commission,
                sell_commission,
                sell_stamp_tax,
                buy_slippage,
                sell_slippage,
            )
            layer_gross_returns[f"Layer {layer_idx + 1}"].append(layer_gross_ret)
            layer_costs[f"Layer {layer_idx + 1}"].append(trade_cost)
            previous_weights_by_layer[f"Layer {layer_idx + 1}"] = current_weights

        layer_dates.append(date)

    # ret_df 的index是调仓日期，Layer 1, Layer 2, ..., Layer n，value = 该层该日的真实未来收益率（已扣交易成本）
    gross_df = pd.DataFrame(layer_gross_returns, index=pd.to_datetime(layer_dates))
    cost_df = pd.DataFrame(layer_costs, index=pd.to_datetime(layer_dates))
    ret_df = gross_df - cost_df
    ret_df.index.name = "date"
    nav_df = (1 + ret_df).cumprod()

    # 多空组合为最高预测层减最低预测层。
    if ret_df.empty:
        long_short_ret = pd.Series(dtype=float, name="long_short_ret")
        long_short_nav = pd.Series(dtype=float, name="long_short_nav")
    else:
        long_short_ret = (
            gross_df["Layer 1"]
            - gross_df[f"Layer {n_layers}"]
            - cost_df["Layer 1"]
            - cost_df[f"Layer {n_layers}"]
        ).rename("long_short_ret")
        long_short_nav = (1 + long_short_ret).cumprod().rename("long_short_nav")

    return nav_df, ret_df, long_short_nav, long_short_ret


def compute_layer_stats(ret_df, long_short_ret, rebalance_days=10):
    """计算各因子层和多空组合的整段回测区间的年化统计指标。"""
    # 测试集收益是每 rebalance_days 个交易日一个观测值。
    annual_factor = 252 / rebalance_days
    stats_out = {}

    # 分别计算每一层的年化收益、年化波动率和胜率。
    for layer_name in ret_df.columns:
        returns = pd.Series(ret_df[layer_name]).dropna()
        if returns.empty:
            ann_ret = np.nan
            ann_vol = np.nan
        else:
            ann_ret = (1 + returns.mean()) ** annual_factor - 1
            ann_vol = returns.std() * np.sqrt(annual_factor)

        stats_out[layer_name] = {
            "Ann. Return": ann_ret,
            "Ann. Volatility": ann_vol,
            "Win Rate": (returns > 0).mean() if not returns.empty else np.nan,
            "N Periods": len(returns),
        }

    # 再计算 Layer 1 - Layer N 的多空组合指标。
    ls = pd.Series(long_short_ret).dropna()
    if ls.empty:
        ls_ann_ret = np.nan
        ls_ann_vol = np.nan
    else:
        ls_ann_ret = (1 + ls.mean()) ** annual_factor - 1
        ls_ann_vol = ls.std() * np.sqrt(annual_factor)

    top_returns = pd.Series(ret_df["Layer 1"]).dropna() if "Layer 1" in ret_df else pd.Series(dtype=float)
    top_ann_ret = stats_out.get("Layer 1", {}).get("Ann. Return", np.nan)
    top_ann_vol = stats_out.get("Layer 1", {}).get("Ann. Volatility", np.nan)

    return {
        "Layer Stats": stats_out,
        "L/S Ann. Return": ls_ann_ret,
        "L/S Volatility": ls_ann_vol,
        "L/S Sharpe": ls_ann_ret / ls_ann_vol if pd.notna(ls_ann_vol) and ls_ann_vol > 0 else np.nan,
        "TOP IR": top_ann_ret / top_ann_vol if pd.notna(top_ann_vol) and top_ann_vol > 0 else np.nan,
        "TOP Win Rate": (top_returns > 0).mean() if not top_returns.empty else np.nan,
        "TOP Ann. Return": top_ann_ret,
        "N Periods": len(ret_df),
    }


def save_layered_plots(
    nav_with_cost,
    ls_nav_with_cost,
    stats_with_cost,
):
    """保存分层净值、多空净值和各层年化收益图片。"""
    if plt is None:
        print("matplotlib is not installed; layered test image outputs were skipped.")
        return

    if nav_with_cost.empty:
        print("No valid layered test records; layered test plots were skipped.")
        return

    layer_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    if len(nav_with_cost.columns) > len(layer_colors):
        cmap = plt.get_cmap("tab10")
        layer_colors = [cmap(i) for i in range(len(nav_with_cost.columns))]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, column in enumerate(nav_with_cost.columns):
        ax.plot(
            nav_with_cost.index,
            nav_with_cost[column],
            color=layer_colors[i],
            linewidth=2,
            label=column,
        )
    ax.set_title("Layered NAV (Transaction Cost 0.2%)")
    ax.set_ylabel("Net Asset Value")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(res_path("layered_test_v2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    if not ls_nav_with_cost.empty:
        ax.plot(
            ls_nav_with_cost.index,
            ls_nav_with_cost,
            color="#3498db",
            linewidth=2,
            label=f"Cost 0.2% Ann.Ret={stats_with_cost['L/S Ann. Return'] * 100:.1f}%",
        )
    ax.set_title("AlphaNet Long-Short Portfolio NAV")
    ax.set_ylabel("Net Asset Value")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(res_path("long_short_v2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    layer_stats = stats_with_cost["Layer Stats"]
    names = list(layer_stats)
    returns = [layer_stats[name]["Ann. Return"] * 100 for name in names]
    colors = ["#e74c3c" if value > 0 else "#3498db" for value in returns]
    bars = ax.bar(names, returns, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_title("Annualized Return by Layer (Transaction Cost 0.2%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, value in zip(bars, returns):
        offset = 0.3 if value >= 0 else -0.3
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(res_path("layer_returns_v2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# 将多个滚动窗口的分层收益按日期拼接，避免窗口重叠时重复计算。
def combine_layer_returns(layer_frames, n_layers):
    """合并滚动窗口的分层收益，返回日期索引的收益表。"""
    layer_columns = [f"Layer {i + 1}" for i in range(n_layers)]
    if not layer_frames:
        empty = pd.DataFrame(columns=layer_columns)
        empty.index.name = "date"
        return empty

    combined = pd.concat(layer_frames).reset_index()
    combined["date"] = pd.to_datetime(combined["date"])
    combined = (
        combined.sort_values(["date", "round"])
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
    )
    return combined[layer_columns].sort_index()


def combine_top_benchmark_returns(curves):
    """Combine TOP vs Benchmark curves from rolling windows."""
    columns = [
        "top_ret",
        "bench_ret",
        "excess_ret",
        "cum_top",
        "cum_bench",
        "cum_excess",
        "excess_drawdown",
    ]
    if not curves:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = "date"
        return empty

    combined = pd.concat(curves).reset_index()
    combined["date"] = pd.to_datetime(combined["date"])
    combined = (
        combined.sort_values(["date", "round"])
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
    )
    return combined[columns].sort_index()


# 遍历所有滚动窗口
# ============================================================================
# 5. 滚动窗口回测主流程
#    每个窗口使用训练集拟合标准化器，加载对应模型，并在测试集生成预测值。
#    RankIC、因子分层和 Top 组策略都只使用当前测试窗口的数据。
# ============================================================================
for start, valid_start, test_start, end in splits:
    test_set = myDataset(X[test_start:end], Y[test_start:end])
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    # 5.1 当前数据已在构建数据集阶段完成逐样本逐行标准化，这里直接构造测试集。
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    best_net = AlphaNet_v2(d=10, stride=10, n=X.shape[1])
    # 5.2 加载当前滚动窗口对应的模型并生成预测值。
    model_path = MODEL_DIR / f"{model_name}_{cnt}.pt"
    load_model(best_net, model_path, device=device)
    test_preds = predict_model(best_net, test_loader, device=device)

    # 每个测试集中所有样本对应的date，code，pred，target(每行一只股票一天)
    # date只包含测试集中的调仓日期
    # 5.3 组装测试信号表：每行对应一只股票在一个调仓日的预测值和未来收益。
    signal_df = build_signal_frame(
        test_preds,
        Y[test_start:end],
        target_dates[test_start:end],
        Y_codes[test_start:end],
    )
    signal_df.to_csv(res_path(f"signal_df_round_{cnt}.csv"), index=False)

    if sample_meta is not None:
        meta_slice = sample_meta.iloc[test_start:end].reset_index(drop=True)
        signal_df = pd.concat([signal_df.reset_index(drop=True), meta_slice], axis=1)
        signal_df = signal_df.loc[:, ~signal_df.columns.duplicated()].copy()
    signal_df = apply_tradeability_filter(signal_df)

    # ------------------------------------------------------------------------
    # 5.4 RankIC 分析：统计当前测试窗口每个调仓日的截面预测相关性。
    # ------------------------------------------------------------------------
    ic_df_round = compute_rank_ic_by_date(signal_df)
    ic_df_round.to_csv(res_path(f"rankic_round_{cnt}.csv"), index=False)
    ic_df_round["round"] = cnt
    if not ic_df_round.empty:
        ic_curves.append(ic_df_round)

    # ------------------------------------------------------------------------
    # 5.5 因子分层回测：只计算含交易成本的各层组合收益。
    # ------------------------------------------------------------------------
    layer_nav_with_cost, layer_ret_with_cost, ls_nav_with_cost, ls_ret_with_cost = layered_test(
        signal_df,
        n_layers=layer_num,
        min_stocks_per_layer=layer_min_stocks_per_layer,
        buy_commission=BUY_COMMISSION,
        sell_commission=SELL_COMMISSION,
        sell_stamp_tax=SELL_STAMP_TAX,
        buy_slippage=BUY_SLIPPAGE,
        sell_slippage=SELL_SLIPPAGE,
    )
    layer_stats_with_cost = compute_layer_stats(
        layer_ret_with_cost,
        ls_ret_with_cost,
    )
    top_benchmark_round = compute_top_vs_benchmark(
        signal_df,
        n_layers=layer_num,
        min_stocks_per_layer=layer_min_stocks_per_layer,
    )
    if not top_benchmark_round.empty:
        top_benchmark_curves.append(top_benchmark_round.reset_index().assign(round=cnt))
    if not layer_ret_with_cost.empty:
        layer_returns_with_cost.append(layer_ret_with_cost.assign(round=cnt))
        layer_round_results.extend(
            [
                {
                    "round": cnt,
                    "scenario": "with_cost",
                    "ls_annual_return": layer_stats_with_cost["L/S Ann. Return"],
                    "ls_sharpe": layer_stats_with_cost["L/S Sharpe"],
                    "top_annual_return": layer_stats_with_cost["TOP Ann. Return"],
                    "top_ir": layer_stats_with_cost["TOP IR"],
                    "top_win_rate": layer_stats_with_cost["TOP Win Rate"],
                    "n_periods": layer_stats_with_cost["N Periods"],
                },
            ]
        )


# ============================================================================
# 7. RankIC 汇总输出
#    合并所有滚动窗口的 RankIC，按日期去重后计算累计 RankIC 和整体指标。
# ============================================================================
if ic_curves:
    ic_curve_df = pd.concat(ic_curves, ignore_index=True)
    ic_curve_df = ic_curve_df.sort_values(["date", "round"])
    ic_curve_df = ic_curve_df.drop_duplicates(subset=["date"], keep="last")

if ic_curves and not ic_curve_df.empty:
    # 计算累计 RankIC、均值、标准差、IC_IR 和正 IC 占比。
    ic_curve_df["cumulative_rankic"] = ic_curve_df["RankIC"].cumsum()
    ic_curve_df.to_csv(res_path("rankic_curve_v2.csv"), index=False)

    rankic_series = ic_curve_df["RankIC"]
    rankic_mean = rankic_series.mean()
    rankic_std = rankic_series.std()
    rankic_ir = rankic_mean / rankic_std if rankic_std > 0 else np.nan
    positive_ratio = (rankic_series > 0).mean()
    ic_summary = pd.DataFrame(
        [
            {
                "RankIC Mean": rankic_mean,
                "RankIC Std": rankic_std,
                "IC_IR": rankic_ir,
                "IC>0 Ratio": positive_ratio,
                "N Periods": len(rankic_series),
            }
        ]
    )
    ic_summary.to_csv(res_path("ic_summary_v2.csv"), index=False)

    print("Overall Mean IC:", rankic_mean * 100, "%")
    print("Overall Std IC:", rankic_std)
    print("Overall IC_IR:", rankic_ir)
    print("Overall Positive Ratio:", positive_ratio * 100, "%")
    print("RankIC Periods:", len(rankic_series))

    save_rankic_plots(
        ic_curve_df,
        rankic_mean,
        rankic_std,
        rankic_ir,
        positive_ratio,
    )
else:
    print("No valid RankIC records; RankIC plots and summary were skipped.")

# ============================================================================
# 8. 因子分层回测汇总输出
#    合并各滚动窗口的分层收益，计算分层净值、长短组合净值和统计结果。
# ============================================================================
all_layer_ret_with_cost = combine_layer_returns(layer_returns_with_cost, layer_num)

if not all_layer_ret_with_cost.empty:
    all_ls_ret_with_cost = (
        all_layer_ret_with_cost["Layer 1"]
        - all_layer_ret_with_cost[f"Layer {layer_num}"]
    ).rename("long_short_ret")
    all_layer_nav_with_cost = (1 + all_layer_ret_with_cost).cumprod()
    all_ls_nav_with_cost = (1 + all_ls_ret_with_cost).cumprod().rename("long_short_nav")

    all_layer_stats_with_cost = compute_layer_stats(
        all_layer_ret_with_cost,
        all_ls_ret_with_cost,
    )

    layer_curve_output = all_layer_ret_with_cost.add_suffix("_ret_with_cost")
    layer_curve_output["long_short_ret_with_cost"] = all_ls_ret_with_cost
    layer_curve_output["long_short_nav_with_cost"] = all_ls_nav_with_cost
    layer_curve_output.to_csv(res_path("layer_curve_v2.csv"))

    layer_nav_output = all_layer_nav_with_cost.add_suffix("_nav_with_cost")
    layer_nav_output.to_csv(res_path("layer_nav_v2.csv"))

    layer_summary_rows = []
    for layer_name, layer_stats in all_layer_stats_with_cost["Layer Stats"].items():
        layer_summary_rows.append(
            {
                "scenario": "with_cost",
                "portfolio": layer_name,
                "annual_return": layer_stats["Ann. Return"],
                "annual_volatility": layer_stats["Ann. Volatility"],
                "win_rate": layer_stats["Win Rate"],
                "n_periods": layer_stats["N Periods"],
            }
        )
    layer_summary_rows.append(
        {
            "scenario": "with_cost",
            "portfolio": "Long-Short",
            "annual_return": all_layer_stats_with_cost["L/S Ann. Return"],
            "annual_volatility": all_layer_stats_with_cost["L/S Volatility"],
            "win_rate": np.nan,
            "n_periods": all_layer_stats_with_cost["N Periods"],
        }
    )

    pd.DataFrame(layer_summary_rows).to_csv(res_path("layer_summary_v2.csv"), index=False)
    pd.DataFrame(layer_round_results).to_csv(res_path("layer_round_results_v2.csv"), index=False)

    print("\nFactor Layered Test Results")
    print("Cost 0.2%:")
    for layer_name, layer_stats in all_layer_stats_with_cost["Layer Stats"].items():
        print(
            f"  {layer_name}: Annualized Return = "
            f"{layer_stats['Ann. Return'] * 100:.2f}%"
        )
    print(
        f"  Long-Short Annualized Return = "
        f"{all_layer_stats_with_cost['L/S Ann. Return'] * 100:.2f}%, "
        f"Sharpe = {all_layer_stats_with_cost['L/S Sharpe']:.2f}"
    )
    print(
        f"  TOP IR = {all_layer_stats_with_cost['TOP IR']:.2f}, "
        f"TOP Win Rate = {all_layer_stats_with_cost['TOP Win Rate'] * 100:.2f}%"
    )

    save_layered_plots(
        all_layer_nav_with_cost,
        all_ls_nav_with_cost,
        all_layer_stats_with_cost,
    )
else:
    print("No valid factor layered test records; layered outputs were skipped.")


# ============================================================================
# 9. TOP vs Benchmark
# ============================================================================
top_vs_benchmark_df = combine_top_benchmark_returns(top_benchmark_curves)

if not top_vs_benchmark_df.empty:
    top_vs_benchmark_df.to_csv(res_path("top_vs_benchmark_curve_v2.csv"))

    annual_factor = 252 / REBALANCE_DAYS
    top_ret = top_vs_benchmark_df["top_ret"].dropna()
    bench_ret = top_vs_benchmark_df["bench_ret"].dropna()
    excess_ret = top_vs_benchmark_df["excess_ret"].dropna()

    top_ann_ret = (1 + top_ret.mean()) ** annual_factor - 1 if not top_ret.empty else np.nan
    bench_ann_ret = (
        (1 + bench_ret.mean()) ** annual_factor - 1 if not bench_ret.empty else np.nan
    )
    excess_ann_ret = (
        (1 + excess_ret.mean()) ** annual_factor - 1 if not excess_ret.empty else np.nan
    )
    excess_ann_vol = excess_ret.std() * np.sqrt(annual_factor) if not excess_ret.empty else np.nan
    excess_ir = (
        excess_ann_ret / excess_ann_vol
        if pd.notna(excess_ann_vol) and excess_ann_vol > 0
        else np.nan
    )
    max_dd = top_vs_benchmark_df["excess_drawdown"].min()
    win_rate = (excess_ret > 0).mean() if not excess_ret.empty else np.nan

    pd.DataFrame(
        [
            {
                "top_annual_return": top_ann_ret,
                "benchmark_annual_return": bench_ann_ret,
                "ann_excess_return": excess_ann_ret,
                "ann_tracking_error": excess_ann_vol,
                "information_ratio": excess_ir,
                "max_excess_drawdown": max_dd,
                "win_rate": win_rate,
                "n_periods": len(top_vs_benchmark_df),
            }
        ]
    ).to_csv(res_path("top_vs_benchmark_summary_v2.csv"), index=False)

    print("\nTOP vs Benchmark Results")
    print(f"  TOP Annualized Return = {top_ann_ret * 100:.2f}%")
    print(f"  Benchmark Annualized Return = {bench_ann_ret * 100:.2f}%")
    print(f"  Ann. Excess Return = {excess_ann_ret * 100:.2f}%")
    print(f"  Ann. Tracking Error = {excess_ann_vol * 100:.2f}%")
    print(f"  Information Ratio = {excess_ir:.2f}")
    print(f"  Max Excess Drawdown = {max_dd * 100:.2f}%")
    print(f"  Win Rate = {win_rate * 100:.2f}%")

    save_top_vs_benchmark_plots(top_vs_benchmark_df)
else:
    print("No valid TOP vs Benchmark records; outputs were skipped.")
