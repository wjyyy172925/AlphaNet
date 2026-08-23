import os
import pickle

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
    backtest_group_strategy,
    build_rolling_splits,
    build_signal_frame,
    calc_performance,
    load_dataset,
    load_model,
    myDataset,
    predict_model,
    to_date_array,
)


device = torch.device("cpu")
print("Using CPU.")

X, Y, dates, Y_codes = load_dataset(".")
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

target_dates = to_date_array(dates)
splits = build_rolling_splits(target_dates)

group_num = 10
model_name = "alphanet_v2"
group_results = []
group_curves = []
strategy_curves = []
ic_curves = []
cnt = 0

BUY_COMMISSION = 0.0003
SELL_COMMISSION = 0.0003
SELL_STAMP_TAX = 0.0005
BUY_SLIPPAGE = 0.0005
SELL_SLIPPAGE = 0.0005


def build_top_group_cost_curve(
    signal_df,
    group_num,
    buy_commission,
    sell_commission,
    sell_stamp_tax,
    buy_slippage,
    sell_slippage,
):
    signal_df = signal_df.copy().dropna(subset=["pred", "next_rn"])
    rows = []
    previous_weights = {}

    for date, day_df in signal_df.groupby("date"):
        day_df = day_df.sort_values("pred", ascending=True).copy()
        size = len(day_df)
        if size == 0:
            continue

        day_df["group"] = np.ceil(
            day_df["pred"].rank(method="first") / (size / group_num)
        ).astype(int)
        day_df["group"] = day_df["group"].clip(1, group_num)

        top_df = day_df[day_df["group"] == group_num].copy()
        if top_df.empty:
            continue

        gross_ret = top_df["next_rn"].mean()
        weight = 1.0 / len(top_df)
        current_weights = {
            str(code): weight for code in top_df["code"].astype(str).tolist()
        }

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

        rows.append(
            {
                "date": date,
                "strategy_gross_ret": gross_ret,
                "strategy_cost": trade_cost,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "turnover": buy_turnover + sell_turnover,
            }
        )
        previous_weights = current_weights

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "strategy_gross_ret",
                "strategy_cost",
                "buy_turnover",
                "sell_turnover",
                "turnover",
            ]
        )
        empty.index.name = "date"
        return empty

    return pd.DataFrame(rows).sort_values("date").set_index("date")


def compute_rank_ic_by_date(signal_df, min_stocks=30):
    """Compute cross-sectional RankIC and stock count for each rebalance date."""
    records = []
    valid_df = signal_df.dropna(subset=["pred", "next_rn"])

    for date, day_df in valid_df.groupby("date"):
        if len(day_df) < min_stocks:
            continue
        if day_df["pred"].nunique() < 2 or day_df["next_rn"].nunique() < 2:
            continue

        rank_ic, _ = stats.spearmanr(day_df["pred"], day_df["next_rn"])
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
    """Save RankIC time-series, cumulative curve, and summary table images."""
    if plt is None:
        print("matplotlib is not installed; RankIC image outputs were skipped.")
        return

    rankic_series = ic_curve_df["RankIC"]

    # RankIC time series and cumulative RankIC curve.
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
    plt.savefig("rankic_analysis_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # IC summary table image, following the all_procedure presentation.
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
    plt.savefig("ic_summary_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# 遍历所有滚动窗口
for start, valid_start, test_start, end in splits:
    train_set = myDataset(X[start:valid_start], Y[start:valid_start], is_train=True)
    train_scaler = train_set.get_scaler()
    test_set = myDataset(X[test_start:end], Y[test_start:end], scaler=train_scaler, is_train=False)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    best_net = AlphaNet_v2(d=10, stride=10, n=X.shape[1])
    model_path = os.path.join("Models", f"{model_name}_{cnt}.pt")
    load_model(best_net, model_path, device=device)
    test_preds = predict_model(best_net, test_loader, device=device)

    # 每个测试集中所有样本对应的date，code，pred，next_rn(每行一只股票一天)
    # date只包含测试集中的调仓日期
    signal_df = build_signal_frame(
        test_preds,
        Y[test_start:end],
        target_dates[test_start:end],
        Y_codes[test_start:end],
    )

    # 只有测试集里的调仓日才计算 IC
    ic_df_round = compute_rank_ic_by_date(signal_df)
    valid_ic = ic_df_round["RankIC"].to_numpy()
    group_curve, strategy_curve, strategy_group, strategy_stats = backtest_group_strategy(
        signal_df,
        group_num=group_num,
    )

    cost_curve = build_top_group_cost_curve(
        signal_df=signal_df,
        group_num=group_num,
        buy_commission=BUY_COMMISSION,
        sell_commission=SELL_COMMISSION,
        sell_stamp_tax=SELL_STAMP_TAX,
        buy_slippage=BUY_SLIPPAGE,
        sell_slippage=SELL_SLIPPAGE,
    )
    strategy_curve = strategy_curve.join(cost_curve, how="left")
    strategy_curve[["strategy_cost", "buy_turnover", "sell_turnover", "turnover"]] = (
        strategy_curve[["strategy_cost", "buy_turnover", "sell_turnover", "turnover"]]
        .fillna(0.0)
    )
    strategy_curve["strategy_gross_ret"] = strategy_curve["strategy_gross_ret"].fillna(
        strategy_curve["strategy_ret"]
    )
    strategy_curve["strategy_ret"] = (
        strategy_curve["strategy_gross_ret"] - strategy_curve["strategy_cost"]
    )
    strategy_curve["strategy_nav"] = (1 + strategy_curve["strategy_ret"]).cumprod()
    strategy_stats = calc_performance(strategy_curve["strategy_ret"])

    mean_ic = valid_ic.mean() if len(valid_ic) else np.nan
    std_ic = valid_ic.std() if len(valid_ic) else np.nan
    ic_ratio = mean_ic / std_ic if len(valid_ic) and std_ic != 0 else np.nan
    positive_ratio = np.mean(valid_ic > 0) if len(valid_ic) else np.nan

    print(
        f"Round {cnt}: Mean IC: {mean_ic * 100:.4f}%, Std IC: {std_ic:.4f}, "
        f"IC_IR: {ic_ratio:.4f}, Positive Ratio: {positive_ratio * 100:.4f}%"
    )
    print(
        f"Round {cnt}: Strategy Group: {strategy_group}, Strategy Annual Return: {strategy_stats['annual_return']:.4f}, "
        f"Sharpe: {strategy_stats['sharpe_ratio']:.4f}, Max Drawdown: {strategy_stats['max_drawdown']:.4f}"
    )
    print(
        f"Round {cnt}: Avg Turnover: {strategy_curve['turnover'].mean():.4f}, "
        f"Avg Cost: {strategy_curve['strategy_cost'].mean():.6f}"
    )

    ic_curves.append(ic_df_round.assign(round=cnt))
    group_results.append(
        {
            "round": cnt,
            "strategy_group": strategy_group,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ic_ir": ic_ratio,
            "positive_ic_ratio": positive_ratio,
            "strategy_annual_return": strategy_stats["annual_return"],
            "strategy_annual_volatility": strategy_stats["annual_volatility"],
            "strategy_sharpe_ratio": strategy_stats["sharpe_ratio"],
            "strategy_max_drawdown": strategy_stats["max_drawdown"],
            "strategy_avg_turnover": strategy_curve["turnover"].mean(),
            "strategy_avg_cost": strategy_curve["strategy_cost"].mean(),
        }
    )
    group_curves.append(group_curve.assign(round=cnt, strategy_group=strategy_group))
    strategy_curves.append(strategy_curve.reset_index().assign(round=cnt, strategy_group=strategy_group))
    cnt += 1

with open("group_results_v2.pickle", "wb") as f:
    pickle.dump(group_results, f)

if group_curves:
    group_curve_df = pd.concat(group_curves, ignore_index=True)
    group_curve_df.to_csv("group_curve_v2.csv", index=False)

if strategy_curves:
    strategy_curve_df = pd.concat(strategy_curves, ignore_index=True)
    strategy_curve_df.to_csv("strategy_curve_v2.csv", index=False)

if ic_curves:
    ic_curve_df = pd.concat(ic_curves, ignore_index=True)
    ic_curve_df = ic_curve_df.sort_values(["date", "round"])
    ic_curve_df = ic_curve_df.drop_duplicates(subset=["date"], keep="last")

if ic_curves and not ic_curve_df.empty:
    ic_curve_df["cumulative_rankic"] = ic_curve_df["RankIC"].cumsum()
    ic_curve_df.to_csv("rankic_curve_v2.csv", index=False)

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
    ic_summary.to_csv("ic_summary_v2.csv", index=False)

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

if strategy_curves:
    all_curve = pd.concat(strategy_curves, ignore_index=True)
    overall_strategy = calc_performance(all_curve["strategy_ret"])
    print("Overall Strategy Return:", overall_strategy["annual_return"])
