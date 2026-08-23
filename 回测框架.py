import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import AlphaNet_v2
from utils import (
    backtest_group_strategy,
    build_rolling_splits,
    build_signal_frame,
    calc_performance,
    compute_daily_ic,
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
ic_results = []
group_results = []
group_curves = []
strategy_curves = []
cnt = 0

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

    daily_ic = compute_daily_ic(signal_df)
    valid_ic = daily_ic[np.isfinite(daily_ic)]
    group_curve, strategy_curve, strategy_group, strategy_stats = backtest_group_strategy(
        signal_df,
        group_num=group_num,
    )

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

    ic_results.append(valid_ic)
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

if ic_results:
    all_ic = np.concatenate(ic_results, axis=0)
    print("Overall Mean IC:", np.nanmean(all_ic) * 100, "%")
    print("Overall Std IC:", np.nanstd(all_ic))
    print("Overall IC_IR:", np.nanmean(all_ic) / np.nanstd(all_ic))
    print("Overall Positive Ratio:", np.mean(all_ic > 0) * 100, "%")

if strategy_curves:
    all_curve = pd.concat(strategy_curves, ignore_index=True)
    overall_strategy = calc_performance(all_curve["strategy_ret"])
    print("Overall Strategy Return:", overall_strategy["annual_return"])
