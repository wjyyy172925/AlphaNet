import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import AlphaNet_v2
from utils import (
    build_rolling_splits,
    make_run_output_dir,
    load_dataset,
    load_sample_meta,
    myDataset,
    save_model,
    to_date_array,
)


device = torch.device("cpu")
print("Using CPU.")
OUTPUT_DIR = make_run_output_dir("Training_Results")
print(f"Results will be saved to: {OUTPUT_DIR}")

X, Y, dates, _ = load_dataset(".")
sample_meta = load_sample_meta(".")
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
os.makedirs("Models", exist_ok=True)

# 对Y标准化
Y = (Y - np.mean(Y)) / np.std(Y+1e-8)

if sample_meta is not None and len(sample_meta) != len(X):
    raise ValueError("sample_meta.csv 与 X_fe.npy 样本数不一致")

if sample_meta is not None and "sample_tradeable" in sample_meta.columns:
    sample_tradeable = sample_meta["sample_tradeable"].fillna(False).to_numpy(dtype=bool)
else:
    sample_tradeable = np.ones(len(X), dtype=bool)

target_dates = to_date_array(dates)
splits = build_rolling_splits(target_dates)

# Sanity check
net = AlphaNet_v2(d=10, stride=10, n=X.shape[1])
net(torch.tensor(X[:5]).float())

torch.manual_seed(42)
lr = 0.0001
n_epoch = 50
batch_size = 1000
model_name = "alphanet_v2"

results = {
    "round": [],
    "train": [],
    "valid": [],
}

cnt = 0

for start, valid_start, test_start, _ in splits:
    train_mask = sample_tradeable[start:valid_start]
    valid_mask = sample_tradeable[valid_start:test_start]

    X_train = X[start:valid_start][train_mask]
    Y_train = Y[start:valid_start][train_mask]
    X_valid = X[valid_start:test_start][valid_mask]
    Y_valid = Y[valid_start:test_start][valid_mask]

    if len(X_train) == 0 or len(X_valid) == 0:
        print(
            f"第 {cnt} 个滚动窗口没有足够的可交易样本: "
            f"train={len(X_train)}, valid={len(X_valid)}"
        )
        continue

    net = AlphaNet_v2(d=10, stride=10, n=X.shape[1])
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = myDataset(X_train, Y_train)
    valid_set = myDataset(X_valid, Y_valid)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    model_path = os.path.join("Models", f"{model_name}_{cnt}.pt")
    count = 0
    train_loss_lst, valid_loss_lst = [], []
    best_valid_loss = float("inf")

    for epoch in range(n_epoch):
        net.train()
        train_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            preds = net(x)
            loss = criterion(preds, y.unsqueeze(dim=1))
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        net.eval()
        valid_loss = 0
        with torch.no_grad():
            for x, y in tqdm(valid_loader):
                x, y = x.to(device), y.to(device)
                preds = net(x)
                loss = criterion(preds, y.unsqueeze(dim=1))
                valid_loss += loss.item()
        valid_loss /= len(valid_loader.dataset)

        print(
            f"Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}"
        )

        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            count = 0
            save_model(net, model_path)
            print(f"Saved model with validation loss of {best_valid_loss:.4f}")
        else:
            count += 1

        if count >= 5:
            break

    results["round"].append(str(cnt))
    results["train"].append(train_loss_lst)
    results["valid"].append(valid_loss_lst)
    with open(OUTPUT_DIR / "train_results_v2.pickle", "wb") as f:
        pickle.dump(results, f)

    cnt += 1

print("Training complete.")
