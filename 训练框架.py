import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import AlphaNet_v2
from utils import build_rolling_splits, load_dataset, myDataset, save_model, to_date_array


device = torch.device("cpu")
print("Using CPU.")

X, Y, dates, _ = load_dataset(".")
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
os.makedirs("Models", exist_ok=True)

target_dates = to_date_array(dates)
splits = build_rolling_splits(target_dates)

# Sanity check
net = AlphaNet_v2(d=10, stride=10, n=X.shape[1])
net(torch.tensor(X[:5]).float())

torch.manual_seed(42)
lr = 0.0001
n_epoch = 10
batch_size = 1000
model_name = "alphanet_v2"

results = {
    "round": [],
    "train": [],
    "valid": [],
}

cnt = 0

for start, valid_start, test_start, _ in splits:
    net = AlphaNet_v2(d=10, stride=10, n=X.shape[1])
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = myDataset(X[start:valid_start], Y[start:valid_start], is_train=True)
    train_scaler = train_set.get_scaler()
    valid_set = myDataset(
        X[valid_start:test_start],
        Y[valid_start:test_start],
        scaler=train_scaler,
        is_train=False,
    )

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
    with open("train_results_v2.pickle", "wb") as f:
        pickle.dump(results, f)

    cnt += 1

print("Training complete.")
