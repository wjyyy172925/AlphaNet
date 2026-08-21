# Converted from 训练框架（分类任务）.ipynb

# ------------------------------------------------------------------------
# # 训练框架（分类任务）
# ------------------------------------------------------------------------

# pip install audtorch

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from audtorch.metrics.functional import pearsonr
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, confusion_matrix

# 设置设备
# 因为特征提取层的计算无法完全矩阵化，使用CPU训练更快

device = torch.device("cpu")
print("Using CPU.")

# ------------------------------------------------------------------------
# ## 数据准备
# ------------------------------------------------------------------------

# 导入数据
X = np.load('X_fe.npy')
Y = np.load('Y_fe_cls.npy')
dates = np.load('Y_dates.npy')

print('Shape of X: ', X.shape)
print('Shape of Y: ', Y.shape)

Y[:5]

class myDataset(Dataset):
    '''
    自定义数据集，将原始数据从 numpy arrays 转换成 float 格式的 tensors
    '''
    
    def __init__(self, X, y):
        super(myDataset, self).__init__()
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------------------------------------------------
# ## 模型
# ------------------------------------------------------------------------

from models import *

# 保存模型参数到本地
def save_model(model, name):
    torch.save(model.state_dict(), name)

# 从本地导入模型参数
def load_model(model, name):
    weights = torch.load(name)
    model.load_state_dict(weights)

# ------------------------------------------------------------------------
# ## 测试
# ------------------------------------------------------------------------

# 用小量数据测试模型是否能正常工作
net = AlphaNet_v2(d=10, stride=10, n=15)
net(torch.tensor(X[:5]).float())

# ------------------------------------------------------------------------
# ## 滚动训练区间
# ------------------------------------------------------------------------

from datetime import datetime, timedelta

# 获取所有标签对应的日期 target_dates 以及数据集中所有不重复的日期 unique_dates
target_dates = np.array([datetime.strptime(str(date), '%Y-%m-%d').date() for date in dates])
unique_dates = sorted(np.unique(target_dates))

# 从2011.01.31开始到2023.05.31，每隔半年滚动训练（测试集为半年126个交易日）
# 每次训练数据量为1500个交易日，其中80%是训练集，20%是验证集
start_dates = []
starts, ends = [], []

# 找出所有的训练区间，以便后续划分数据集
i, start, end = 0, 0, 0
while i + 1200 + 300 + 126 <= len(unique_dates):
    start_dates.append(i)
    start = sum(target_dates < unique_dates[i])
    starts.append(start)
    end = sum(target_dates < unique_dates[i+1200+300+126])
    ends.append(end)
    i += 126

# 总共有6个训练区间，模型会在6个数据集上滚动训练
start_dates

# ------------------------------------------------------------------------
# ## 验证指标
# ------------------------------------------------------------------------

# 计算准确率
def compute_accuracy(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    return accuracy_score(np.array(y_true), np.array(y_pred))

# 计算f1-score
def compute_f1(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    return f1_score(np.array(y_true), np.array(y_pred))

# 计算MCC
def compute_MCC(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    return matthews_corrcoef(np.array(y_true), np.array(y_pred))

# 计算4种指标
def evaluate_metrics(preds, labels):
    labels = [int(item) for item in labels]
    preds = [0 if item < 0.5 else 1 for item in preds]
    accuracy = compute_accuracy(labels, preds)
    f1 = compute_f1(labels, preds)
    mcc = compute_MCC(labels, preds)
    cm = confusion_matrix(labels, preds)
    return accuracy, f1, mcc, cm

# ------------------------------------------------------------------------
# ## 训练模型
# ------------------------------------------------------------------------

# 设置随机种子，保证训练结果一致
torch.manual_seed(42)

# 设置训练参数：学习率、训练迭代次数、批量大小
lr = 0.0001
n_epoch = 10
batch_size = 2000

# 初始化训练的对象
model_name = 'alphanet_v2_cls'
net = AlphaNet_v2(d=10, stride=10, n=15)

# 初始化输出处理层（Sigmoid函数）、损失函数和优化器
f = nn.Sigmoid()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

# 初始化一个字典，用来储存模型训练期间的表现
results = {}
results['date'] = []
results['train'] = []
results['valid'] = []
results['test'] = []

# 维护 cnt 变量，记录当前是第几个训练轮次
cnt = 0

# 滚动窗口
for start, end in zip(starts, ends):

    # 按照 8:1:1 划分出训练、验证和测试集
    n = end - start
    train_set = myDataset(X[start:start+int(n*8/10)], Y[start:start+int(n*8/10)])
    valid_set = myDataset(X[start+int(n*8/10):start+int(n*9/10)], Y[start+int(n*8/10):start+int(n*9/10)])
    test_set = myDataset(X[start+int(n*9/10):end], Y[start+int(n*9/10):end])
    
    # 创建loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # 当前训练轮次的模型储存地址
    model_path = 'Models/' + model_name + '_' + str(cnt) + '.pt'
    
    count = 0
    train_loss_lst, valid_loss_lst = [], []
    best_valid_loss = float('inf')
    
    for epoch in range(n_epoch):
        
        # 训练
        net.train()
        train_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            preds = f(net(x))
            loss = criterion(preds, y.unsqueeze(dim=1))
            train_loss += loss.item() * len(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset.X)
        
        
        # 验证
        net.eval()
        valid_loss = 0
        with torch.no_grad():
            for x, y in tqdm(valid_loader):
                x, y = x.to(device), y.to(device)
                preds = f(net(x))
                loss = criterion(preds, y.unsqueeze(dim=1))
                valid_loss += loss.item() * len(x)
        valid_loss /= len(valid_loader.dataset.X)
        
        
        # 监测训练效果
        print("Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}".format(epoch+1, train_loss, valid_loss))
        
        # 记录训练效果
        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
        
        # 若当前模型验证效果比历史最佳更好，更新本地模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(net, model_path)
            print("Saved model with validation loss of {:.4f}".format(best_valid_loss)) 
        else:
            count += 1
            
       # 早停：若累计有5次迭代，模型都没有进步，停止本轮训练
        if count >= 5:
            break

    
    # 测试最佳模型效果
    best_net = AlphaNet_v2(d=10, stride=10, n=15)
    load_model(best_net, model_path)
    best_net.eval()
    test_loss = 0
    test_preds, test_labels= [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            preds = f(best_net(x))
            test_preds.append(preds)
            test_labels.append(y)
            loss = criterion(preds, y.unsqueeze(dim=1))
            test_loss += loss.item() * len(x)  
    test_loss /= len(test_loader.dataset.X)
    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)
    
    # 实时监控训练过程中的指标
    test_a_i, test_f1_i, test_mcc_i, test_cm_i = evaluate_metrics(test_preds, test_labels)
    print("\n")
    print("-" * 50)
    print('Test Results: ')
    print("Accuracy: {:.4f}".format(test_a_i))
    print("F1 Score: {:.4f}".format(test_f1_i))
    print("MCC Score: {:.4f}".format(test_mcc_i))
    print("-" * 50)
    print("\n")
    
    
    # 记录当前训练轮次的指标变动，并更新本地储存结果
    results['date'].append(str(cnt))       
    results['train'].append(train_loss_lst)  
    results['valid'].append(valid_loss_lst)  
    results['valid'].append([test_a_i, test_f1_i, test_mcc_i, test_cm_i])
    with open('train_results_v2_cls.pickle', 'wb') as file:
        pickle.dump(results, file)
    
    # 下一轮
    cnt += 1

