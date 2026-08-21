# Converted from 回测框架.ipynb

# ------------------------------------------------------------------------
# # 回测框架
# ------------------------------------------------------------------------

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from audtorch.metrics.functional import pearsonr
from torch.utils.data import DataLoader, Dataset
import pickle
from models import *
import scipy.stats as stats



# 设置设备
# 因为特征提取层的计算无法完全矩阵化，使用CPU训练更快

device = torch.device("cpu")
print("Using CPU.")

# ------------------------------------------------------------------------
# ## 数据准备
# ------------------------------------------------------------------------

# 导入数据
X = np.load('X_fe.npy')
Y = np.load('Y_fe_norm.npy')
dates = np.load('Y_dates.npy')

print('Shape of X: ', X.shape)
print('Shape of Y: ', Y.shape)

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
# ## 单因子回测
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# **RankIC（Rank Information Coefficient）** 是用来衡量选股因子与股票收益排名之间的相关性的指标，从而评估以选股因子的有效性和稳定性。
# 
# 计算选股因子的RankIC的一般步骤如下：
# 
# 1. 对于每个时间点，根据选股因子的值对股票进行排名，得到每个股票在因子上的排名值
# 
# 2. 对于每个时间点，根据股票的实际收益对股票进行排名，得到每个股票在收益上的排名值
# 
# 3. 计算因子排名和收益排名之间的相关性，可以使用：
# 
#    - 秩相关系数（Spearman's rank correlation coefficient）
#    - 皮尔逊相关系数（Pearson correlation coefficient）
#    
#    
# 4. 对所有时间点的RankIC进行统计分析，例如：计算平均值、标准差、假设检验等
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# **IC_IR（Information Coefficient Information Ratio）**是一种用于评估选股模型的指标，结合了选股因子的RankIC和预测准确性。
# 
# IC_IR的计算方法如下：
# 
# 1. 算选股因子的RankIC
# 
# 2. 计算因子的平均IC：将每个时间点的RankIC取平均，得到选股因子的平均IC
# 
# 3. 计算因子的IC标准差：计算RankIC的标准差，衡量选股因子在不同时间点上的波动性
# 
# 4. 计算IC_IR：IC_IR = mean(IC) / std(IC)
# 
# IC_IR的值越高，表示选股因子的选股能力越强，具有更高的预测准确性和稳定性。
# ------------------------------------------------------------------------


def compute_RankIC(X, Y, model, target_dates):
    
    results = []
    unique_dates = np.unique(target_dates)
    
    # 针对每个目标日期，对比当天真的股票收益率排名和预测的排名
    for date in tqdm(unique_dates):
        
        # 获取当日所有股票的信息
        idx = np.where(target_dates==date)[0]
        
        # 当日小于20支股票，跳过该日
        if len(idx) < 20:
            continue
        
        # 预测个股收益率值
        model.eval()
        y_preds = -model(torch.tensor(X[idx]).float()).squeeze().detach().numpy()
        
        # 计算排名
        y_rank = np.argsort(Y[idx]).argsort() + 1
        y_pred_rank = np.argsort(y_preds).argsort() + 1
        
        # 计算排名之间的相关度
        correlation, _ = stats.spearmanr(y_rank, y_pred_rank)
        results.append(correlation)
        
    return np.array(results)


results = []

# 选择模型：'alphanet_v2'，'alphanet_att'，'alphanet_v2_fe'
model_name = 'alphanet_v2'

cnt = 0

# 使用每个训练区间的最佳模型，来预测对应区间测试集的收益率，计算IC值
for start, end in zip(starts, ends):
    
    # 导入模型
    model_path = 'Models/' + model_name + '_' + str(cnt) + '.pt'
    net = AlphaNet_v2(d=10, stride=10, n=15)
    load_model(net, model_path)
    
    # 预测 + 验证
    n = end - start
    test_res = compute_RankIC(X[start+int(n*9/10):end], Y[start+int(n*9/10):end], net, target_dates[start+int(n*9/10):end])
    
    print(model_path,
          round(100*np.mean(test_res), 2), 
          round(100*np.std(test_res), 2), 
          round(np.mean(test_res)/np.std(test_res), 4), 
          round(100* sum(test_res > 0) / len(test_res), 2))
    
    results.append(test_res)

    with open('test_results_v2.pickle', 'wb') as f:
        pickle.dump(results, f)
    
    cnt += 1

