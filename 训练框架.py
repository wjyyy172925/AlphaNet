# Converted from 训练框架.ipynb

# ------------------------------------------------------------------------
# # 训练框架
# ------------------------------------------------------------------------

# pip install audtorch

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


device = torch.device("cpu")
print("Using CPU.")

# ------------------------------------------------------------------------
# ## 数据准备
# ------------------------------------------------------------------------

# 导入数据
X = np.load('X_fe.npy')             # (样本数,特征数,窗口长度) 
Y = np.load('Y_fe.npy')        # (样本数,)      
dates = np.load('Y_dates.npy')      # (样本数,)

print('Shape of X: ', X.shape)
print('Shape of Y: ', Y.shape)

class myDataset(Dataset):
    '''
    自定义数据集，将原始数据从 numpy arrays 转换成 float 格式的 tensors
    '''
    
    def __init__(self, X, y, scaler = None, is_train = True):
        super(myDataset, self).__init__()
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.origin_shape = X.shape

        # (B, n, T) → (B*T, n)
        X_2d = X.transpose(0,2,1).reshape(-1,self.origin_shape[1])

        if is_train:
            self.scaler = StandardScaler()
            X_trans = self.scaler.fit_transform(X_2d)
        else:
            self.scaler = scaler
            X_trans = self.scaler.transform(X_2d)

        self.X = X_trans.reshape(self.origin_shape)
        self.X = torch.as_tensor(self.X, dtype = torch.float32)
        self.y = torch.as_tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_scaler(self):
        return self.scaler

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

# 用前5条数据测试模型是否能正常工作
net = AlphaNet_v2(d=10, stride=10, n=15)
net(torch.tensor(X[:5]).float())

# ------------------------------------------------------------------------
# ## 滚动训练区间
# ------------------------------------------------------------------------

from datetime import datetime, timedelta

# 获取所有标签对应的日期 target_dates 以及数据集中所有不重复的日期 unique_dates
'''
target_dates 是按日期全局升序排列的数组
同一个日期的所有股票样本会连续聚集在一起
同一个日期分组内部，不同股票的样本顺序取决于它们之前在多线程处理中被合并的顺序
'''
target_dates = np.array([datetime.strptime(str(date), '%Y-%m-%d').date() for date in dates])
unique_dates = sorted(np.unique(target_dates))

# 从2011.01.31开始到2023.05.31，每隔半年滚动训练（测试集为半年126个交易日）
# 每次训练数据量为1500个交易日，其中80%是训练集，20%是验证集
start_dates = []
starts, valid_starts, test_starts, ends = [], [], [], []

# 找出所有的训练区间，以便后续划分数据集
i, start, end = 0, 0, 0
k = int(1500 * 0.8)
while i + 1500 + 126 <= len(unique_dates):
    start_dates.append(i)
    start = sum(target_dates < unique_dates[i])
    starts.append(start)

    valid_start = sum(target_dates < unique_dates[i+k])
    valid_starts.append(valid_start)

    test_start = sum(target_dates < unique_dates[i+1500])
    test_starts.append(test_start)

    end = sum(target_dates < unique_dates[i+1500+126])
    ends.append(end)
    i += 126

# 总共有6个训练区间，模型会在6个数据集上滚动训练
print(f'start_dates:{start_dates}')

# ------------------------------------------------------------------------
# ## 训练模型
# ------------------------------------------------------------------------

# 设置随机种子，保证训练结果一致
torch.manual_seed(42)

# 设置训练参数：学习率、训练迭代次数、批量大小
lr = 0.0001
n_epoch = 10
batch_size = 1000

# 初始化一个字典，用来储存模型训练期间的表现
results = {}
results['round'] = []
results['train'] = []
results['valid'] = []
results['test'] = []

# 维护 cnt 变量，记录当前是第几个训练轮次
cnt = 0
model_name = 'alphanet_v2'

# 滚动窗口
for start, valid_start, test_start, end in zip(starts, valid_starts, test_starts, ends):
    # 初始化训练的对象：'alphanet_v2'，'alphanet_att'，'alphanet_v2_fe'
    net = AlphaNet_v2(d=10, stride=10, n=15)

    # 初始化损失函数和优化器
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 测试集为126天，训练集：验证集 = 4:1
    n = end - start
    train_set = myDataset(X[start:valid_start], Y[start:valid_start], is_train=True)
    train_scaler = train_set.get_scaler()
    valid_set = myDataset(X[valid_start:test_start], Y[valid_start:test_start], scaler = train_scaler, is_train=False)
    test_set = myDataset(X[test_start:end], Y[test_start:end], scaler = train_scaler, is_train=False)
    
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
            preds = net(x)
            loss = criterion(preds, y.unsqueeze(dim=1))
            train_loss += loss.item()
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
                preds = net(x)
                loss = criterion(preds, y.unsqueeze(dim=1))
                valid_loss += loss.item()  
        valid_loss /= len(valid_loader.dataset.X)
        
        
        # 监测训练效果
        print("Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}".format(epoch+1, train_loss, valid_loss))
        
        # 记录训练效果
        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
        
        # 更新本地模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            count = 0
            save_model(net, model_path)
            print("Saved model with validation loss of {:.4f}".format(best_valid_loss)) 
        else:
            count += 1
            
        # 早停：若累计有5次迭代，模型都没有进步，停止本轮训练
        if count >= 5:
            break
    
    
    # 记录当前训练轮次的指标变动，并更新本地储存结果
    results['round'].append(str(cnt))          
    results['train'].append(train_loss_lst)   
    results['valid'].append(valid_loss_lst)   
    with open('train_results_v2.pickle', 'wb') as f:
        pickle.dump(results, f)
    
    # 下一轮
    cnt += 1

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

