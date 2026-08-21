import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm


# df_merged: raw data
# df_merged_fe: data with ratio features
# df_merged_fe1: data with ratio and mean features

file_name = 'df_merged.csv'
df_merged = pd.read_csv(file_name)


# 创建标签列，保留不变
df_merged['target'] = df_merged['return']

X, Y, Y_dates, empty = [], [], [], []


for code in tqdm(df_merged['code'].unique()):
    
    x, y, dates = [], [], []
    
    # 每个个股单独采样
    df = df_merged[df_merged['code']==code]
    
    i = 0
    while i + 40 < len(df):
        
        # 标签对应的日期
        date = df.iloc[i+40]['date']
        dates.append(date)
        
        # 特征：30天的历史窗口，构建 ”数据图片“
        window = df.iloc[i:i+30, 1:-1]
        window.set_index('date', inplace=True)
        window = window.transpose()          # 转置为 (特征数, 天数) = (9, 30)
        x.append(np.array(window))
        
        # 标签：第i+40天的收益率
        y.append(df.iloc[i+40]['target'])
        
        # 每间隔10个交易日采样一次
        i += 10
    
    # 如果该个股的数据不够，跳过
    if not x or not y:
        empty.append(code)
        continue
    
    # 将该个股的所有的 样本-标签 组合加入到数据集中
    x = np.stack(x)         # 将list转为numpy数组，形状: (B, n, T)
    y = np.stack(y)
    y_dates = np.stack(dates)
    X.append(x)                 # 加入该股票所有日期的特征
    Y.append(y)                 # 加入该股票所有日期的标签
    Y_dates.append(y_dates)     # 加入该股票所有的标签日期


# 根据标签日期对数据集进行排序
Y_dates = np.concatenate(Y_dates, axis=0)
order = np.argsort(Y_dates)     # 获取按日期升序排列的索引
X = np.concatenate(X, axis=0)[order]
Y = np.concatenate(Y, axis=0)[order]
Y_dates = Y_dates[order]

# 储存数据集到本地
np.save('X_fe.npy', X)
np.save('Y_fe.npy', Y)
np.save('Y_dates.npy', Y_dates)

