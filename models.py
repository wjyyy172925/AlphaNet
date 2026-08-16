import torch
import torch.nn as nn
from audtorch.metrics.functional import pearsonr



# --------------------特征提取层--------------------

class ts_corr(nn.Module):
    """
    计算过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的相关系数
    """

    def __init__(self, d=10, stride=10):
        """
        d: 计算窗口的天数
        stride：计算窗口在时间维度上的进步大小
        """
        super(ts_corr, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        batch_size, n, T = X.shape      # (B,特征数,T)
        unfolded_X = X.unfold(2, self.d, self.stride)
        # 生成所有特征两两组合的索引，并且每一对只出现一次，不计算特征自己和自己的相关系数。
        rows, cols = torch.triu_indices(n, n, offset=1, device=X.device)

        x = unfolded_X[:, rows, :, :]
        y = unfolded_X[:, cols, :, :]
        corr = pearsonr(x.flatten(0, 2), y.flatten(0, 2))

        return corr.reshape(batch_size, len(rows), unfolded_X.size(2))


class ts_cov(nn.Module):
    """
    计算过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的协方差
    """

    def __init__(self, d=10, stride=10):
        super(ts_cov, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        B, n, T = X.shape

        # [B, n, w, d]
        unfolded_X = X.unfold(2, self.d, self.stride)

        # 特征两两组合
        rows, cols = torch.triu_indices(
            n, n, offset=1, device=X.device
        )

        # [B, h, w, d]
        x = unfolded_X[:, rows, :, :]
        y = unfolded_X[:, cols, :, :]

        # 过去 d 天的均值
        x_mean = torch.mean(x, dim=3, keepdim=True)
        y_mean = torch.mean(y, dim=3, keepdim=True)

        # 样本协方差
        cov = (
            (x - x_mean) * (y - y_mean)
        ).sum(dim=3) / (self.d - 1)

        # [B, h, w]
        return cov



class ts_stddev(nn.Module):
    """
    计算过去 d 天 X 值构成的时序数列的标准差
    """

    def __init__(self, d=10, stride=10):
        super(ts_stddev, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        # X: [B, n, T]

        # [B, n, w, d]
        unfolded_X = X.unfold(2, self.d, self.stride)

        # 对每个 d 天窗口计算标准差
        # 输出：[B, n, w]
        std = torch.std(unfolded_X, dim=3)

        return std

class ts_zscore(nn.Module):
    """
    计算过去 d 天 X 值构成的时序数列的平均值除以标准差
    """

    def __init__(self, d=10, stride=10):
        super(ts_zscore, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        # X: [B, n, T]

        # [B, n, w, d]
        unfolded_X = X.unfold(2, self.d, self.stride)

        # [B, n, w]
        mean = torch.mean(unfolded_X, dim=3)
        std = torch.std(unfolded_X, dim=3)

        # 防止标准差为0
        zscore = mean / (std + 1e-8)

        return zscore

class ts_return(nn.Module):
    def __init__(self, d=10, stride=10):
        super(ts_return, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        unfolded_X = X.unfold(2, self.d, self.stride)
        return1 = unfolded_X[:, :, :, -1] / (
            unfolded_X[:, :, :, 0] + 1e-8
        ) - 1

        return return1

    
class ts_decaylinear(nn.Module):
    def __init__(self, d=10, stride=10):
        super(ts_decaylinear, self).__init__()
        self.d = d
        self.stride = stride

        # 如下设计的权重系数满足离现在越近的日子权重越大
        weights = torch.arange(d, 0, -1, dtype=torch.float32)
        weights = weights / weights.sum()

        # 注册权重，不用在前向传播函数中重复计算
        self.register_buffer('weights', weights)

    def forward(self, X):
        unfolded_X = X.unfold(2, self.d, self.stride)

        # 在时间维度上，将 weights 与 unfolded_X 相乘
        decaylinear = torch.sum(
            unfolded_X * self.weights.view(1, 1, 1, -1),
            dim=-1
        )

        return decaylinear
    
    

    
# --------------------AlphaNet-v1--------------------

class AlphaNet(nn.Module):
    '''
    第一版AlphaNet：输入 + 特征提取层 + 池化层 + 特征展平/残差连接 + 降维度层 + 输出层
    '''

    def __init__(self, d=10, stride=10, d_pool=3, s_pool=3, n=9):
        super(AlphaNet, self).__init__()
        
        # d-回看窗口大小，stride-时间步大小
        self.d = d
        self.stride = stride

        # ts_corr() 和 ts_cov() 输出的特征数量
        h = int(n * (n - 1) / 2)

        # 特征提取层
        self.feature_extractors = nn.ModuleList([
            ts_corr(self.d, self.stride),
            ts_cov(self.d, self.stride),
            ts_stddev(self.d, self.stride),
            ts_zscore(self.d, self.stride),
            ts_return(self.d, self.stride),
            ts_decaylinear(self.d, self.stride),
            nn.AvgPool1d(self.d, self.stride)
        ])

        # 特征提取层后面接的批归一化层
        self.batch_norms1 = nn.ModuleList([
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n)
        ])

        # 池化层（无参数）
        self.avg_pool = nn.AvgPool1d(d_pool, s_pool)    # kernel_size, stride
        self.max_pool = nn.MaxPool1d(d_pool, s_pool)
        
        # 池化层后面接的批量归一化层
        self.batch_norms2 = nn.ModuleList([])
        # corr和cov 2个特征提取器（会产生h个特征） × 每个提取器3种池化方式
        for _ in range(2):
            for _ in range(3):
                self.batch_norms2.append(nn.BatchNorm1d(h))
        # 剩余的5个特征提取器（会产生h个特征） × 每个提取器3种池化方式
        for _ in range(5):
            for _ in range(3):
                self.batch_norms2.append(nn.BatchNorm1d(n))

        # 特征展平并拼接后的总数
        # 池化不改变特征数量，因此 提取特征 和 池化特征 数量相同
        n_in = 2 * (h*2*3 + n*5*3)

        # 线性层，输出层，激活函数，失活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear_layer = nn.Linear(n_in, 30)
        self.output_layer = nn.Linear(30, 1)

        # 初始化线性层和输出层
        self.initialize_weights()


    def initialize_weights(self):
        
        # 用 truncated_normal 方法初始化线性层和输出层的权重
        nn.init.trunc_normal_(self.linear_layer.weight)
        nn.init.trunc_normal_(self.output_layer.weight)


    def forward(self, X):
        # X: [B, n, T]; C=h for corr/cov branches, otherwise C=n.
        # C：不同特征提取算子处理后的特征数量
        # w: 特征提取后保留的时间窗口数量。
        # p: 特征在 Avg/Max/Min 池化后保留的时间窗口数量：
        # w=floor((T-d)/stride)+1, p=floor((w-d_pool)/s_pool)+1.
        features_fe, features_p, i = [], [], 0
        for extractor, batch_norm in zip(self.feature_extractors, self.batch_norms1):
            
            # 特征提取 + 批量归一化 + 展平
            x = extractor(X)                  # [B, C, w]
            x = batch_norm(x)                 # [B, C, w]
            features_fe.append(x.flatten(start_dim=1))  # [B, C*w]
            
            # 池化层 + 批量归一化 + 展平
            x_avg = self.batch_norms2[i](self.avg_pool(x))       # [B, C, p]
            x_max = self.batch_norms2[i+1](self.max_pool(x))     # [B, C, p]
            x_min = self.batch_norms2[i+2](-self.max_pool(-x))   # [B, C, p]
            features_p.append(x_avg.flatten(start_dim=1))        # [B, C*p]
            features_p.append(x_max.flatten(start_dim=1))        # [B, C*p]
            features_p.append(x_min.flatten(start_dim=1))        # [B, C*p]
            i += 3
         
        # 残差连接
        f1 = torch.cat(features_fe, dim=1)    # [B, w*(2h+5n)]
        f2 = torch.cat(features_p, dim=1)     # [B, 3p*(2h+5n)]
        features = torch.cat([f1, f2], dim=1) # [B, (w+3p)*(2h+5n)]
        
        # 线性层 + 激活 + 失活 + 输出层
        features = self.linear_layer(features)  # [B, 30]
        features = self.relu(features)          # [B, 30]
        features = self.dropout(features)       # [B, 30]
        output = self.output_layer(features)    # [B, 1]

        return output

    
    

# --------------------AlphaNet-v2--------------------

class AlphaNet_v2(nn.Module):
    '''
    研报中的改进版，相比AlphaNet-v1:
        1. 扩充了6比率类特征
        2. 用LSTM替换池化层和全连接层
    '''

    def __init__(self, d=10, stride=10, n=15):
        super(AlphaNet_v2, self).__init__()
        
        # d-回看窗口大小，stride-时间步大小
        self.d = d
        self.stride = stride

        # ts_corr() 和 ts_cov() 输出的特征数量
        h = int(n * (n - 1) / 2)

        # 特征提取层
        self.feature_extractors = nn.ModuleList([
            ts_corr(self.d, self.stride),
            ts_cov(self.d, self.stride),
            ts_stddev(self.d, self.stride),
            ts_zscore(self.d, self.stride),
            ts_return(self.d, self.stride),
            ts_decaylinear(self.d, self.stride)
        ])

        # 特征提取层后面接着的批量归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n)
        ])
        
        
        # 特征总数
        n_in = 2 * h + 4 * n

        # LSTM层，批量归一化，输出层
        self.lstm = nn.LSTM(n_in, 30, 1, batch_first=True)
        self.bn = nn.BatchNorm1d(30)
        self.output_layer = nn.Linear(30, 1)

        # 初始化输出层的权重
        self.initialize_weights()


    def initialize_weights(self):
        # 用 truncated_normal 方法初始化输出层的权重
        nn.init.trunc_normal_(self.output_layer.weight)


    def forward(self, X):

        # 特征提取 + 批量归一化
        features = []
        for extractor, batch_norm in zip(self.feature_extractors, self.batch_norms):
            x = extractor(X)
            x = batch_norm(x)
            features.append(x)
        features = torch.cat(features, dim=1)

        # 将输入转换为: (batch_size, sequence_length, feature_size)
        features = features.transpose(1, 2)

        # LSTM + 批量归一化
        features, _ = self.lstm(features)
        features = features.transpose(1,2)
        features = self.bn(features)
        features = features.transpose(1,2)
         
        # 取LSTM最后一个时间步的隐藏状态作为最后的特征
        features = features[:, -1, :]
        
        # 输出层
        output = self.output_layer(features)

        return output
    
   
    

# --------------------AlphaNet-Attention--------------------

class AlphaNet_att(nn.Module):
    '''
    在AlphaNet-v2的基础上，用Multi-head Self-Attention层替换掉LSTM层
    '''

    def __init__(self, d=10, stride=10, n=15):
        super(AlphaNet_att, self).__init__()

        # d-回看窗口大小，stride-时间步大小
        self.d = d
        self.stride = stride

        # ts_corr() 和 ts_cov() 输出的特征数量
        h = int(n * (n - 1) / 2)

        # 特征提取层
        self.feature_extractors = nn.ModuleList([
            ts_corr(self.d, self.stride),
            ts_cov(self.d, self.stride),
            ts_stddev(self.d, self.stride),
            ts_zscore(self.d, self.stride),
            ts_return(self.d, self.stride),
            ts_decaylinear(self.d, self.stride)
        ])

        # 特征提取层后面接的批量归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n)
        ])
        

        # 特征总数
        n_in = 2 * h + 4 * n
        
        # 多头自注意力机制 + 层归一化
        self.mha = nn.MultiheadAttention(embed_dim=n_in,
                                         num_heads=3,
                                         dropout=0.1,
                                         batch_first=True)
        self.ln = nn.LayerNorm(n_in)
        
        # 输出层
        self.output_layer = nn.Linear(n_in, 1)

        # 初始化输出层的权重
        self.initialize_weights()


    def initialize_weights(self):
        # 用 truncated_normal 方法初始化输出层的权重
        nn.init.trunc_normal_(self.output_layer.weight)


    def forward(self, X):
        
        # 特征提取 + 批量归一化
        features = []
        for extractor, batch_norm in zip(self.feature_extractors, self.batch_norms):
            x = extractor(X)
            x = batch_norm(x)
            features.append(x)
        features = torch.cat(features, dim=1)

        # 将输入转换为: (batch_size, sequence_length, feature_size)
        features = features.transpose(1,2)
        
        # 自注意力计算 + 层归一化
        features, _ = self.mha(features, features, features)
        features = self.ln(features)
        
        # 取所有时间步的均值作为最后的特征
        features = torch.mean(features, dim=1)
        
        # 输出层
        output = self.output_layer(features)

        return output
