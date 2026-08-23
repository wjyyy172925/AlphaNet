import numpy as np
from scipy import stats

print("=" * 60)
print("Spearman相关系数计算方法的等价性验证")
print("=" * 60)

# 模拟数据：5只股票的真实收益率和模型预测值
Y_true = np.array([0.05, 0.12, -0.03, 0.08, 0.01])      # 真实收益率
Y_pred = np.array([0.04, 0.10, 0.02, 0.07, 0.00])      # 模型预测值

print("\n【原始数据】")
print(f"真实收益率:      {Y_true}")
print(f"模型预测值:      {Y_pred}")

# ============================================================
# 方法1：直接使用 spearmanr（最简洁）
# ============================================================
corr1, p1 = stats.spearmanr(Y_true, Y_pred)

print("\n" + "=" * 60)
print("方法1：直接使用 stats.spearmanr(Y_true, Y_pred)")
print(f"  结果: 相关系数 = {corr1:.6f}, p值 = {p1:.6f}")
print("=" * 60)

# ============================================================
# 方法2：手动计算排名后，使用 spearmanr
# ============================================================
rank_true = np.argsort(Y_true).argsort() + 1
rank_pred = np.argsort(Y_pred).argsort() + 1
corr2, p2 = stats.spearmanr(rank_true, rank_pred)

print("\n" + "=" * 60)
print("方法2：手动计算排名后，使用 stats.spearmanr(rank_true, rank_pred)")
print(f"  真实收益率的排名: {rank_true}")
print(f"  模型预测的排名:   {rank_pred}")
print(f"  结果: 相关系数 = {corr2:.6f}, p值 = {p2:.6f}")
print("=" * 60)

# ============================================================
# 方法3：手动计算排名后，使用 pearsonr（Spearman的本质）
# ============================================================
corr3, p3 = stats.pearsonr(rank_true, rank_pred)

print("\n" + "=" * 60)
print("方法3：手动计算排名后，使用 stats.pearsonr(rank_true, rank_pred)")
print(f"  结果: 相关系数 = {corr3:.6f}, p值 = {p3:.6f}")
print("=" * 60)

# ============================================================
# 验证等价性
# ============================================================
print("\n" + "=" * 60)
print("【验证结果】")
print(f"  方法1 == 方法2? {np.isclose(corr1, corr2)}")
print(f"  方法1 == 方法3? {np.isclose(corr1, corr3)}")
print(f"  方法2 == 方法3? {np.isclose(corr2, corr3)}")
print("=" * 60)

# ============================================================
# 额外测试：展示排名计算过程
# ============================================================
print("\n" + "=" * 60)
print("【排名计算过程详解】")
print("=" * 60)

print("\n1. 真实收益率 Y_true = [0.05, 0.12, -0.03, 0.08, 0.01]")
print("   对应股票:         A     B      C     D     E")
print("\n2. 从小到大排序:")
print("   [-0.03, 0.01, 0.05, 0.08, 0.12]")
print("   对应:    C,    E,    A,    D,    B")
print("   排名:    1,    2,    3,    4,    5")
print("\n3. 映射回原始股票顺序:")
print(f"   真实收益率排名: {rank_true}")
print("   解释: A排第3, B排第5, C排第1, D排第4, E排第2")