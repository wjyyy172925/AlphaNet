import pickle

# 读取单个对象
with open('./train_results_v2.pickle', 'rb') as f:
    data = pickle.load(f)

print(data)