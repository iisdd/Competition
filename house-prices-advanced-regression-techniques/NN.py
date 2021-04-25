# 效果很差,房价区别不明显
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # 激励函数
import matplotlib.pyplot as plt


# 加载预处理过的数据
df_train = pd.read_csv('data/processed_train.csv', index_col='Id')
df_test = pd.read_csv('data/processed_test.csv', index_col='Id')

################### 数据预处理 ######################
target = df_train['SalePrice']
df_train = df_train.drop('SalePrice', axis=1)   # 把房价那一列挖掉


# df -> np -> tensor
train_np = np.array(df_train)
train_tensor = torch.tensor(train_np).float()   # 注意精度要是float不能是double
target_np = np.array(target)
target_tensor = torch.tensor(target_np).float()
test_np = np.array(df_test)
test_tensor = torch.tensor(test_np).float()
################### 数据预处理 ######################

n_features = df_train.shape[1]
print(n_features)       #

################### 搭建网络 ######################
model = torch.nn.Sequential(
    torch.nn.Linear(n_features ,256),
    torch.nn.ReLU(),
    torch.nn.Linear(256 , 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256 ,1),
)

optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
loss_func = nn.MSELoss()
################### 搭建网络 ######################


################### 训练网络 ######################
cost = []
for t in range(2000):
    prediction = model(train_tensor)
    # print(prediction.shape, target_tensor.shape)
    loss = loss_func(prediction , target_tensor)

    optimizer.zero_grad() # 梯度清零,防止爆炸
    loss.backward() # 误差反向传递
    optimizer.step()
    if t % 20 == 0:
        # 显示学习的过程
        print(t, loss)
        cost.append(loss)

plt.plot(cost)
plt.show()
################### 训练网络 ######################

################### 输出结果 ######################
preds = model(test_tensor).detach().numpy().ravel()
preds = np.expm1(preds)                                             # 还原成对数变换之前的
print(preds.shape)
my_submission = pd.DataFrame({'Id': df_test.index+2, 'SalePrice': preds})

my_submission.to_csv('output/NN.csv', index=False)        # index=False: 不存索引, 否则会多出一列来

################### 输出结果 ######################
