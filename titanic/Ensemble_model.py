# 已经由五种模型()得出一次预测结果了
# 再喂入另一种模型做最终预测
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
# 读下ID
test = pd.read_csv('data/test.csv', header=0)
test_idx = pd.DataFrame({'PassengerId': test['PassengerId']})

# 加载5个模型对测试集的预测结果
train = pd.read_csv('data/5train.csv', header=0)
test = pd.read_csv('data/5test.csv', header=0)
y = train['Real y']
train.drop('Real y', axis = 1, inplace = True)

################################ 融合模型接NN ####################################
# # 转一下数据类型
# train_np = np.array(train)
# train_tensor = torch.tensor(train_np).float()   # 注意精度要是float不能是double
# print(train_tensor.shape)
# target_np = np.array(y)
# target_tensor = torch.tensor(y).float()
# test_np = np.array(test)
# test_tensor = torch.tensor(test_np).float()
#
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(train.shape[1], 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64 , 1),
#     torch.nn.Sigmoid()
#     )
#
# optimizer = torch.optim.Adam(model.parameters() , lr = 0.003)
# loss_func = nn.MSELoss()
#
# for i in range(500):
#     y_pred = model(train_tensor)
#     y_pred = torch.squeeze(y_pred, -1)
#     loss = loss_func(y_pred, target_tensor)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if i % 50 == 0:
#         print('Loss:', loss)
#
# output = np.round(model(test_tensor).detach().numpy().ravel())
# y_pred = pd.DataFrame({'Survived': output}).astype(int)
#
# ensemble_summition = pd.concat((test_idx, y_pred), axis=1)
# ensemble_summition.to_csv('output predict/ensemble_output_nn.csv', index=False)
################################ 融合模型接NN ####################################

################################ 融合模型接SVM ####################################
# from sklearn.svm import SVC
#
# model = SVC(kernel='linear', C=0.025)
# model.fit(train, y)
# y_pred = model.predict(test)
# y_pred = pd.DataFrame({'Survived': y_pred})
# print(test_idx.head(), '\n', y_pred.head())
# summition = pd.concat((test_idx, y_pred), axis=1)
# summition.to_csv('output predict/ensemble_output_SVM.csv', index=False)       # 一定要去掉index
################################ 融合模型接SVM ####################################

################################ 融合模型接XGB ####################################
from xgboost.sklearn import XGBClassifier
model = XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    # gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    nthread=-1,
    scale_pos_weight=1
)
# 模型 训练
model.fit(train, y)
# 预测值
y_pred = model.predict(test)
y_pred = pd.DataFrame({'Survived': y_pred})
print(test_idx.head(), '\n', y_pred.head())
summition = pd.concat((test_idx, y_pred), axis=1)
summition.to_csv('output predict/ensemble_output_xgb.csv', index=False)       # 一定要去掉index
################################ 融合模型接XGB ####################################