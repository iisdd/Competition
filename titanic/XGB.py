# 使用XGB分类器拟合预处理过的数据
from xgboost.sklearn import XGBClassifier
import pandas as pd
# 读取预处理过的数据
train = pd.read_csv('data/processed_train.csv', header=0)
test_idx = pd.read_csv('data/test.csv', header=0)
test = pd.read_csv('data/processed_test.csv', header=0)
y = train['Survived']
train.drop(["Survived"], axis = 1, inplace = True)
test_idx = pd.DataFrame({'PassengerId': test_idx['PassengerId']})
# print(train.head())
model = XGBClassifier(
    # 树的个数
    n_estimators=3000,
    # 如同学习率
    learning_rate=0.001,
    # 构建树的深度，越大越容易过拟合
    max_depth=4,
    # 随机采样训练样本 训练实例的子采样比
    subsample=0.8,
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    min_child_weight=2,
    # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子
    gamma=0.9,
    # 生成树时进行的列采样
    colsample_bytree=0.8,
    eval_metric = 'logloss',
)

# 模型 训练
model.fit(train, y)
# 预测值
y_pred = model.predict(test)
y_pred = pd.DataFrame({'Survived': y_pred})
print(test_idx.head(), '\n', y_pred.head())
XGB_summition = pd.concat((test_idx, y_pred), axis=1)
XGB_summition.to_csv('output predict/XGB_output.csv', index=False)       # 一定要去掉index

