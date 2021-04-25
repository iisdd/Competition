# 使用RF分类器拟合预处理过的数据
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# 读取预处理过的数据
train = pd.read_csv('data/processed_train.csv', header=0)
test_idx = pd.read_csv('data/test.csv', header=0)
test = pd.read_csv('data/processed_test.csv', header=0)
y = train['Survived']
train.drop(["Survived"], axis = 1, inplace = True)
test_idx = pd.DataFrame({'PassengerId': test_idx['PassengerId']})
# print(train.head())
model = RandomForestClassifier(
    n_jobs=-1,                  # 出动的CPU数, -1:有多少用多少
    n_estimators=1500,          # 决策树数目
    warm_start=True,            # 可以添加更多决策器
    max_depth=6,                # 树的最大深度
    min_samples_leaf=2,         # 叶子最小样本数
    max_features='sqrt',
    verbose=0,
)

# 模型 训练
model.fit(train, y)
# 预测值
y_pred = model.predict(test)
y_pred = pd.DataFrame({'Survived': y_pred})
print(test_idx.head(), '\n', y_pred.head())
RF_summition = pd.concat((test_idx, y_pred), axis=1)
RF_summition.to_csv('output predict/RF_output.csv', index=False)       # 一定要去掉index