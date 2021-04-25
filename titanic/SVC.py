# 使用SVC分类器拟合预处理过的数据
# 所谓SVC即非线性SVM, 二分类分割线可以是曲线了, 但是本例中用的是linear的kernel, 所以其实就是SVM
# SVC中的参数C越大, 对于训练集来说, 其误差越小, 但是很容易发生过拟合, C越小，则允许有更多的训练集误分类
from sklearn.svm import SVC
import pandas as pd
# 读取预处理过的数据
train = pd.read_csv('data/processed_train.csv', header=0)
test_idx = pd.read_csv('data/test.csv', header=0)
test = pd.read_csv('data/processed_test.csv', header=0)
y = train['Survived']
train.drop(["Survived"], axis = 1, inplace = True)
test_idx = pd.DataFrame({'PassengerId': test_idx['PassengerId']})
# print(train.head())
model = SVC(kernel='linear', C=0.025)

# 模型 训练
model.fit(train, y)
# 预测值
y_pred = model.predict(test)

y_pred = pd.DataFrame({'Survived': y_pred})
print(test_idx.head(), '\n', y_pred.head())
SVC_summition = pd.concat((test_idx, y_pred), axis=1)
SVC_summition.to_csv('output predict/SVC_output.csv', index=False)       # 一定要去掉index

