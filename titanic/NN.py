# 使用神经网络拟合预处理过的数据
# 需要实现的功能:
# 1.将带标签的数据分割为训练集&验证集(sklearn)
# 2.自动调参训练找出最好的参数(grid search)

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold   # 分割训练集和验证集
kf = KFold(n_splits=5, random_state=None)   # random_state是随机种子


# 读取预处理过的数据
train = pd.read_csv('data/processed_train.csv', header=0)
test_idx = pd.read_csv('data/test.csv', header=0)
test = pd.read_csv('data/processed_test.csv', header=0)
y = train['Survived']
train.drop(["Survived"], axis = 1, inplace = True)
test_idx = pd.DataFrame({'PassengerId': test_idx['PassengerId']})

# # 轮流当test set
# for train_idx, test_idx in  kf.split(train, y):
#     print('#' * 80)
#     print(train_idx, test_idx)
#     print('#' * 80)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def create_model(n_hidden=32, lr=0.001):
    # create model
    model = Sequential()
    model.add(Dense(n_hidden, input_dim=train.shape[1], activation='relu'))       # 第一层要规定输入数据长度
    model.add(Dense(units=n_hidden, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))       # 2分类
    optimizer = Adam(lr=lr)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   # 二分类损失函数为交叉熵
    return model

# print(model.summary())      #打印网络结构

# model = KerasClassifier(build_fn=create_model, verbose=0)
#
# N_HIDDENS = [64, 128, 256]
# EPOCHS = [300, 500, 1000]
# LR = [0.001, 0.003, 0.01, 0.03]
# param_grid = dict(n_hidden = N_HIDDENS, nb_epoch = EPOCHS, lr = LR)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(train, y)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#
#
# means = grid_result.cv_results_['mean_test_score']
# params = grid_result.cv_results_['params']
# for mean,param in zip(means,params):
#     print("%f  with:   %r" % (mean,param))

# Best: 0.769939 using {'lr': 0.01, 'n_hidden': 64, 'nb_epoch': 1000}


model = create_model(n_hidden=64, lr=0.01)

model.fit(train, y, epochs=1000, verbose=1)

y_pred = model.predict(test).ravel()
res = np.zeros(y_pred.shape)
res[y_pred>0.5] = 1
# print(type(res[0]))     # <class 'numpy.float64'>
y_pred = pd.DataFrame({'Survived': res}).astype(int)        # 转换下数据类型,不然打0分

NN_summition = pd.concat((test_idx, y_pred), axis=1)
NN_summition.to_csv('output predict/NN_output.csv', index=False)

