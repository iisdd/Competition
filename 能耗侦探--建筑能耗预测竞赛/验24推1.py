# 目前为止最佳结果,92.8%,剩余可能超越的就是多变量了
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import  os

np.random.seed(7)

###################################数据提取部分#######################################################
# 先提取出来训练集和测试集,先对照明进行预测,取 2015-2016年建筑 0的数据为 training_data,2017年的 test_data
dataset = pd.read_csv('train.csv', usecols=[3], engine='python')
dataset = dataset.iloc[:52608,0]  # 取2015-2017年的建筑0的数据,这里不用 3:4 直接降维变成一维(52608 , )
dataset = np.array(dataset,dtype='float32')
training_data , test_data = [] , []
#print(alldata[-1])
# 先把照明能耗Q的数据拿出来
Q_data = []
for idx , val in enumerate(dataset):
    if idx % 2 == 0:
        Q_data.append(val)

# # 看下耗电平均值的变化
# print('2015年平均每日耗电 : ' , np.mean(Q_data[ :8760])) # 429
# print('2016年平均每日耗电 : ' , np.mean(Q_data[8760 : 2*8760+24])) # 381
# print('2017年平均每日耗电 : ' , np.mean(Q_data[2*8760+24 : ])) # 370

################# 缩放数据 ###############
# 归一化 Q_data,因为 LSTM基本用的都是tanh激活函数,让他处在梯度正常的范围
scaler = MinMaxScaler(feature_range=(0, 1))
Q_data = np.reshape(Q_data , (-1 , 1))  # 升维才能送进归一化
Q_data = scaler.fit_transform(Q_data)
################# 缩放数据 ###############

training_data = Q_data[ : 2*8760+24]
training_data = np.reshape(training_data , (-1 , 1)) # 变成二维才能送进 Create_dataset
test_data = Q_data[2*8760+24 : ]
test_data = np.reshape(test_data , (-1 , 1))


# 输入数据集,输出 data_X(目标的前look_back个数据) , data_Y(目标数据)
def Create_dataset(dataset,look_back=1): # 验一推一
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back - 1 ):
        a = dataset[i:(i + look_back) , 0]   # 根据 look_back个数据来预测
        data_X.append(a)
        data_Y.append(dataset[i + look_back , 0])  # 保持 Y 也是二维
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    return  data_X,data_Y

# 预测步长为 24 , 验24推一, 24 -> 1
look_back = 24
train_X , train_Y = Create_dataset(training_data , look_back)
test_X , test_Y = Create_dataset(test_data , look_back)
#print(train_X.shape) # (17542 , TIME_STEP)


# 重构输入数据格式 [samples, time_steps , look_back] = [17542,1,1]  注意:time_steps != look_back
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

###################################数据提取部分#######################################################


###################################LSTM部分##########################################
# 构建 LSTM 网络
EPOCHS = 1000
BATCH_SIZE = 16  # 这个太小就太慢了

def Train_Model(train_X,train_Y): # 喂入的数据为三维,(m , time_step , features)
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))  # input_shape = (time_step , features)
    #model.add(LSTM(batch_input_shape=(BATCH_SIZE , 1, look_back) , units=4 , stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_split=0.1)
    # verbose = 2 : 每一个epoch打印一次loss和val_loss , 抽了10%去当交叉验证集
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return model

###################################LSTM部分##########################################
# model = Train_Model(train_X,train_Y)
# # 保存一下训练结果
# model.save("训练结果/lstm_{}_{}_{}_model.h5".format(EPOCHS,BATCH_SIZE,look_back))

# 加载模型
from keras.models import load_model # 牛!
model = load_model('训练结果/lstm_{}_{}_{}_model.h5'.format(EPOCHS,BATCH_SIZE,look_back))

# 对训练数据的Y进行预测
train_predict = model.predict(train_X)
# 对测试数据的Y进行预测
test_predict  = model.predict(test_X)
# 对数据进行逆缩放(反归一化)
train_predict = scaler.inverse_transform(train_predict)
train_Y = scaler.inverse_transform([train_Y])
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

# 计算RMSE误差
trainScore = math.sqrt(mean_squared_error(train_Y[0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_Y[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# 计算准确率
#print(test_predict.shape) # (8758,1)
#print(test_Y.shape)  # (1 , 8758)
accuracy = np.mean(1 - (np.abs(test_predict - test_Y.T)/test_Y.T))
print('测试集准确率 : ' , accuracy)
plt.plot(test_predict[ : 30 , :])
plt.plot(test_Y.T[ : 30 , :])
plt.title('part of test set')
plt.show()

print(test_Y[0])
print(test_predict.T[0])
output = pd.DataFrame({'real data': test_Y[0], 'predict data': test_predict.T[0]})

print(output.head())
output.to_csv('predict.csv', index=False)


# 画图
# 先把x轴坐标移到位
# 构造一个和Q_data格式相同的数组，共26302行，dataset为总数据集，把预测的17542行训练数据存进去
trainPredictPlot = np.empty_like(Q_data)
# 用nan填充数组
trainPredictPlot[ : , : ] = np.nan
# 将训练集预测的Y添加进数组，从第1位到第17542位，共93行
trainPredictPlot[look_back:len(train_predict)+look_back , : ] = train_predict

# 构造一个和dataset格式相同的数组，共26302行，把预测的后8760行测试数据数据放进去
testPredictPlot = np.empty_like(Q_data)
testPredictPlot[:] = np.nan
# 将测试集预测的Y添加进数组，从第17543位到最后26302行，共8760行
testPredictPlot[len(train_predict)+(look_back*2)+1:len(Q_data)-1 , : ] = test_predict
plt.plot(scaler.inverse_transform(Q_data))  # 三年的数据
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(['origin data', 'train predict', 'test predict'])
plt.title('All data')
plt.savefig('predict.jpg', dpi=800)
plt.show()
