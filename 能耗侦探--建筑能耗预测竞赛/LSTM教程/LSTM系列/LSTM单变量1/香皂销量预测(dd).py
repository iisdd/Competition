# 这个例子有点搞笑嗷,就是求了个延迟为 1的数据标准差
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from math import sqrt
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m') # 每个月一个数据
series = read_csv('data_set/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# summarize first few rows
print(series.head())
# line plot
series.plot()
pyplot.show()

# 分割训练集与测试集,24个训练,12个测试
X = series.values
train , test = X[0 : -12] , X[-12 : ]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()




