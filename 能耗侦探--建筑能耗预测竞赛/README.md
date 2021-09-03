# 能耗侦探
2020-05-20

包括两个部分: LSTM教程 & 比赛内容
具体来说就是给出三年的建筑用电(包括两部分:照明&空调),用前两年的数据预测第三年的,最后效果最好的模型是根据前24小时的用电量预测下一小时的(合理!),准确率可以达到93%,如果加入别的特征(例如各种天气数据)应该可以有更好的预测效果.

收获就是:

1.学习了LSTM的基本原理与代码实现 

2.练习了pandas数据处理的基本操作(包括读取,分析,输出csv)


![Loss](https://github.com/iisdd/Competition/blob/main/%E8%83%BD%E8%80%97%E4%BE%A6%E6%8E%A2--%E5%BB%BA%E7%AD%91%E8%83%BD%E8%80%97%E9%A2%84%E6%B5%8B%E7%AB%9E%E8%B5%9B/upload_pic/Loss.png)
Epoch=2000, Batch_Size=16,最后结果

Train Score: 36.62 RMSE

Test Score: 36.95 RMSE




