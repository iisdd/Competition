# 泰坦尼克
一个根据给出信息来判断游客是否生还的二分类问题,具体的步骤为: 

1.数据处理:分析数据统计信息, 填补缺失值(补0,none,随机数,按桶平均)

2.特征工程:手动增加一些特征, 删除一些特征, 把类别特征转化成数字(种类少->Label, 种类多->one-hot)

3.初级模型训练:把预处理过的数据送入第一层的模型(随机森林, 额外树, AdamBoost, GradientBoost, SVM)

4.stacked model:把初级模型的输出作为输入喂给高级模型(XGB)再训练一次

### 收获:

1.初步接触kaggle,熟悉了相关的操作

2.锻炼了数据处理能力(numpy, pandas)

3.学到了Ensemble模型的方法,sklearn的放第一层,XGB,LGB放第二层


# 把连续特征离散化

## 分桶船票价格(按数量均分)
![](https://github.com/iisdd/Competition/blob/main/titanic/upload_pic/Fare%20distribution.png)

## 分桶年龄(按间距分)
![](https://github.com/iisdd/Competition/blob/main/titanic/upload_pic/Age%20distribution.png)

## 称号很多,也把它分桶
![](https://github.com/iisdd/Competition/blob/main/titanic/upload_pic/Title%20of%20Name.png)

## 分桶后的称号
![](https://github.com/iisdd/Competition/blob/main/titanic/upload_pic/merged%20name.png)



* 船厢等级越高(Pclass越小)存活率越高
* 性别为女比性别为男更容易存活
* 年龄越小越容易活
* 船票越贵越容易活
* 名字越长越容易活
* 有船舱的更容易活
* 孤家寡人容易死
* 称号越稀有越容易活
