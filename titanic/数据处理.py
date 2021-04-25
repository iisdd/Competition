#################################### 导入包 #######################################
import pandas as pd
import numpy as np
import re  # 正则化表达
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import missingno as mg  # 用于查看缺失值
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')

# 用5个模型做第一层融合
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
#################################### 导入包 #######################################

#################################### 加载&分析数据 #######################################
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

y = train['Survived'].ravel()  # 保存标签
y = pd.DataFrame({'Survived': y})

PassengerId = test['PassengerId']  # 把测试集index取出来

print(f'训练集有{train.shape[0]}行{train.shape[1]}列')
print(f'测试集有{test.shape[0]}行{test.shape[1]}列')
# 测试集没有结果(是否存活)
ntrain = train.shape[0]
ntest = test.shape[0]

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['Survived', 'PassengerId'], axis=1, inplace=True)


print('\n数据性分析:')
print(all_data.describe().T)  # 数据性分析(不包括文字特征)


# 统计训练集特征变量的属性和内存
print('\n变量类型:')
print(all_data.info())

# 使用missingno进行缺失值可视化
mg.matrix(all_data)
plt.show()


# 统计每个变量缺失的比例
def missing_percentage(df):
    # 总缺失值由高到低排序
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().
                                                             sum().sort_values(ascending = False) != 0]
    # 缺失百分比
    percentage = round(df.isnull().sum().sort_values(ascending = False) / len(df) * 100, 2)[df.isnull().
                                                            sum().sort_values(ascending = False) != 0]
    # axis=1: 横着并拢
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])
print('\n各特征缺失比例:')
print(missing_percentage(all_data))
#################################### 加载&分析数据 #######################################

############################### 填补缺失值 #####################################
# Cabin缺太多,直接用新特征有无Cabin代替,所以只用填三个: Fare, Embarked, Age
# 1.Fare缺一个(测试集里),用test的中位数填充
all_data['Fare'] = all_data['Fare'].fillna(test['Fare'].median())
# 2.Embarked缺一个,直接用S填充
all_data['Embarked'] = all_data['Embarked'].fillna('S')
# 3.Age,用随机数填补
age_avg = all_data['Age'].mean()
age_std = all_data['Age'].std()
age_null_count = all_data['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
age_nan_index = np.isnan(all_data['Age'])
all_data['Age'][age_nan_index] = age_null_random_list
all_data['Age'] = all_data['Age'].astype(int)


# 检查填补情况, Cabin不管
print('\n填补后的缺失比例:')
print(missing_percentage(all_data))
############################### 填补缺失值 #####################################

############################### 特征工程 #####################################
# 新增一些特征

# 改画风以及设置图片大小代码!!!
style.use('fivethirtyeight')

# 1.名字长度
all_data['Name_length'] = all_data['Name'].apply(len)
# 2.是否有舱, null即代表没有舱
all_data['Has_Cabin'] = 1 - all_data['Cabin'].isnull()
# 高级写法, null是float数据类型
# all_data['Has_Cabin'] = all_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# 3.家族规模, 兄弟姐妹,伴侣,爸妈小孩加自己
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
# 4.孤身一人
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1
# 5.票价档次(均分4档)
all_data['CategoricalFare'] = pd.qcut(all_data['Fare'], 4)

plt.figure()
sns.countplot(all_data['CategoricalFare'])
plt.title('Fare distribution')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()
# pd.qcut: 数据分箱,把连续型数字均分成4个离散档次(0%~25%, 25%~50%, 50%~75%, 75%~100%),适用于上下限分布不均的数据
# pd.cut: 等距分箱,把连续型数字成4个等距离离散档次(0~25岁, 25~50, 50~75, 75~100),适用于均匀分布的数据集
# 6.年龄(等距分箱)
all_data['CategoricalAge'] = pd.cut(all_data['Age'], 5)
plt.figure()
sns.countplot(all_data['CategoricalAge'])
plt.title('Age distribution')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()

# 7.称号(Ex:船长, 服务员, 博士...)
# 取出.前面(包括.)的字符串
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # re.search:找出第一个符合正则表达式的字符串,如果没找到返回None
    if title_search:
        return title_search.group(1)
    return ""
all_data['Title'] = all_data['Name'].apply(get_title)
# 画直方图看下哪些称号是冷门称号

plt.figure()
sns.countplot(all_data['Title'])  # 在框架里画画
plt.title('Title of Name')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()


# 把冷门称号统一为Rare, (Countess:伯爵夫人)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
# 归并到同义词
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
# 再看一下分布情况
fig, ax1 = plt.subplots(constrained_layout = True, figsize=(18,12))
sns.countplot(all_data['Title'], ax=ax1)
plt.title('merged name')
plt.show()


# 手动将类别特征数字化
# 1.性别
all_data['Sex'] = all_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# 2.称号
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
all_data['Title'] = all_data['Title'].map(title_mapping)
# 3.登船点
all_data['Embarked'] = all_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# 以下的4&5的分法由上面的图分析得来的
# 4.船票钱(分4档)
# .loc[行, 列] = 定位赋值
all_data.loc[all_data['Fare'] <= 7.91, 'Fare'] = 0
all_data.loc[(all_data['Fare'] > 7.91) & (all_data['Fare'] <= 14.454), 'Fare'] = 1
all_data.loc[(all_data['Fare'] > 14.454) & (all_data['Fare'] <= 31), 'Fare'] = 2
all_data.loc[all_data['Fare'] > 31, 'Fare'] = 3
# 字符串 -> 整型
all_data['Fare'] = all_data['Fare'].astype(int)
# 5.年龄(分5档)
all_data.loc[all_data['Age'] <= 16, 'Age'] = 0
all_data.loc[(all_data['Age'] > 16) & (all_data['Age'] <= 32), 'Age'] = 1
all_data.loc[(all_data['Age'] > 32) & (all_data['Age'] <= 48), 'Age'] = 2
all_data.loc[(all_data['Age'] > 48) & (all_data['Age'] <= 64), 'Age'] = 3
all_data.loc[all_data['Age'] > 64, 'Age'] = 4 ;
print('\n数字化后的特征:')
print(all_data.head())


# 删除一些没用的特征
drop_features = ['Name', 'Ticket', 'Cabin', 'SibSp', 'CategoricalAge', 'CategoricalFare']
all_data = all_data.drop(drop_features, axis=1)
print('\n删减后的特征:')
print(all_data.head())

# 看一下哪些因素和生存关系比较紧密
train = all_data[:ntrain]
test = all_data[ntrain:]

train_with_survived = pd.concat((train, y), axis=1)
print(train_with_survived.head())
# 画一个多重共线性(train.corr())的关系的热力图
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize=(18, 12))
# 搞一个掩码mask盖住上三角部分
mask = np.zeros_like(train_with_survived.corr(), dtype=np.bool)
# np.triu_indices_from(mask)  # 返回上三角矩阵的idx
mask[np.triu_indices_from(mask)] = True  # 上三角全遮掉
sns.heatmap(train_with_survived.corr(), cmap=sns.diverging_palette(20, 220, n=200),  # 调色
           mask=mask, annot=True, # 在格子里写入数字
           center=0,  # 热力图中间的颜色
           )
plt.title('heatmap of features', fontsize=30)
plt.show()

# 输出切好的数据
train_with_survived.to_csv('data/processed_train.csv', index=False)
test.to_csv('data/processed_test.csv', index=False)


############################## stack model的素材 #######################################
SEED = 0  # 保证结果一样
NFOLDS = 5  # 每次切分训练集和验证集比例4比1
kf = KFold(NFOLDS, shuffle=False)


# sklearn融合模型通用训练模块
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):  # train不返回模型
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):  # fit返回模型
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):  # 看哪些特征比较关键
        print(self.clf.fit(x, y).feature_importances_)


# 切分训练集和验证集
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):  # 循环洗牌产生新的训练集&测试集对
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)  # 用随出来的这一组训练集训练

        oof_train[test_index] = clf.predict(x_te)  # 填空: 5次训练的合集在训练集上的表现
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)  # 5个模型的test输出取平均
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# 训练参数
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',  # 线性分割,即SVM
    'C' : 0.025
    }


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# 存一份numpy
y_train = y['Survived'].ravel()
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# 训练5种模型的预测结果作为新特征
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)    # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)     # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)     # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test)  # Support Vector Classifier


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
     'GradientBoost': gb_oof_train.ravel(),
     'Support Vector Classifier': svc_oof_train.ravel(),
     'Real y': y_train,
    })
base_predictions_test = pd.DataFrame( {'RandomForest': rf_oof_test.ravel(),
     'ExtraTrees': et_oof_test.ravel(),
     'AdaBoost': ada_oof_test.ravel(),
     'GradientBoost': gb_oof_test.ravel(),
     'Support Vector Classifier': svc_oof_test.ravel(),
    })
print(base_predictions_train.head())
print(base_predictions_train.shape)
# print(et_oof_test.shape)
base_predictions_train.to_csv('data/5train.csv', index=False)
base_predictions_test.to_csv('data/5test.csv', index=False)
# 完成了五个模型对测试集的预测,在把他们作为特征送入XGB
############################## stack model的素材 #######################################