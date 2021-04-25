import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew  # norm用来生成正态分布

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

from subprocess import check_output


############################## 数据加载&预处理 ###################################

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print("\n训练集形状 : {} ".format(train.shape))
print("测试集形状 : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print('删掉Id后:')
print("\n训练集形状 : {} ".format(train.shape))
print("测试集形状 : {} ".format(test.shape))


def customized_num_scatterplot(y, x, title=None):
    style.use('fivethirtyeight')  # 好看的画风
    plt.subplots(figsize=(12, 8))  # (宽, 高)
    ax = sns.scatterplot(y=y, x=x)
    ax.set_title(title)

customized_num_scatterplot(train.SalePrice, train.GrLivArea, title='Before Deleting outliers')

# 删除离群点,居住面积太大并且房价太便宜的
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


customized_num_scatterplot(train.SalePrice, train.GrLivArea, title='After Deleting outliers')
plt.show()
############################## 数据加载&预处理 ###################################
############################## 观察数据分布 ###################################
style.use('fivethirtyeight')
# 1.找出最接近的norm分布曲线
sns.distplot(train['SalePrice'] , fit=norm);
plt.title('SalePrice before normalized')
(mu, sigma) = norm.fit(train['SalePrice'])
print('正态化之前房价的分布拟合:')
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.show()
# 2.用QQ图判断数据是否为正态分布,蓝点和红线越重合就越符合正态分布
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
stats.probplot(train['SalePrice'], plot=ax)
plt.show()
# 对房价取log让它趋近于正态分布
train['SalePrice'] = np.log1p(train['SalePrice'])

# 3.变换后的房价曲线
sns.distplot(train['SalePrice'] , fit=norm);
plt.title('SalePrice after normalized')

(mu, sigma) = norm.fit(train['SalePrice'])
print('正态化之后房价的分布拟合:')
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.show()


# 先合并找出缺失比例最多的前20名
n_train = train.shape[0]
n_test = test.shape[0]
y_train = train.SalePrice.values  # pd->np,把log后的y抽出来
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop('SalePrice', axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
print('总共的缺失属性数: ', all_data_na.shape[0], '个')
# drop()括号里要跟index,
# print(all_data_na)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data)

fig, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='45')  # 特征文字的倾斜角度
ax.set_facecolor('white')  # 图中的背景色
sns.barplot(x=all_data_na.index, y=all_data_na)
sns.color_palette('rocket', as_cmap=True)  # cmap:数字到颜色的映射
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()

# 数据关联性分析
corrmat = train.corr()  # 只有数字变量能求关联性,一共37个
# 画热力图
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, cmap="YlGnBu", vmax=0.9, square=True)
plt.show()

############################## 观察数据分布 ###################################

############################## 填补缺失值 ###################################
# 一共34个有缺失值的属性
# 1.那种缺失就代表没有的变量,把NA代表没有的替换成None
# 第一种填None的(15个)
fill_None = ["FireplaceQu", "PoolQC", "MiscFeature", "Alley", "Fence",
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual',
             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', "MasVnrType"]

for col in (fill_None):
    all_data[col] = all_data[col].fillna('None')

# 2.按桶分类用中位数代替缺失值(1个)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# 3.填0(10个)
fill_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
             'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', "MasVnrArea"]

for col in (fill_zero):
    all_data[col] = all_data[col].fillna(0)

# 4.填众数(6个)
fill_mode = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd',
             'SaleType']

for col in (fill_mode):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# 5.直接丢掉特征(1个)
all_data = all_data.drop(['Utilities'], axis=1)

# 6.填入指定值(1个)
all_data["Functional"] = all_data["Functional"].fillna("Typ")


# 检查下还有没有缺失值了
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print('填充后的缺失值:')
print(missing_data)
############################## 填补缺失值 ###################################

############################## 特征工程 ###################################
# 比较少的类型变量变成label(一列搞定)
# 比较多的类型变成展开成one-hot(变成很多列)
# 户型从数字变成种类
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# 总体情况有个打分也从数字变成种类
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# 售出年月也改成类别
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# 把n个种类标签化成(0~n-1)的数字
# 与one-hot不同,one-hot会把性别(男or女)拆成两个变量,Label会把男=0,女=1
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual',
        'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure',
        'GarageFinish', 'LandSlope','LotShape', 'PavedDrive', 'Street',
        'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold',
        'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape
print('Shape all_data: {}'.format(all_data.shape))

# 加一个新特征,总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# 看一下特征的偏度
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("数值特征的偏度:")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness
# 修正偏度
# 修正偏斜的特征
skewness = skewness[abs(skewness) > 0.75]
print("一共有{}个偏斜特征".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15  # 估计出来的
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)



# Label的搞定了,现在来处理one-hot的(get_dummies)
all_data = pd.get_dummies(all_data)
print(all_data.shape)  # (扩展成220个特征)

train = all_data[:n_train]
test = all_data[n_train:]

# 存一下处理过的训练集和测试集
idx_all = pd.DataFrame({'Id': all_data.index+1})
all_id = pd.concat((idx_all, all_data), axis=1)  # 带Id的全数据集

processed_train = pd.concat((all_id[:n_train], pd.DataFrame({'SalePrice': y_train})), axis=1)
processed_test = all_id[n_train:]

# 输出处理完的数据
processed_train.to_csv('data/processed_train.csv', index=False)
processed_test.to_csv('data/processed_test.csv', index=False)
############################## 特征工程 ###################################
