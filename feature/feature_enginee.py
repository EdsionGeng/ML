from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
iris.data
iris.target

# 标准化 x=(x-u)/s
StandardScaler().fit_transform(iris.data)

# 区间缩放法 x'=(x-min)/(max-min)
from sklearn.preprocessing import MinMaxScaler

MinMaxScaler.fit_transform(iris.data)
# 归一化　简单来 说，标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下。
# 归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”。规则为l2的归一化公式如下：
# 　使用preproccessing库的Normalizer类对数据进行归一化的代码如下：
from sklearn.preprocessing import Normalizer

Normalizer.fit_transform(iris.data)
# 对定量特征二值化
from sklearn.preprocessing import Binarizer

Binarizer(threshold=3).fit_transform(iris.data)
# 对定性特征哑编码
from sklearn.preprocessing import OneHotEncoder

OneHotEncoder.fit_transform(iris.target.reshape((-1, 1)))
# 缺失值计算
from numpy import vstack, array, nan
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=nan, strategy="mean")
imputer.fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))

# 数据变换
from sklearn.preprocessing import PolynomialFeatures

PolynomialFeatures().fit_transform(iris.data)
# 基于单变元函数的数据变换可以使用统一方式完成，使用preprocessing进行对数函数转换
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer

FunctionTransformer(log1p).fit_transform(iris.data)
"""
类	功能	说明
StandardScaler	无量纲化	标准化，基于特征矩阵的列，将特征值转换至服从标准正态分布
MinMaxScaler	无量纲化	区间缩放，基于最大最小值，将特征值转换到[0, 1]区间上
Normalizer	归一化	基于特征矩阵的行，将样本向量转换为“单位向量”
Binarizer	二值化	基于给定阈值，将定量特征按阈值划分
OneHotEncoder	哑编码	将定性数据编码为定量数据
Imputer	缺失值计算	计算缺失值，缺失值可填充为均值等
PolynomialFeatures	多项式数据转换	多项式数据转换
FunctionTransformer	自定义单元数据转换	使用单变元的函数来转换数据

"""

# 特征选择：
# 方差选择法
from sklearn.feature_selection import VarianceThreshold

VarianceThreshold(threshold=3).fit_transform(iris.data)
# 相关系数法 　使用相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的P值。
# 用feature_selection库的SelectKBest类结合相关系数来选择特征的代码如下：
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。
# 在此定义为计算相关系数
# 参数k为选择的特征个数

SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

# 卡方检验
from sklearn.feature_selection import chi2

# 经典的卡方检验是检验定性自变量对定性因变量的相关性。假设自变量有N种取值，因变量有M种取值，
# 考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距，构建统计量：
SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)

# 互信息法

# 递归特征消除法 递归消除特征法使用一个基模型来进行多轮训练，
# 每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。使用feature_selection库的RFE类来选择特征的代码如下：
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)

# 基于惩罚项 的特征选择法　使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。
# 使用feature_selection库的SelectFromModel类结合带L1惩罚项的逻辑回归模型，来选择特征的代码如下：
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)
from sklearn.decomposition import PCA

PCA(n_components=2).fit_transform(iris.data)

from sklearn.decomposition import LatentDirichletAllocation

LatentDirichletAllocation(n_components=2).fit_transform(iris.data, iris.target)
