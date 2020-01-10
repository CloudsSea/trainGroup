

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as ppf

plt.style.use('ggplot')



from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder  # 标签编码
from sklearn.preprocessing import RobustScaler, StandardScaler  # 去除异常值与数据标准化
from sklearn.pipeline import Pipeline, make_pipeline  # 构建管道
from scipy.stats import skew  # 偏度
# from sklearn.preprocessing import Imputer
import sys

# %% md

## 检视原数据
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder  # 标签编码
from sklearn.preprocessing import RobustScaler, StandardScaler  # 去除异常值与数据标准化
from sklearn.pipeline import Pipeline, make_pipeline  # 构建管道
from scipy.stats import skew  # 偏度


# from sklearn.preprocessing import Imputer

# %% md

## <font color=red>管道建设：pipeline--方便组合各种特征以及对特征的处理，方便后续的机器学习的特征的重做</font>


# %%

##自己写一个转换函数
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    ##对三个年份来进行一个标签编码,这里可以随便自己添加
    def transform(self, X):
        lab = LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        X["BldgType"] = lab.fit_transform(X["BldgType"])

        return X


# %%

# 转换函数
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):  # 偏度
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"])  # 而是去除了包含了对象数据类型，取出来绝大部分是数值型
        skewness = X_numeric.apply(lambda x: skew(x))  # 匿名函数，做成字典的形式
        skewness_features = skewness[abs(skewness) >= self.skew].index  # 通过条件来涮选出skew>=0.5的索引的条件，取到了全部数据，防止数据的丢失
        X[skewness_features] = np.log1p(X[skewness_features])  # 求对数，进一步让他更符合正态分布
        X = pd.get_dummies(X)  ##一键独热，独热编码，（试错经历）
        return X
# %%
if __name__ == '__main__':


    train = pd.read_csv("./train.csv")

    # %%

    test = pd.read_csv("./test.csv")

    # %%

    train.head()  # 默认显示前五行

    # %%

    test.head()



    ## 数据探索性分析 pandas_profiling

    # %%

    ppf.ProfileReport(train)

    # %%

    plt.figure(figsize=(10, 8))
    sns.boxplot(train.YearBuilt, train.SalePrice)  ##箱型图是看异常值的，离群点

    # %%

    plt.figure(figsize=(12, 6))
    plt.scatter(x=train.GrLivArea, y=train.SalePrice)  ##可以用来观察存在线型的关系
    plt.xlabel("GrLivArea", fontsize=13)
    plt.ylabel("SalePrice", fontsize=13)
    plt.ylim(0, 800000)

    # %%

    train.drop(train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index, inplace=True)  # pandas 里面的条件索引

    # %%

    full = pd.concat([train, test], ignore_index=True)

    # %%

    full.drop("Id", axis=1, inplace=True)

    # %%

    full.head()

    # %%

    # full.info()#查看数据的一个信息

    # %% md

    # 数据清洗--空值填充、空值的删除，不处理

    # %%

    ##查看缺失值，并且缺失的个数要从高到低排序

    # %%

    miss = full.isnull().sum()  # 统计出空值的个数

    # %%

    miss[miss > 0].sort_values(ascending=True)  # 由低到高排好序

    # %%

    full.info()

    # %% md

    ## 空值的填充与删除

    # %% md

    # %%

    cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
             "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
             "MasVnrType"]
    for col in cols1:
        full[col].fillna("None", inplace=True)

    # %%

    # %%

    cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
    for col in cols:
        full[col].fillna(0, inplace=True)


    # %%

    full["LotFrontage"].fillna(np.mean(full["LotFrontage"]), inplace=True)



    # %%

    cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType",
             "Exterior1st", "Exterior2nd"]
    for col in cols2:
        full[col].fillna(full[col].mode()[0], inplace=True)

    # %% md


    # %%


    for col in cols2:
        full[col] = full[col].astype(str)  ##astype来进行数据转换成字符串类型

    # %%

    lab = LabelEncoder()

    # %%

    full["Alley"] = lab.fit_transform(full.Alley)
    full["PoolQC"] = lab.fit_transform(full.PoolQC)
    full["MiscFeature"] = lab.fit_transform(full.MiscFeature)
    full["Fence"] = lab.fit_transform(full.Fence)
    full["FireplaceQu"] = lab.fit_transform(full.FireplaceQu)
    full["GarageQual"] = lab.fit_transform(full.GarageQual)
    full["GarageCond"] = lab.fit_transform(full.GarageCond)
    full["GarageFinish"] = lab.fit_transform(full.GarageFinish)
    full["GarageYrBlt"] = full["GarageYrBlt"].astype(str)
    full["GarageYrBlt"] = lab.fit_transform(full.GarageYrBlt)
    full["GarageType"] = lab.fit_transform(full.GarageType)
    full["BsmtExposure"] = lab.fit_transform(full.BsmtExposure)
    full["BsmtCond"] = lab.fit_transform(full.BsmtCond)
    full["BsmtQual"] = lab.fit_transform(full.BsmtQual)
    full["BsmtFinType2"] = lab.fit_transform(full.BsmtFinType2)
    full["BsmtFinType1"] = lab.fit_transform(full.BsmtFinType1)
    full["MasVnrType"] = lab.fit_transform(full.MasVnrType)
    full["BsmtFinType1"] = lab.fit_transform(full.BsmtFinType1)

    # %%

    full.head()

    # %%

    full["MSZoning"] = lab.fit_transform(full.MSZoning)
    full["BsmtFullBath"] = lab.fit_transform(full.BsmtFullBath)
    full["BsmtHalfBath"] = lab.fit_transform(full.BsmtHalfBath)
    full["Utilities"] = lab.fit_transform(full.Utilities)
    full["Functional"] = lab.fit_transform(full.Functional)
    full["Electrical"] = lab.fit_transform(full.Electrical)
    full["KitchenQual"] = lab.fit_transform(full.KitchenQual)
    full["SaleType"] = lab.fit_transform(full.SaleType)
    full["Exterior1st"] = lab.fit_transform(full.Exterior1st)
    full["Exterior2nd"] = lab.fit_transform(full.Exterior2nd)

    # %%

    full.head()

    # %%

    full.drop("SalePrice", axis=1, inplace=True)  ##删除

    # %%

    # full2 = pd.get_dummies(full)##独热编码

    # %%




    # %%

    # 构建管道
    pipe = Pipeline([  ##构建管道的意思
        ('labenc', labelenc()),
        ('skew_dummies', skew_dummies(skew=2)),
    ])

    # %%

    # 保存原来的数据以备后用，为了防止写错
    full2 = full.copy()

    # %%

    pipeline_data = pipe.fit_transform(full2)

    # %%

    pipeline_data.shape

    # %%

    pipeline_data.head()

    # %%

    n_train = train.shape[0]  # 训练集的行数
    X = pipeline_data[:n_train]  # 取出处理之后的训练集
    test_X = pipeline_data[n_train:]  # 取出n_train后的数据作为测试集
    y = train.SalePrice
    X_scaled = StandardScaler().fit(X).transform(X)  # 做转换
    y_log = np.log(train.SalePrice)  ##这里要注意的是，更符合正态分布
    # 得到测试集
    test_X_scaled = StandardScaler().fit_transform(test_X)

    # %% md

    ## <font color= red> 特征的选择--基于特征重要性图来选择</font>

    # %%

    from sklearn.linear_model import Lasso  ##运用算法来进行训练集的得到特征的重要性，特征选择的一个作用是，wrapper基础模型

    lasso = Lasso(alpha=0.001)
    lasso.fit(X_scaled, y_log)

    # %%

    FI_lasso = pd.DataFrame({"Feature Importance": lasso.coef_}, index=pipeline_data.columns)  # 索引和重要性做成dataframe形式

    # %%

    FI_lasso.sort_values("Feature Importance", ascending=False)  # 由高到低进行排序

    # %%

    FI_lasso[FI_lasso["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh", figsize=(15, 25))
    plt.xticks(rotation=90)
    plt.show()  ##画图显示


    # %% md

    ## <font color=red> 得到特征重要性图之后就可以进行特征选择与重做,这里也提供一条其他的思路，特征不重要的要删除</font>

    # %%

    ##大家的发挥空间比较大，可以随意的定制，自己做两种特征
    class add_feature(BaseEstimator, TransformerMixin):  # 自己定义转换函数--fit_transform由自己定义
        def __init__(self, additional=1):
            self.additional = additional

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if self.additional == 1:
                X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
                X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            else:
                X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
                X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

                X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
                X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
                X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
                X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
                X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
                X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
                X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
                X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
                X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]

                X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
                X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
                X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
                X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
                X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
                X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]

                X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
                X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
                X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
                X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X[
                    "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]

                return X


    # %%

    pipe = Pipeline([  # 把后面的东西加到管道里面来
        ('labenc', labelenc()),
        ('add_feature', add_feature(additional=2)),
        ('skew_dummies', skew_dummies(skew=4)),
    ])

    # %%

    pipe

    # %%

    n_train = train.shape[0]  # 训练集的行数
    X = pipeline_data[:n_train]  # 取出处理之后的训练集
    test_X = pipeline_data[n_train:]  # 取出n_train后的数据作为测试集
    y = train.SalePrice
    X_scaled = StandardScaler().fit(X).transform(X)  # 做转换
    y_log = np.log(train.SalePrice)  ##这里要注意的是，更符合正态分布
    # 得到测试集
    test_X_scaled = StandardScaler().fit_transform(test_X)

    # %% md

    ## 模型的构建

    # %%

    from sklearn.tree import DecisionTreeRegressor  # 导入模型

    # %%

    model = DecisionTreeRegressor()

    # %%

    model1 = model.fit(X_scaled, y_log)

    # %% md

    ## 模型训练好了后就对test进行预测

    # %%

    predict = np.exp(model1.predict(test_X_scaled))  ##np.exp是对上面的对数变换之后的反变换

    # %%

    result = pd.DataFrame({'Id': test.Id, 'SalePrice': predict})
    result.to_csv("submission1.csv", index=False)

# %%


# %%


# %%


# %%


