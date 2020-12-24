#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 
 @IsAvailable: 
 @Time       : 2020/12/24 18:50
 @Author     : 剑怜情
'''
import pandas as pd
from mxnet import nd

# 1 # 读取数据 / 练习==============================================
csv_path = "D:\plantainz\OuO_DeepLearning\DataSet\housePrices"
train_data = pd.read_csv(csv_path + '\\train.csv')
test_data = pd.read_csv(csv_path + '\\test.csv')
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

# 2 # 预处理
# > 连续 + 离散处理 > 重组
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index
print(type(all_features.dtypes))
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean()) / x.std()
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features,dummy_na=True)
print(all_features.shape)

n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values)
train_labels = train_labels.reshape((-1,1))

# ooooo走你！










