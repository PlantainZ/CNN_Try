#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 房价预测 简洁版本
 @IsAvailable: false
 @Time       : 2020/12/24 18:50
 @Author     : 剑怜情
'''

import pandas as pd
from mxnet import nd,gluon,autograd
from mxnet.gluon import loss as gloss,nn,data as gdata
import d2lzh as d2l

# 1 # 读取数据 / 练习==============================================
csv_path = "D:\plantainz\OuO_DeepLearning\DataSet\housePrices" # company
# csv_path = "C:\XXXX_Learning\XX_DataSet\kaggle_house_price" # home

train_data = pd.read_csv(csv_path + '\\train.csv')
test_data = pd.read_csv(csv_path + '\\test.csv')
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))


# 2 # 预处理======================================================
# > 连续 + 离散处理 > 重组
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index
# print(all_features.dtypes)
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean()) / x.std()
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features,dummy_na=True)
# print(all_features.shape)

n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values)
train_labels = train_labels.reshape((-1,1))

# ooooo走你！=====================================================
loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

def log_rmse(net,features,labels):
    clipped_preds = nd.clip(net(features),1,float('inf'))
    rmse = nd.sqrt(2*loss(clipped_preds.log(),labels.log()).mean())
    return rmse.asscalar()

def train(train_features,train_labels,
          test_features,test_labels,
          net,num_epochs,batch_size,weight_decay,learning_rate):
    train_ls,test_ls = [],[]
    train_iter = gdata.DataLoader(
        gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True
    )

    trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':learning_rate,'wd':weight_decay})

    for epoch in range(num_epochs):
        for X,y in range(train_iter):
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net,train_features,train_labels))

        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

# 5 # 模型选择 ==============================================================
# > 设置超参数 > 产出
def get_k_fold_data(k,i,X,y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train,y_train = None,None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        X_part,y_part =











