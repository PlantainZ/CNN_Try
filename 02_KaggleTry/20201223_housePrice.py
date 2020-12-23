#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 
 @IsAvailable: 
 @Time       : 2020/12/23 9:30
 @Author     : 剑怜情
'''
import d2lzh
import numpy as np
import pandas as pd
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import data as gdata,loss as gloss,nn


# 1 # 读取数据集，并观察数据=================================================
csv_path = "D:\plantainz\OuO_DeepLearning\DataSet\housePrices"
train_data = pd.read_csv(csv_path + "\\train.csv")
test_data = pd.read_csv(csv_path + "\\test.csv")
print("1. 看一眼数据集的形状：")
print("train_data:",train_data.shape)
print("test_data:",test_data.shape)

print("\n2. 看一眼数据集前4个样本的，前四个特征 + 后两个特征 + 标签：")
print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])

# 忽略第一列的id,把测试数据 放到 训练数据的下方，连接成一排数据。看行数！！
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
print('all_features :',all_features)


# 2 # 预处理数据集 ==========================================================
# 进行标准化。均值为μ，标准差为~。
# 每个值先减去μ再除以~
# 缺失的特征集就替换为均值！
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # 获取值为数字的 特征名称
print('numeric_features:',numeric_features)
print('numeric_features.shape:',numeric_features.shape)

# 这里的all_features，选了所有纵列。 【为什么？】
all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x - x.mean()) / (x.std()))
print('\nall_features[numeric_features]:',all_features[numeric_features])
print('all_features[\'MSSubClass\']:',all_features['MSSubClass'])

# 标准化后，每个特征的均值都变成0，所以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 离散数值转为指示特征。
all_features = pd.get_dummies(all_features,dummy_na=True)
print('after pd.get_dummies,all_features.shape:',all_features.shape) # 总特征数直接从79 -> 331

# 通过values属性得到numpy格式的数据，并转换为ndarray
n_train = train_data.shape[0]
print('n_train:',n_train)

# 重新再分割训练 & 测试数据
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
print('\n Then we get dataSet: ================================================')
print('train_features:',train_features)
print('test_features:',test_features)

train_labels = nd.array(train_data.SalePrice.values)
print('\n before reshape((-1,1)),train_labels:',train_labels)
train_labels = train_labels.reshape((-1,1)) # 整成一列
print('\nIn the End,train_labels:',train_labels)


# 3 # 走你！ ===========================================================
loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

def log_rmse(net,features,labels):
    # <1的数设置为1，取对数时候的值就会更稳定!
    clipped_preds = nd.clip(net(features),1,float('inf'))
    rmse = nd.sqrt(2**loss(clipped_preds.log(),labels.log()).mean())
    return rmse.asscalar()

def train(net,train_features,train_labels,
          test_features,test_labels,
          num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)

    # 然后使用Adam算法!
    trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':learning_rate,'wd':weight_decay})

    # 开始epoch训练
    for epoch in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

# 4 # K fold 交叉验证！！ =============================================================
def get_k_fold_data(k,i,X,y):
    assert k > 1
    fold_size = X.shape([0]) // k   # 整除k！！
    x_train,y_train = None,None
    for j in range(k):
        idx = slice(j * fold_size,(j+1) * fold_size)
        X_part,y_part = X[idx,:],y[idx]