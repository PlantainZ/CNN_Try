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

# 1. 读取数据
csv_path = "D:\plantainz\OuO_DeepLearning\DataSet\housePrices"
train_data = pd.read_csv(csv_path + '\\train.csv')
test_data = pd.read_csv(csv_path + '\\test.csv')
all_features = pd.concat((train_data.iloc[:,1:],test_data.iloc[:,1:]))


# 2. 预处理
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x-x.mean())/x.std())
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features,dummy_na=True)

n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)

train_labels = nd.array(train_data.iloc[:,-1])
train_labels = train_labels.reshape((-1,1))


# 3. 走你~
loss = gloss.L2Loss()
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

def log_rmse(net,features,labels):
    clipped_preds = nd.clip(net(features),1,float('inf'))
    return nd.sqrt(2*loss(clipped_preds.log(),labels.log()).mean()).asscalar()

def train(net,train_features,train_labels,
          test_features,test_labels,
          num_epochs,batch_size,learning_rate,weight_decay):
    train_ls,test_ls= [],[]
    train_iter = gdata.DataLoader(gdata.ArrayDataset((train_features,train_labels)))
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':learning_rate,
                                                        'wd':weight_decay})

    for num_epochs in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(train_features)
                l = loss(y_hat,y)
            l.backward()
            trainer.step(batch_size)

        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

# 4. k-fold验证
def get_k_fold_data(k,i,X,y):
    assert k > 1
    k_fold = X.shape[0]//k
    x_train , y_train = None,None

    for j in range(k):
        idx = slice(j*k_fold,(j+1)*k_fold)
        x_part,y_part = X[idx,:],y[idx]

        if j==i:
            x_valid ,y_valid = x_part,y_part
        elif x_train is None:
            x_train,y_train =x_part,y_part
        else:
            x_train = nd.concat(x_train,x_part,dim = 0)
            y_train = nd.concat(y_train,y_part,dim = 0)
    return x_train,y_train,x_valid,y_valid

def k_fold(k,x_train,y_train,num_epochs,batch_size,learning_rate,weight_decay):
    train_l_sum ,valid_l_sum=0,0

    for i in range(k):
        data = get_k_fold_data(k,i,x_train,y_train)
        net = get_net()
        train_ls,valid_ls = train(net,*data,num_epochs,batch_size,learning_rate,weight_decay)
        train_l_sum +=train_ls[-1]
        valid_l_sum+=train_ls[-1]

        if(i==0):
            d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse',
                         range(1,num_epochs+1),valid_ls,)
            d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse',
                         range(1,num_epochs+1),valid_ls,
                         ['train','valid'])
        return train_l_sum/k,valid_l_sum/k

# 5. 对照
k,num_epochs,batch_size,learning_rate,weight_decay = 5,100,64,5,0
train_l,valid_l = k_fold(k,train_features,train_labels,num_epochs,batch_size,learning_rate,weight_decay)

def train_and_pred(train_features,test_features,train_labels,test_data,
                   num_epochs,batch_size,learning_rate,weight_decay):
    net = get_net()
    train_ls,_ = train(net,train_features,train_labels,None,None,
                       num_epochs,batch_size,learning_rate,weight_decay)

    d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse')

    preds = net(test_features).asnumpy()























