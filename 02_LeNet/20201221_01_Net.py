#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : LeNet的网络结构
 @IsAvailable: true
 @Time       : 2020/12/21 14:18
 @Author     : 剑怜情
'''

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time

# 注意！！用的全是sigmoid
net = nn.Sequential()
net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),

        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        nn.Dense(10))

X = nd.random.normal(shape=(1,1,28,28))
net.initialize()
for layer in net:
    X=layer(X)
    print(layer.name,'output shape:\t',X.shape) # 可以这样看output信息！！！


# 训练模型
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)



# 2 # 判断有没有gpu可以用！！=====================================================
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()
print(ctx)


# 3 # 开始操作 ==================================================================
def evaluate_accuracy(data_iter,net,ctx):
    acc_sum,n = nd.array([0],ctx=ctx),0
    for X,y in data_iter:
        X,y = X.as_in_context(ctx),y.as_in_context(y).astype('float32')
        acc_sum = (net(X).argmax(axis=1) == y).sum()
        n+=y.size
    return acc_sum.asscalar() / n


def evaluate_accuracy(data_iter,net,ctx):
    acc_sum,n = nd.array([0],ctx=ctx),0
    for X,y in data_iter:
        # 如果ctx代表GPU及相应显存，将数据复制到显存上
        X,y = X.as_in_context(ctx),y.as_in_context(ctx).astype('float32')
        acc_sum = (net(X).argmax(axis=1) == y).sum()
        n+=y.size
    return acc_sum.asscalar()/n