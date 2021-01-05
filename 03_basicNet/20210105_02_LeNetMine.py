#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 
 @IsAvailable: 
 @Time       : 2021/1/5 9:53
 @Author     : 剑怜情
'''

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time

# 1. 定义网络
net = nn.Sequential()
net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),

        nn.Dense(120),
        nn.Dense(84),
        nn.Dense(10))
net.initialize()

X = nd.random.normal(shape=(1,28,28))
for layer in net:
    X = layer(X)





















