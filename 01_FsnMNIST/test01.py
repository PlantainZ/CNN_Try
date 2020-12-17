#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 
 @IsAvailable: 
 @Time       : 2020/12/17 10:53
 @Author     : 剑怜情
'''

from mxnet.gluon import data as gdata
import d2lzh as d2l
from mxnet import nd

# 1 # 看看数据长什么样============================================
data_train = gdata.vision.FashionMNIST(train=True)
data_test = gdata.vision.FashionMNIST(train = False)

X,y = data_train[0]
# 自己看形状！！
print(X.shape)
print(y.shape)

# 2 # 行列sum ==================================================
x = nd.array([1,2,3],[4,5,6])
print(x.sum(axis = 0,keepdims=True)) # 只留下一行！
print(y.sum(axis = 1,keepdims = True)) # 只留下一列！
