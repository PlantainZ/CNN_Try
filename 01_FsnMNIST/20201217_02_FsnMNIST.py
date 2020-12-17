#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 用FsnMNIST 实现softmax回归的从零实现
 @IsAvailable: true
 @Time       : 2020/12/17 11:19
 @Author     : 剑怜情
'''

import d2lzh as d2l
from mxnet import autograd,nd
from mxnet.gluon import data as gdata
import sys,time


# 1 # 数据生成==========================================================================================
mnist_train = gdata.vision.FashionMNIST(train = True)
mnist_test = gdata.vision.FashionMNIST(train = False)
print('len of train: %d,len of test: %d ' %(len(mnist_train),len(mnist_test)))

# 2 # 处理矩阵生成iter ========================================================================================
# > 整一个transformer > DataLoader生成iter
batch_size = 64
# uint8 --> float32,并除以255，规范化在[0,1]之间。
transformer = gdata.vision.transforms.ToTensor() # 还会将最后一维的通道合并到第一维方便cnn计算！！(ง •_•)ง cnnyyds!!!

# 数据读取经常是训练的瓶颈,所以对于读取也有它的处理
if sys.platform.startswith('win'): # gluon的dataLoader允许使用多进程来加速数据读取
    num_workers = 0 # 可是windows不支持qaq
else:
    num_workers = 4

# 看这！有变化。因为这个是从零实现
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)


# 3 # ==========================================================================================
num_inputs = 784
num_outputs = 10

# 领会下这个W矩阵的形状，为什么输出10个类，这里纵列也要10个
W = nd.random.normal(scale=0.01,shape=(num_inputs,num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()

# 4 # 实现softmax运算=====================================================================
def softmax(X):
    X_exp = X.exp() # 对每个元素进行底为e的指数计算，每一行都变成了非负数！
    partition = X_exp.sum(axis=1,keepdims=True) # sum到最后只留下一列！！。
    return X_exp/partition # 这里有广播机制.最后每行的和为1

X = nd.random.normal(shape=(2,5))
X_prob = softmax(X)
print("x_prob:",X_prob)
print("x_prob.row sum:",X_prob.sum(axis = 1,keepdims = True))

# 5 # 定义模型==============================================================================
def net(X):
    # reshape() 中的第一个参数，-1表示让X变成1个/几个任意维度。
    # 第二个参数是col。如果不指明就默认为unknow，生成一行unknow列
    # 当然第二个参数如果指定了，那就生成对应的列数。
    return softmax(nd.dot(X.reshape((-1,num_inputs)),W)+b) # 这里生成(1,784)形状后与W点乘，生成的形状是(1,10)

def cross_entropy(y_hat,y):
    return -nd.pick(y_hat,y).log()

# 6 # 定义损失函数 =========================================================================
y_hat = nd.array([[0.1,0.3,0.6],[0.3,0.2,0.5]]) # 两个样本预测值
y=nd.array([0,2],dtype='int32')
nd.pick(y_hat,y) #













