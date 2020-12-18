#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 
 @IsAvailable: 
 @Time       : 2020/12/18 11:42
 @Author     : 剑怜情
'''
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
import d2lzh as d2l
from mxnet import nd,autograd,gluon
from mxnet import initializer as init

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
print("mnist_train.shape:",len(mnist_train))
print("mnist_test.shape:",len(mnist_test))

# 处理成iter
batch_size=64
train_iter,test_iter = d2l.load_data_fashion_mnist(64)

# 定义W,b
num_input = 784
num_outputs = 10

# W = nd.random.normal(scale=1,shape=(num_input,num_outputs))
# b = nd.zeros(shape=num_outputs)

# W.attach_grad()
# b.attach_grad()

# 定义模型、分类模型、损失函数
def softmax(X):
    x_exp = X.exp()
    partition = x_exp.sum(axis=1,keepdims = True)
    return x_exp/partition

net = nn.Sequential()
net.add(nn.Dense(10)) # 输出是10个分类
net.initialize(init.Normal(sigma=0.01))

def cross_entropy(y_hat,y):
    return -nd.pick(y_hat,y).log()

def accuracy(y_hat,y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


num_epoch=5
trianer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})

# 这里把这个函数展开来写
# d2l.train_ch3(net,train_iter,test_iter,cross_entropy,num_epoch,batch_size,[W,b],lr)













