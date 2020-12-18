#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 用FsnMNIST 实现softmax回归的从零实现
 @IsAvailable: true
 @Time       : 2020/12/17-18 11:19
 @Author     : 剑怜情
'''

import d2lzh as d2l
from mxnet import autograd,nd
from mxnet.gluon import data as gdata
import sys,time


# 1 # 数据生成( utils 1 )==========================================================================================
mnist_train = gdata.vision.FashionMNIST(train = True)
mnist_test = gdata.vision.FashionMNIST(train = False)
print('len of train: %d,len of test: %d ' %(len(mnist_train),len(mnist_test))) # 6W,1W


# 2 # 处理矩阵生成iter ========================================================================================
# > 整一个transformer > (设置线程数) > DataLoader生成iter
batch_size = 64
# 看这！有变化。因为这个是从零实现
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)


# 3 # 定义Wb==========================================================================================
num_inputs = 784
num_outputs = 10

# 领会下这个W矩阵的形状，为什么输出10个类，这里纵列也要10个
W = nd.random.normal(scale=0.01,shape=(num_inputs,num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()


# 4 # 定义模型 (utils 2 & 3) =====================================================================
def softmax(X):
    X_exp = X.exp() # 对每个元素进行底为e的指数计算，每一行都变成了非负数！
    partition = X_exp.sum(axis=1,keepdims=True) # sum到最后只留下一列！！。
    return X_exp/partition # 这里有广播机制.最后每行的和为1

def net(X):
    # reshape() 中的第一个参数，-1表示让X变成1个/几个任意维度。
    # 第二个参数是col。如果不指明就默认为unknow，生成一行unknow列
    # 当然第二个参数如果指定了，那就生成对应的列数。
    return softmax(nd.dot(X.reshape((-1,num_inputs)),W)+b) # 这里生成(1,784)形状后与W点乘，生成的形状是(1,10).
    # nd.dot是专门进行矩阵乘法的！！

def cross_entropy(y_hat,y):
    return -nd.pick(y_hat,y).log()

# *损失函数中的pick原理
y_hat = nd.array([[0.1,0.3,0.6],[0.3,0.2,0.5]]) # 两个样本预测值
y=nd.array([0,2],dtype='int32')
# 在y_hat中取第一行的第0+1个，第二行的第2+1个！！
nd.pick(y_hat,y) # 为了得到两个样本的 标签的 预测概率，可以用Pick!!


# 5 # 计算分类准确率 ( utils 4 ) =======================================================================
def accuracy(y_hat,y):
    # y_hat.argmax 返回每行中最大元素的索引(即第几列)，返回结果与变量y形状相同。
    # 因为标签类型是int，要把y 变成浮点数 再进行相等判断。
    return (y_hat.argmax(axis = 1)== y.astype('float32')).mean().asscalar()

# *accuracy函数的小测试
print('accuracy test:',accuracy(y_hat,y))
print('accuracy,d2l_test:',d2l.evaluate_accuracy(data_iter=test_iter,net = net))


# 6 # 训练模型 =========================================================
num_epoch ,lr = 5,0.1
d2l.train_ch3(net,train_iter,test_iter,
              cross_entropy,
              num_epoch,batch_size,[W,b],lr)


# 7 # 分类预测 ========================================================
for X,y in test_iter:
    break
true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true,pred in zip(true_labels,pred_labels)]

d2l.show_fashion_mnist(X[0:9],titles[0:9])








