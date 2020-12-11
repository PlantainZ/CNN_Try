#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script  : 线性回归 v0
 @IsAvailable: false 仅供原理学习
 @Time    : 2020/12/11 11:34
 @Author  : 剑怜情
'''

from mxnet import nd
from d2lzh import data_iter,autograd

# 生成数据
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
labels = features[:,0] * true_w[0] + features[:,1]*true_w[1] +true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)
# print(labels)

# 初始化模型参数：一组w，一个b
w = nd.random.normal(scale=0.01,shape=(num_inputs,1))
b = nd.zeros(shape=(1,))
# 创建参数的梯度
w.attach_grad()
b.attach_grad()

# 定义模型
def linreg(X,w,b):
    return nd.dot(X,w) +b # 注意是点乘！！

# 定义损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2 / 2

# 定义优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param = param - param.grad * lr /batch_size

# 训练模型
lr = 0.03
num_epoch = 3
loss = squared_loss
batch_size = 10

for epoch in range(num_epoch): # 多少轮
    # 监督学习！
    for X,y in data_iter(batch_size,features,labels):
        with autograd.record():
            l = loss(net(features,w,b),y)
        l.backward()
        sgd([w,b],lr,batch_size)

# 训练模型
lr = 0.03
num_epochs = 3
loss = squared_loss
batch_size = 10
for epoch in range(num_epochs):
    # X和y分别是小批量样本的特征和标签。data_iter会专门返回这个东西。
    for X,y in data_iter(batch_size,features,labels): # features:(10,2) , w:(2,1) ↓↓↓
        with autograd.record():
            l = loss(net(features,w,b),y) # 有关小批量X & y的损失，形状是(10,1)

        # 我实在没搞懂↓↓↓这是干嘛的。
        l.backward() # 小批量的损失，对模型参数求梯度~ 注意这里。因为l不是标量，所以会(Σ l中元素)得到新的变量再计算。

        sgd([w,b],lr,batch_size) # 迭代模型参数

    train_l = loss(net(features,w,b),labels)
    print('epoch %d,loss %f' %(epoch+1,train_l.mean().asnumpy()))



