#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 线性回归 v2
 @IsAvailable: true
 @Time       : 2020/12/12 14:33
 @Author     : 剑怜情
'''

from mxnet import nd,init,autograd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet.gluon import Trainer

# 1.生成数据集
input_num = 2
input_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = nd.random.normal(scale=1,shape=(input_examples,input_num))
labels = features[:,0] * true_w[0] +features[:,1] * true_w[1] + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)


# 2. 组装、读取数据集
# 设置batch_size > 打包监督数据 > 整一个可循环obj

batch_size = 10
dataset = gdata.ArrayDataset(features,labels)
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

for x,y in data_iter:
    print(x,y)

# 定义模型：生成容器 > 层 > w > 损失函数 > 优化器
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

# 开始训练 > 定义epochs > 循环batch > 计算loss + 优化 > 输出误差
num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter: # 每次都循环100个轮次。
        with autograd.record():
            l = loss(net(X),y)
        print("l数据！！：",l)
        l.background()
        trainer.step(batch_size)


num_epochs=3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        with autograd.record():
            l=loss(net(X),y)
        l.background()
