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

# 生成数据集
input_num = 2
input_examples = 1000
true_w = [2,5.1]
true_b = -3.1
features = nd.random.normal(scale=1,shape=(input_examples,input_num))
labels = features[:,1]*true_w[1] + features[:,0]*true_w[0]+true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

# 打包数据
batch_size = 10
dataset = gdata.ArrayDataset(features,labels)
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

# 生成模型
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

# 开始训练
epochs = 4
for epoch in range(1,epochs+1):
    for x,y in data_iter:
        with autograd.record():
            logits = net(x)
            l = loss(logits,y)
        l.backward() # 薅到梯度
        trainer.step(batch_size) # 更新参数~~~
    l = loss(net(features),labels)
    print('epoch: %d , loss: %f' %(epoch,l.mean().asnumpy()))







