#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 线性回归 v1
 @IsAvailable: true
 @Time       : 2020/12/11 11:40
 @Author     : 剑怜情
'''
from mxnet import autograd,nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn      # ←←←这个模块定义了大量nn的层

# 1 # 生成数据集====================================================
num_input = 2
num_examples = 1000
true_w = [2,-3.4]
true_b =4.2
feature = nd.random.normal(scale=1,shape=(num_examples,num_input))
labels = feature[:,0]*true_w[0] + feature[:,1]*true_w[1] + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)


# 2 # 组装、读取数据集===============================================
batch_size = 10
# 训练数据的特征 & 标签，打包成一组
dataset = gdata.ArrayDataset(feature,labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

# 来尝试打印上面的随机读取：
i = 0
for x,y in data_iter:
    i+=1
    print(x,y)
    print("轮次：",i) # 会是100！！


# 3 # 定义模型======================================================
net = nn.Sequential() # Sequential实例是，装很多层的容器。















