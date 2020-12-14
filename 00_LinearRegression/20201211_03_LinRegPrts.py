#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 线性回归 v1
 @IsAvailable: true
 @Time       : 2020/12/11 11:40
 @Author     : 剑怜情
'''
from mxnet import autograd,nd,init,gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn      # ←←←这个模块定义了大量nn的层
from mxnet.gluon import loss as gloss


# 1 # 生成数据集====================================================
num_input = 2
num_examples = 1000
true_w = [2,-3.4]
true_b =4.2
feature = nd.random.normal(scale=1,shape=(num_examples,num_input))
# 注意，labels是一个标量！！
labels = feature[:,0]*true_w[0] + feature[:,1]*true_w[1] + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)


# 2 # 组装、读取数据集===============================================
# 设置batch_size > 打包监督数据 > 整一个可循环obj

batch_size = 10 # 每批次大小为10
# 训练数据的特征 & 标签，打包成一组
dataset = gdata.ArrayDataset(feature,labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

# 来尝试打印上面的随机读取：
for x,y in data_iter: # 会打印100个轮次
    print(x,y)


# 3 # 定义模型======================================================
# 生成容器 > 层 > w > 损失函数 > 优化器

net = nn.Sequential() # Sequential实例是，装很多层的容器。注意线性回归是单层网络
net.add(nn.Dense(1)) # 全连接层在gluon中是一个Dense实例，在这定义该层输出个数为1

# 初始化模型参数,但是不知道为什么这里为None
net.initialize(init.Normal(sigma=0.01)) # 指定权重参数每一个都在 初始化时 随机采样于均值为0，标准差为0.01的分布

# 定义损失函数
loss = gloss.L2Loss() # 平方损失 / L2范数损失

# 定义优化算法
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

# 4 # 训练模型======================================================
# 定义epochs > 循环batch > 计算loss + 优化 > 输出误差

num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter: # y它在打包的时候就是labels，所以是个标量
        with autograd.record(): # 梯度会记录到context
            l = loss(net(X),y) # l长度:[batch_size(10),1]。

        l.backward() # 等价于执行l.sum.backward()，所有相加成一个元素。因为设定了输出层输出个数为1！
        # 会求该变量有关模型参数的梯度

        trainer.step(batch_size) # 指明批量的大小，从而对批量中样本梯度求平均
    l=loss(net(feature),labels)
    print('epoch %d,loss:%f' %(epoch,l.mean().asnumpy()))












