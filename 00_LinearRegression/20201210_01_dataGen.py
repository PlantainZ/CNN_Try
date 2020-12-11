#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script  : 学习生成数据
 @IsAvailable: true
 @Time    : 2020/12/11 11:32
 @Author  : 剑怜情
'''

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd,nd
import random


# 生成数据集，1000r x 2c
input_num = 1000
dimention = 2
weight = [2,-3.4]
b = 4.2
features = nd.random.normal(scale=1,shape=(input_num,dimention)) # 输入的数据。
labels = weight[0] * features[:,0] + weight[1] * features[:,1] + b # 计算
labels += nd.random.normal(scale=0.01,shape=labels.shape) # 然后生成的是【1000*1】维度的数据


def set_figsize(figsize=(5.5,3.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize']=figsize

set_figsize()
plt.scatter(features[:,1].asnumpy(),labels.asnumpy(),5) # 第三个参数s：点的大小
plt.show()
