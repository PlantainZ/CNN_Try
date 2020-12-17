#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 图像分类数据集 数据处理
 @IsAvailable: true
 @Time       : 2020/12/15 9:44
 @Author     : 剑怜情
 @dataSet    : Fashion-MNIST , 一个10类服饰分类数据集，
                               mxnet.gluon.vision自动从神仙爷爷yann lecun的主页拉取数据 !!!
'''

import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys,time

# 1 # 获取数据集==========================================================================================
mnist_train = gdata.vision.FashionMNIST(train = True)
mnist_test = gdata.vision.FashionMNIST(train = False)
print('len of train: %d,len of test: %d ' %(len(mnist_train),len(mnist_test)))


# 2*# 看数据都长什么样子==================================================================================
# 得到第一个样本的feature & label
feature,label = mnist_train[0]
print("feature.shape:",feature.shape) # 对应宽高均=28的图像，0-255间8位无符号int。用三维NDArray存储，最后一维是通道数(=1,黑白)。
print("feature.dtype:",feature.dtype)

# 看看它长什么样子,X是图像矩阵，y是分类。包把图像打出来了！！
X,y = mnist_train[0:9]
d2l.show_fashion_mnist(X,d2l.get_fashion_mnist_labels(y))


# 3 # 处理矩阵生成iter ========================================================================================
# > 整一个transformer > DataLoader生成iter
batch_size = 64
# uint8 --> float32,并除以255，规范化在[0,1]之间。
transformer = gdata.vision.transforms.ToTensor() # 还会将最后一维的通道合并到第一维方便cnn计算！！(ง •_•)ง cnnyyds!!!

# 数据读取经常是训练的瓶颈,所以对于读取也有它的处理
if sys.platform.startswith('win'): # gluon的dataLoader允许使用多进程来加速数据读取
    num_workers = 0 # 可是windows不支持qaq
else:
    num_workers = 4

# transform_first 将ToTensor的变换应用到每个data样本（feature + label）的第一个元素上
# （即整个图像！！因为上边，ToTensor已经把所有多维数据都压到第一维）
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),batch_size,shuffle=True,num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),batch_size,shuffle=False,num_workers=num_workers)

# 4*# 看看读取一遍数据要多久=============================================================================
start = time.time()
for x,y in train_iter:
    continue
print('%.2f sec' %(time.time() - start))
