#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 一些问题集结
 @IsAvailable: true
 @Time       : 2020/12/17 10:53
 @Author     : 剑怜情、席纷
'''

from mxnet.gluon import data as gdata
import d2lzh as d2l
from mxnet import nd,autograd
import sys

# 1 # 看看数据长什么样============================================
data_train = gdata.vision.FashionMNIST(train=True)
data_test = gdata.vision.FashionMNIST(train = False)

X,y = data_train[0]
# 自己看形状！！
print('X.shape =',X.shape)
print('y.shape =',y.shape)


# 2 # 行列sum ===================================================
x = nd.array([[1,2,3],[4,5,6]])
print("x.sum,axis = 0 :",x.sum(axis = 0,keepdims=True)) # 只留下一行！
print("x.sum,axis = 0 :",x.sum(axis = 1,keepdims = True)) # 只留下一列！


# 3 # 测试pick ==================================================
y_hat = nd.array([[0.1,0.3,0.6],[0.3,0.2,0.5]]) # 两个样本预测值
y=nd.array([0,2],dtype='int32')


#在y_hat中取第一行的第0+1个，第二行的第2+1个！！
print("nd.pick测试：",nd.pick(y_hat,y)) # 为了得到两个样本的 标签的 预测概率，可以用Pick!!
print('y_hat.argmax,axis=1 :',y_hat.argmax(axis = 1))


# 4 # 对y.astype('float32').mean().asscalar()的理解==============
print('y.astype(\'float32\'):',y.astype('float32'))
print("y.astype('float32').mean:",y.astype('float32').mean())
print("y.astype('float32').mean.asscalar():",y.astype('float32').mean().asscalar())


# 5 # 一个包内置函数的理解。用于训练模型的函数！========================================
def train_ch3(train_iter,test_iter,
              net,loss,
              num_epochs,batch_size,
              params=None,lr=None,trainer=None):

    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)
                # 传入的y_hat和y都是一行数列！
                l = loss(y_hat,y).sum() # 这里因为要对每个样本都进行损失计算！翻一翻损失函数的定义！！
            l.backward() # 薅到梯度

            # 每个iter都更新梯度，看看用的什么优化器
            if trainer is None:
                d2l.sgd(params,lr,batch_size)
            else:
                trainer.step(batch_size)

            # 将标签化为float数
            y = y.astype('float32')
            train_l_sum += l.asscalar() # 训练总损失也化为float数
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size # 也就是 train_iter一趟的数

        test_acc = evaluate_accuracy(test_iter,net)
        print("epoch %d,loss %.4f,train acc %.3f,test acc %.3f" %(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))


# 6 # 一些其它的操作 ===============================================================
# uint8 --> float32,并除以255，规范化在[0,1]之间。
transformer = gdata.vision.transforms.ToTensor() # 还会将最后一维的通道合并到第一维方便cnn计算！！(ง •_•)ง cnnyyds!!!

# 数据读取经常是训练的瓶颈,所以对于读取也有它的处理
if sys.platform.startswith('win'): # gluon的dataLoader允许使用多进程来加速数据读取
    num_workers = 0 # 可是windows不支持qaq
else:
    num_workers = 4










