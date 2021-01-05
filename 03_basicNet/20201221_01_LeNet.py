#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : LeNet的网络结构
 @IsAvailable: true
 @Time       : 2020/12/21 14:18
 @Author     : 剑怜情
'''

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time

# 注意！！用的全是sigmoid
net = nn.Sequential()
net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),

        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        nn.Dense(10))
net.initialize()


X = nd.random.normal(shape=(1,1,28,28))
for layer in net:
    X=layer(X)
    print(layer.name,'output shape:\t',X.shape) # 可以这样看output信息！！！


# 训练模型
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)



# 2 # 判断有没有gpu可以用！！=====================================================
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()
print(ctx)


# 3 # 开始操作 ==================================================================

def evaluate_accuracy(data_iter,net,ctx):
    acc_sum,n = nd.array([0],ctx=ctx),0
    for X,y in data_iter:
        # 如果ctx代表GPU及相应显存，将数据复制到显存上
        X,y = X.as_in_context(ctx),y.as_in_context(ctx).astype('float32')
        acc_sum = (net(X).argmax(axis=1) == y).sum() # argmax在mxnet中会返回浮点数
        n+=y.size
    return acc_sum.asscalar()/n # 这里转成标量


def train_LeNet(train_iter,test_iter,net,
              batch_size,num_epochs,trainer,ctx):
    print('training on',ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start= 0.0,0.0,0,time.time()
        for X,y in train_iter:
            X,y = X.as_in_context(ctx),y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            trainer.step(batch_size)

            # 都转化成标量！！
            y = y.astype('float32')
            train_l_sum = l.asscalar()
            train_acc_sum = (y_hat.argmax(axis =1 ) ==y).sum().asscalar()
            n+=y.size
            test_acc = evaluate_accuracy(test_iter,net,ctx)
        print('epoch %d,loss %.3f,train_acc %.4f,test_acc %.4f,time: %.1f sec'
              %(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc,time.time()-start))

# 4 # 走你！===============================================================
lr,epoch = 0.1,5
net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier()) # ?
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_LeNet(train_iter,test_iter,net,batch_size,epoch,trainer,ctx)


















