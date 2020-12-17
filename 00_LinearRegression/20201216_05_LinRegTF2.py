#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 线性回归 // mxnet --> tensorflow 2.x
 @IsAvailable: false
 @Time       : 2020/12/16 9:32
 @Author     : 剑怜情
 @Question   : line 44,Shapes (50,) and (50, 1) are incompatible
'''

import numpy as np
import tensorflow as tf
# 数据生成
input_num = 2
input_examples = 1000
true_w = [3.4,1.5]
true_b = 5.2
features = np.random.normal(scale=1,size=(input_examples,input_num))
labels = features[:,0]*true_w[0]+features[:,1]*true_w[1]+true_b
labels += np.random.normal(scale=0.01,size=np.shape(labels))

# 数据转格式
batch_size=50
# 以1000这个维度为单位，随机打乱
lin_data = tf.data.Dataset.from_tensor_slices((features,labels)).shuffle(buffer_size=input_examples).batch(batch_size)
data_iter = iter(lin_data)

# 生成模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1,kernel_initializer=tf.initializers.RandomNormal(stddev=0.01)))
# model.add(tf.keras.layers.Dense(1,activation='softmax'))

loss = tf.losses.MeanSquaredError() # 均方误差~
opt = tf.optimizers.SGD(learning_rate=0.03)
# opt = tf.optimizers.Adam(1e-3)

# 开始轮滚
loss_history = []
num_epochs = 3
for epoch in range(3):
    for X,y in data_iter:
        with tf.GradientTape() as tape:
            loss_value = tf.losses.categorical_crossentropy(y_true=y,y_pred=model(X))
            # loss_value = loss(model(X),y)
            print(y.shape)
        grads = tape.gradient(loss_value,model.trainable_variables)
        opt.apply_gradients(zip(grads,model.trainable_variables))

    l = loss(model(features),labels)
    print("epoch: %d , training loss is: %f" %(epoch,l))

