#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : keras 函数式编程实现鸢尾花分类
 @IsAvailable: true
 @Time       : 2020/12/16 16:04
 @Author     : 剑怜情
'''
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
data = load_iris()
iris_target = data.target
iris_data = np.float32(data.data)
iris_target = np.float32(tf.keras.utils.to_categorical(iris_target,num_classes=3))
print(iris_target)

# 组装！
iris_data = tf.data.Dataset.from_tensor_slices(iris_data).batch(50)
iris_target = tf.data.Dataset.from_tensor_slices(iris_target).batch(50)

# 生成模型
inputs = tf.keras.layers.Input(shape=(4))
x = tf.keras.layers.Dense(32,activation='relu')(inputs) # 输出32位
x = tf.keras.layers.Dense(64,activation='relu')(x) # 输出64
predictions = tf.keras.layers.Dense(3,activation='softmax')(x) # 输出3类
model = tf.keras.Model(inputs = inputs,outputs = predictions)
opt = tf.optimizers.Adam(1e-3)

for range in (0,5):
    pass
    # for  in zip(iris_data,iris_target):

