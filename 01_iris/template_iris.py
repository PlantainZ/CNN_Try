import tensorflow as tf
import numpy as np
arr_list = np.arange(0,100).astype(np.float32) # 用np产生float32类型的0-100
shape = arr_list.shape # 获得它的形状,这里shape是 (100,)，即第一维是100

dataset = tf.data.Dataset.from_tensor_slices(arr_list)
# 从张量切片(?)的角度，转化成tf的dataset

dataset_iterator = dataset.shuffle(shape[0]).batch(10)
# 对dataset里的100个数据，先进行洗牌，再在后续batch中分组
# print(dataset_iterator) # 这样是写不出数据的！必须要指定某个batch，才能print出来

def model(xs):
    # ... 写一些函数
    outputs = tf.multiply(xs,0.1)  # 这里的xs，是一行！10个数字
    return outputs
for it in dataset_iterator:
    logits = model(it)
    print(logits)