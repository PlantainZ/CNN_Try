from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf
data = load_iris()
iris_target = data.target
iris_data = np.float32(data.data)

print("iris_target:",iris_target)
print("iris_data:",iris_data)

iris_target = np.float32(tf.keras.utils.to_categorical(iris_target,num_classes=3))
print("new iris_target:",iris_target)