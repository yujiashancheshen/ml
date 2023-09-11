'''
Author: 汪卓 wangzhuo@zuoyebang.com
Date: 2023-06-19 20:00:18
LastEditors: 汪卓 wangzhuo@zuoyebang.com
LastEditTime: 2023-09-07 15:22:37
FilePath: /MachineLearning/tensorflow_demo/image_classification.py
Description: 地址： https://tensorflow.google.cn/tutorials/keras/classification?hl=zh-cn 
'''
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import sys

# 绘制一个图片
def draw_image(image):
  plt.figure()
  plt.imshow(image)
  plt.colorbar()
  plt.grid(False)
  plt.show()

def draw_images(images, labels, class_names):
  fig, axes = plt.subplots(2, 5, figsize=(10, 4))
  for i, ax in enumerate(axes.flat):
      ax.imshow(images[i])
      j = np.argmax(labels[i])
      ax.set_title(f"pre: {class_names[j]}")
      ax.axis('off')
  plt.show()

# 导入图片数据
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 构建模型
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# 准备数据并训练
train_images = train_images / 255.0
test_images = test_images / 255.0
model.fit(train_images, train_labels, epochs=10)

# 评估结果
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#predictions = probability_model.predict(test_images)
predictions = model.predict(test_images)

draw_images(test_images, predictions, class_names)