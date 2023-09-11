'''
Author: 汪卓 wangzhuo@zuoyebang.com
Date: 2023-06-26 14:48:50
LastEditors: 汪卓 wangzhuo@zuoyebang.com
LastEditTime: 2023-06-27 14:49:43
FilePath: /MachineLearning/MLPClassifier_bp/main.py
Description: 通过sklearn库的MLPClassifier来实现bp网络
'''
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage import io, color, transform

import math
import sys 
import numpy as np
import matplotlib.pyplot as plt

#绘制样本图片
def draw_image(flattened_image, label):
    ax = plt.subplot()
    ax.imshow(flattened_image.reshape(8, 8), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')

    plt.show()

#绘制多个样本图片
def draw_images(images, labels):
    fig, axes = plt.subplots(8, 10, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(8, 8), cmap='gray')
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')

    plt.show()


def get_image(i):
    # 读取自定义图片
    image = io.imread(f'/Users/wangzhuo/Source/MachineLearning/MLPClassifier_bp/{i}.png')
    # 将彩色图片转换为灰度图像
    gray_image = color.rgb2gray(image)
    # 调整图像大小为8x8像素
    resized_image = transform.resize(gray_image, (8, 8), mode='reflect')
    # 将图像像素值转换为与load_digits数据集相同的范围（0到16）
    processed_image = (resized_image * 16).astype(float)
    # 将图像展平为一个一维向量
    flattened_image_ = processed_image.flatten()
    flattened_image = [] 
    for each in flattened_image_:
        flattened_image.append(math.trunc(each))
    return flattened_image


# 加载数字数据集
data = load_digits()
# X为长度1797的list，每个元素的长度是64
X = data.data
y = data.target

#draw_images(X, y)
#sys.exit()

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 创建并训练神经网络模型
model = MLPClassifier(hidden_layer_sizes=(200,), max_iter=1000)
model.fit(X_train, y_train)

# 评估模型性能
#accuracy = model.score(X_test, y_test)
#print("Accuracy:", accuracy)

num = sys.argv[1]

myself_pred = model.predict([get_image(num)])
draw_image(np.array(get_image(num)), f'num is {num} - predict is {myself_pred[0]}')