'''
Author: 汪卓 wangzhuo@zuoyebang.com
Date: 2023-06-15 19:10:46
LastEditors: 汪卓 wangzhuo@zuoyebang.com
LastEditTime: 2023-06-16 19:42:26
FilePath: /k_means/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#从磁盘读取城市经纬度数据
X = []
f = open('/Users/wangzhuo/Source/MachineLearning/k_means/city.txt')
for v in f:
    X.append([float(v.split('，')[1]), float(v.split('，')[2])])
#转换成numpy array
X = np.array(X)
#类簇的数量
n_clusters = 2 
#现在把数据和对应的分类数放入聚类函数中进行聚类
cls = KMeans(n_clusters).fit(X)
#X中每项所属分类的一个列表
cls.labels_

#画图
#markers = ['^', 'x', 'o', '*', '+', '>', '<', '_', 'H', 'd']
markers = ['o', '*']
for i in range(n_clusters):
    members = cls.labels_ == i
    plt.scatter(X[members, 0], X[members, 1], s=30, marker=markers[i], c='b', alpha=0.5)
plt.title('')
plt.show()