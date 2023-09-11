'''
Author: 汪卓 wangzhuo@zuoyebang.com
Date: 2023-09-04 20:13:20
LastEditors: 汪卓 wangzhuo@zuoyebang.com
LastEditTime: 2023-09-04 20:19:14
FilePath: /MachineLearning/gaussian_bayes/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
Description: 《白话大数据和机器学习》 10.1 朴素贝叶斯 
'''
#coding=utf-8
import numpy as np
from sklearn.naive_bayes import GaussianNB

#基因片段A  基因片段B 高血压胆结石
#1： 是    0：否
data_table = [
    [1, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 1],
    [0, 0, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
]
#基因片段
X = [[1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]]
#高血压
y1 = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
#训练
clf = GaussianNB().fit(X, y1)
#预测
p = [[1, 0]]
print(clf.predict(p))


#胆结石
y2 = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
#训练
clf = GaussianNB().fit(X, y2)
#预测
p = [[1, 0]]
print(clf.predict(p))