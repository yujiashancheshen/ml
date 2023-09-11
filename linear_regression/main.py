'''
Author: 汪卓 wangzhuo@zuoyebang.com
Date: 2023-06-15 18:11:36
LastEditors: 汪卓 wangzhuo@zuoyebang.com
LastEditTime: 2023-06-15 18:58:56
Description: 《白话大数据和机器学习》 8.2 拟合 
'''
import numpy as np
import matplotlib.pyplot as plt
#原始数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [0.199, 0.389, 0.580, 0.783, 0.980, 1.177, 1.380, 1.575, 1.771]
A = np.vstack([x, np.ones(len(x))]).T
# A：
#[[ 1.  1.]
# [ 2.  1.]
# [ 3.  1.]
# [ 4.  1.]
# [ 5.  1.]
# [ 6.  1.]
# [ 7.  1.]
# [ 8.  1.]
# [ 9.  1.]]
#调用最小二乘法函数
a, b = np.linalg.lstsq(A, y)[0]
print(a)
print(b)

#转换成numpy array
x = np.array(x)
y = np.array(y)
#画图
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, a * x + b, 'r', label='Fitted line')
plt.show()