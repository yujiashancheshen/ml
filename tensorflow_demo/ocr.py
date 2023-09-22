'''
Author: 汪卓 wangzhuo@zuoyebang.com
Date: 2023-09-19 20:17:11
LastEditors: 汪卓 wangzhuo@zuoyebang.com
LastEditTime: 2023-09-22 17:11:38
FilePath: /MachineLearning/tensorflow_demo/ocr.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()
# 读取图像
image = keras_ocr.tools.read("./ocr.jpg")  # 替换成您的图像文件路径

# 进行文本识别
peredictions = pipeline.recognize([image])

print(peredictions)

# 打印识别结果
# for prediction in predictions[0]:
#    print(f"文本: {prediction[0]}，置信度: {prediction[1]:.2f}")