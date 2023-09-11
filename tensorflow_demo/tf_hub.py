'''
Author: 汪卓 wangzhuo@zuoyebang.com
Date: 2023-09-07 18:02:25
LastEditors: 汪卓 wangzhuo@zuoyebang.com
LastEditTime: 2023-09-11 15:31:55
FilePath: /MachineLearning/tensorflow_demo/tf_hub.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import sys
import requests

url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([
    hub.KerasLayer(url, input_shape=(224, 224, 3))
])
model.summary()

#image_url = "https://geektutu.com/img/icon.png"
#image_url = "https://photo.16pic.com/00/16/34/16pic_1634458_b.jpg"
image_url = "https://img1.baidu.com/it/u=2586083276,1607470616&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=753"
response = requests.get(image_url)
if response.status_code == 200:
    # 指定本地文件名
    local_filename = "./image.jpg"
    # 打开文件以保存下载的图片
    with open(local_filename, 'wb') as file:
        file.write(response.content)
    print(f"图片已成功下载到 {local_filename}")
else:
    print("无法下载图片")
    sys.exit()

image = Image.open("./image.jpg").resize((224,224))

result = model.predict(np.array(image).reshape(1, 224, 224, 3)/255.0)
ans = np.argmax(result[0], axis=-1)

labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', labels_url)
imagenet_labels = np.array(open(labels_path).read().splitlines())
print(imagenet_labels[ans])
