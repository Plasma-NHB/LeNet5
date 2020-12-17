# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:30:49 2020

@author: nhb
"""


#现在我们自己写一个来看我们训练出来的模型是否可用
#导入模块包
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from tensorflow.keras.models import load_model
from PIL import Image
tf.disable_v2_behavior()

# 载入数据
mnist = input_data.read_data_sets("C:/Users/nhb\Desktop/MNISTset/Datasets", one_hot=True)
# 载入数据，数据载入的时候就已经划分好训练集和测试集
train_images = mnist.train.images
train_labels = mnist.train.labels
image_array = train_images[14,:]
image_array = image_array.reshape(28,28)
# 获取一张照片，并把它的shape 变成二维（784->28×28）,用灰度图显示
plt.subplot(2,2,1)
plt.imshow(image_array, cmap='gray')
# 不显示坐标
plt.axis('off')
plt.show()
# 载入我自己写的数字图片
plt.subplot(2,2,2)
img = Image.open('C:/Users/nhb/Desktop/MNISTset/File/W33G.jpg')
# 显示图片
plt.imshow(img, cmap='gray')
# 不显示坐标
plt.axis('off')
plt.show()

# 把图片大小变成28×28，并且把它从3D 的彩色图变为1D 的灰度图
#模式L为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，
#其他数字表示不同的灰度。在PIL中，从模式“RGB”转换为“L”模式是按照下面的公式转换的：
#L = R * 0.299 + G * 0.587+ B * 0.114
image = np.array(img.resize((28, 28)).convert('L'))
# 显示图片,用灰度图显示
plt.subplot(2,2,3)
plt.imshow(image, cmap='gray')
# 不显示坐标
plt.axis('off')
plt.show()

# 我自己写的数字是白底黑字，MNIST数据集的图片是黑底白字
# 所以我们需要先把图片从白底黑字变成黑底白字，就是255-image
# MNIST数据集的数值都是0-1 之间的，所以我们还需要/255.0 对数值进行归一化
image = (255 - image) / 255.0
plt.subplot(2,2,4)
# 显示图片，用灰度图显示
plt.imshow(image, cmap='gray')
# 不显示坐标
plt.axis('off')
plt.show()

# 把数据处理变成4 维数据
image = image.reshape((1, 28, 28, 1))
# 载入训练好的模型
model = load_model('C:/Users/nhb/Desktop/MNISTset/model/mnist.h5')
# predict_classes对数据进行预测并得到它的类别
prediction = model.predict_classes(image)
print("这个数字是："+str(prediction))