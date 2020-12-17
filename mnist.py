# -*- coding: utf-8 -*-

#导入模块包
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from matplotlib.image import imsave
from tensorflow.keras.optimizers import Adam
tf.disable_v2_behavior()


#001导入数据
##读取MNIST数据集。如果不存在会先下载,双引号中的为下载的文件夹，可以修改
###分为训练集、测试集、验证集,数据已经归一化并设置one-hot编码
mnist = input_data.read_data_sets("C:/Users/nhb\Desktop/MNISTset/Datasets", one_hot=True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels
validation_images = mnist.validation.images
validation_labels = mnist.validation.labels


#002查看数据
##查看数据形状
print(mnist.train.num_examples)
print(mnist.test.num_examples)
print(mnist.validation.num_examples)
#这个训练数据集有55000*784个数据，55000个数字，每个数字由784个像素组成。
print ( '训练集数字的形状 :', mnist.train.images.shape)
print ( '训练集标签的形状 :', mnist.train.labels.shape)
print ('我们有'+str(train_images.shape[0])+'个训练集数据，每个数字由'+str(train_images.shape[1])+'个像素组成，标签有'+str(train_labels.shape[1])+'种类型。')
print ( '测试集数字的形状 :', mnist.test.images.shape)
print ( '测试集标签的形状 :', mnist.test.labels.shape)
print ( '我们有'+ str(test_images.shape[0]) + '个测试集数据，每个数字由'+str(test_images.shape[1])+'个像素组成，标签有'+str(test_labels.shape[1])+'种类型。')
print ( '验证集数字的形状 :', mnist.validation.images.shape)
print ( '验证集标签的形状 :', mnist.validation.labels.shape)
print ( '我们有'+ str(validation_images.shape[0]) + '个验证集数据，每个数字由'+str(validation_images.shape[1])+'个像素组成，标签有'+str(validation_labels.shape[1])+'种类型。')


#查看前25个数据并保存
for i in range(25):
    image_array = train_images[i,:]
    image_array_labels = np.argmax(train_labels[i,:])
    image_array = image_array.reshape(28,28)
    plt.subplot(5,5,i+1)
    plt.imshow(image_array,cmap=plt.get_cmap('gray'))
    plt.title(image_array_labels)
    print(image_array_labels)
    #保存图片，需要先自己创建桌面文件夹C:/Users/nhb\Desktop/MNISTset/pic
    label = str(image_array_labels)
    ind = str(i)
    filename = 'mnist_train_'+ind+'_'+label+'.jpg' 
    print(filename)
    filepath = "C:/Users/nhb\Desktop/MNISTset/pic"
    imsave(filepath + "/" +filename ,image_array,cmap = 'gray')


#003创建模型
#创建一个空的容器，Sequential，往里面添加各个层
model = Sequential()
#将数据形状变为-1,28,28,1，其中-1是指当不知道有多少个数据输入时，可以用-1代替，
#28*28像素的图片，单通道的灰度图
train_images = train_images.reshape(-1, 28, 28, 1)
test_images  = test_images.reshape(-1, 28, 28, 1)
# 第一个卷积层
# input_shape 输入数据
# filters 滤波器个数32，生成32 张特征图
# kernel_size 卷积窗口大小5*5
# strides 步长1
# padding padding方式 same/valid
# activation 激活函数为relu,修正线性单元
model.add(Convolution2D(
                         input_shape=(28, 28, 1),
                         filters=32,
                         kernel_size=5,
                         strides=1,
                         padding='same',
                         activation='relu'
                         ))
# 第一个池化层
# pool_size 池化窗口大小2*2
# strides 步长2
# padding padding方式 same/valid
model.add(MaxPooling2D(
                        pool_size=2,
                        strides=2,
                        padding='same'
                        ))
# 第二个卷积层
# filters 滤波器个数64，生成64 张特征图
# kernel_size 卷积窗口大小5*5
# strides 步长1
# padding padding方式 same/valid
# activation 激活函数 
model.add(Convolution2D(
                         filters = 64,
                         kernel_size=5,
                         strides=1,
                         padding='same',
                         activation='relu'
                         ))
# 第二个池化层
# pool_size 池化窗口大小2*2
# strides 步长2
# padding padding方式 same/valid
model.add(MaxPooling2D(
                       pool_size=2,
                       strides=2,
                       padding='same'
                       ))
# 把第二个池化层的输出进行数据扁平化
# 相当于把(64,7,7,64)数据->(64,7*7*64)
model.add(Flatten())
# 第一个全连接层，激活函数为relu
model.add(Dense(1024, activation='relu'))
# Dropout，0.5，将一半数据Dropout，让一半隐层节点为0，减少过拟合
#过拟合表现：训练数据上损失函数小，预测准确率较高，但是在测试数据上损失函数比较大，预测准确率较低。
model.add(Dropout(0.5))
# 第二个全连接层，激活函数为softmax，分为10类
model.add(Dense(10, activation='softmax'))
# 定义优化器,学习率为0.0001
adam = Adam(lr=1e-4)
# 定义优化器，loss function，训练过程中计算准确率，损失函数为交叉熵crossentropy，衡量指标为accuracy
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, batch_size=64, epochs=10, validation_data=(test_images,test_labels))
# 保存模型，到C:/Users/nhb/Desktop/MNISTset/model/mnist.h5，可以自己创建
model.save('C:/Users/nhb/Desktop/MNISTset/model/mnist.h5')

