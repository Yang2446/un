# -*- coding: utf-8 -*-
# 简单的手写数据集
# 优化网络：改批次的大小，添加隐藏层，神经元，改权值和初始值，代价函数，使用交叉熵，其他的优化函数，训练更多次数，训练从21变成200
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# # 载入数据集
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
# 每个批次的大小
batch_size=100
# 计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
# 定义两个placeholder,行跟批次有关系，y标签
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
# 创建一个简单的神经网络
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(x,W)+b)
 # 二次代价函数
# loss=tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 定义一个梯度下降法来进行训练的优化器 ,0.2的学习率
# train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)

# 定义一个最小化代价函数,定义一个训练，

init=tf.global_variables_initializer()
# 比较两个参数是否是一样的，一样返回ture，不一样返回false，变量里面是true，false
# argmax()求标签y最大值在哪个位置，求预测的标签在那个位置，求最大的在第几个位置
# 结果存放在一个布尔型列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
       for batch in range(n_batch):
           batch_xs,batch_ys=mnist.train.next_batch(batch_size)
           sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

       acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
       print("Iter"+str(epoch)+",Testing Accuracy  "+str(acc))
# 二次代价函数
# Iter2,Testing Accuracy  0.8817
# Iter3,Testing Accuracy  0.8882
# Iter4,Testing Accuracy  0.8942
# Iter5,Testing Accuracy  0.8968
# Iter6,Testing Accuracy  0.9
# Iter7,Testing Accuracy  0.9016
# Iter8,Testing Accuracy  0.9037
# Iter9,Testing Accuracy  0.9052
# Iter10,Testing Accuracy  0.9064
# Iter11,Testing Accuracy  0.9071
# Iter12,Testing Accuracy  0.9077
# Iter13,Testing Accuracy  0.9093
# Iter14,Testing Accuracy  0.9103
# Iter15,Testing Accuracy  0.911
# Iter16,Testing Accuracy  0.9115
# Iter17,Testing Accuracy  0.9126
# Iter18,Testing Accuracy  0.9126
# Iter19,Testing Accuracy  0.9132
# Iter20,Testing Accuracy  0.9141
# 交叉熵代价函数
# ter1,Testing Accuracy  0.8937
# Iter2,Testing Accuracy  0.9013
# Iter3,Testing Accuracy  0.9059
# Iter4,Testing Accuracy  0.9082
# Iter5,Testing Accuracy  0.9102
# Iter6,Testing Accuracy  0.9121
# Iter7,Testing Accuracy  0.9131
# Iter8,Testing Accuracy  0.915
# Iter9,Testing Accuracy  0.9164
# Iter10,Testing Accuracy  0.9183
# Iter11,Testing Accuracy  0.9182
# Iter12,Testing Accuracy  0.9177
# Iter13,Testing Accuracy  0.9197
# Iter14,Testing Accuracy  0.9191
# Iter15,Testing Accuracy  0.9205
# Iter16,Testing Accuracy  0.9198
# Iter17,Testing Accuracy  0.9215
# Iter18,Testing Accuracy  0.9216
# Iter19,Testing Accuracy  0.9211
# Iter20,Testing Accuracy  0.9221
