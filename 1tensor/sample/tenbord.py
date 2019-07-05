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

# 设置命名空间
with tf.name_scope('I'):
    # 定义两个placeholder,行跟批次有关系，y标签
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')

# 创建一个简单的神经网络
with tf.name_scope('Layer'):
    with tf.name_scope('t'):
        W=tf.Variable(tf.zeros([784,10]))
    with tf.name_scope('B'):
        b=tf.Variable(tf.zeros([10]))
    with tf.name_scope('P'):
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
    writer=tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(1):
       for batch in range(n_batch):
           batch_xs,batch_ys=mnist.train.next_batch(batch_size)
           sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

       acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
       print("Iter"+str(epoch)+",Testing Accuracy  "+str(acc))
# 二次代价函数