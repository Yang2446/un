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
keep_prob=tf.placeholder(tf.float32)
# 创建一个简单的神经网络
# 使用截断的正态分布，标准差为0.1
W1=tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1=tf.Variable(tf.zeros([2000])+0.1)
# 使用激活函数是双曲正切激活函数
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)


W2=tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2=tf.Variable(tf.zeros([2000])+0.1)
# 使用激活函数是双曲正切激活函数
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)
W3=tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3=tf.Variable(tf.zeros([1000])+0.1)
# 使用激活函数是双曲正切激活函数
L3=tf.nn.tanh(tf.matmul(L2_drop,W3))+b3
L3_drop=tf.nn.dropout(L3,keep_prob)

W4=tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)
 # 二次代价函数
# loss=tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 定义一个梯度下降法来进行训练的优化器 ,0.2的学习率

train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
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
    for epoch in range(31):
       for batch in range(n_batch):
           batch_xs,batch_ys=mnist.train.next_batch(batch_size)
           sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

       test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
       train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})

       print("Iter"+str(epoch)+",Testing Accuracy  "+str(test_acc)+",Training Accuracy  " + str(train_acc))


# Iter0,Testing Accuracy  0.9473,Testing Accuracy  0.95927274
# Iter1,Testing Accuracy  0.9575,Testing Accuracy  0.9748545
# Iter2,Testing Accuracy  0.963,Testing Accuracy  0.98236364
# Iter3,Testing Accuracy  0.9661,Testing Accuracy  0.9865091
# Iter4,Testing Accuracy  0.9671,Testing Accuracy  0.9887273
# Iter5,Testing Accuracy  0.968,Testing Accuracy  0.9900727
# Iter6,Testing Accuracy  0.9694,Testing Accuracy  0.9909091
# Iter7,Testing Accuracy  0.969,Testing Accuracy  0.99176365
# Iter8,Testing Accuracy  0.97,Testing Accuracy  0.9924
# Iter9,Testing Accuracy  0.9702,Testing Accuracy  0.9928
# Iter10,Testing Accuracy  0.9712,Testing Accuracy  0.99314547
# Iter11,Testing Accuracy  0.9713,Testing Accuracy  0.9933091
# Iter12,Testing Accuracy  0.9711,Testing Accuracy  0.9935273
# Iter13,Testing Accuracy  0.9711,Testing Accuracy  0.9936909
# Iter14,Testing Accuracy  0.9715,Testing Accuracy  0.9938909
# Iter15,Testing Accuracy  0.9714,Testing Accuracy  0.99405456
# Iter16,Testing Accuracy  0.972,Testing Accuracy  0.9942182
# Iter17,Testing Accuracy  0.972,Testing Accuracy  0.9944182
# Iter18,Testing Accuracy  0.9719,Testing Accuracy  0.9945273
# Iter19,Testing Accuracy  0.9718,Testing Accuracy  0.99463636
# Iter20,Testing Accuracy  0.9718,Testing Accuracy  0.9947636
# Iter21,Testing Accuracy  0.9717,Testing Accuracy  0.9948909
# Iter22,Testing Accuracy  0.9727,Testing Accuracy  0.99496365
# Iter23,Testing Accuracy  0.9725,Testing Accuracy  0.9951091
# Iter24,Testing Accuracy  0.9723,Testing Accuracy  0.9951818
# Iter25,Testing Accuracy  0.9724,Testing Accuracy  0.99523634
# Iter26,Testing Accuracy  0.9725,Testing Accuracy  0.99538183
# Iter27,Testing Accuracy  0.9727,Testing Accuracy  0.9954
# Iter28,Testing Accuracy  0.972,Testing Accuracy  0.99545455
# Iter29,Testing Accuracy  0.9726,Testing Accuracy  0.9955091
# Iter30,Testing Accuracy  0.9725,Testing Accuracy  0.99554545
# dropout
# Iter0,Testing Accuracy  0.9203,Training Accuracy  0.91194546
# Iter1,Testing Accuracy  0.9321,Training Accuracy  0.9274182
# Iter2,Testing Accuracy  0.9379,Training Accuracy  0.93703634
# Iter3,Testing Accuracy  0.9415,Training Accuracy  0.94014543
# Iter4,Testing Accuracy  0.9447,Training Accuracy  0.94583637
# Iter5,Testing Accuracy  0.9496,Training Accuracy  0.9489818
# Iter6,Testing Accuracy  0.949,Training Accuracy  0.95245457
# Iter7,Testing Accuracy  0.9517,Training Accuracy  0.95469093
# Iter8,Testing Accuracy  0.9545,Training Accuracy  0.95745456
# Iter9,Testing Accuracy  0.9553,Training Accuracy  0.9592364
# Iter10,Testing Accuracy  0.957,Training Accuracy  0.96107274
# Iter11,Testing Accuracy  0.9583,Training Accuracy  0.96236366
# Iter12,Testing Accuracy  0.9606,Training Accuracy  0.9638364
# Iter13,Testing Accuracy  0.9601,Training Accuracy  0.96525455
# Iter14,Testing Accuracy  0.961,Training Accuracy  0.9659455
# Iter15,Testing Accuracy  0.9623,Training Accuracy  0.9668364
# Iter16,Testing Accuracy  0.9623,Training Accuracy  0.9676545
# Iter17,Testing Accuracy  0.963,Training Accuracy  0.9694545
# Iter18,Testing Accuracy  0.9648,Training Accuracy  0.97025454
# Iter19,Testing Accuracy  0.9656,Training Accuracy  0.9707636
# Iter20,Testing Accuracy  0.9678,Training Accuracy  0.97136366
# Iter21,Testing Accuracy  0.9665,Training Accuracy  0.9725636
# Iter22,Testing Accuracy  0.9662,Training Accuracy  0.9729091
# Iter23,Testing Accuracy  0.9675,Training Accuracy  0.9737091
# Iter24,Testing Accuracy  0.9673,Training Accuracy  0.9734727
# Iter25,Testing Accuracy  0.9672,Training Accuracy  0.9751273
# Iter26,Testing Accuracy  0.9686,Training Accuracy  0.9757636
# Iter27,Testing Accuracy  0.9693,Training Accuracy  0.97581816
# Iter28,Testing Accuracy  0.9696,Training Accuracy  0.97687274
# Iter29,Testing Accuracy  0.9702,Training Accuracy  0.9771636
# Iter30,Testing Accuracy  0.971,Training Accuracy  0.9775636