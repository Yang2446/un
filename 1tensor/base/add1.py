# -*- coding: utf-8 -*-
# 变量 Variable,add,赋值assign,
import tensorflow as tf
state=tf.Variable(0,name='counter')
new_value=tf.add(state,1)
#
update=tf.assign(state,new_value)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
