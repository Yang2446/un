
import tensorflow as tf
m1=tf.placeholder(tf.float32)
m2=tf.placeholder(tf.float32)
mul=tf.multiply(m1,m2)
with tf.Session() as sess:
    print(sess.run(mul,feed_dict={m1:[2],m2:[3]}))