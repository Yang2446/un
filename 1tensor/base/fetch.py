import tensorflow as tf
m1=tf.constant(3)
m2=tf.constant(2)
m3=tf.constant(5)
add=tf.add(m2,m3)
mul=tf.multiply(m1,add)
with tf.Session() as sess:
    result=sess.run([add,mul])
    print(result)