import tensorflow as tf
import numpy as np

A=tf.placeholder(dtype="float",shape=(1,2),name="A")
B=tf.placeholder(dtype="float",shape=(1,2),name="B")
C=tf.matmul(A,B,transpose_a=False,transpose_b=True)
D=tf.shape(A)
E=tf.multiply(A,B)
a=np.array([1,2],ndmin=2)
b=np.array([1,3],ndmin=2)
with tf.Session() as sess:
	c,d,e=sess.run([C,D,E],feed_dict={A:a,B:b})
	print c,"shape:",d,e	

