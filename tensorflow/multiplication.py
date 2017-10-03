import tensorflow as tf
import numpy as np

A=tf.placeholder(dtype="float",shape=(1,1),name="A")
D=tf.shape(A)
B=tf.placeholder(dtype="float",shape=(1,1),name="B")
C=tf.matmul(A,B,name="multiplication")
with tf.Session() as sess:
	a=np.array([2],ndmin=2)
	b=np.array([4],ndmin=2)
	print a.shape
	a[0][0]=3
	b[0][0]=4
	var=sess.run(C,feed_dict={A:a,B:b})	
	print var[0]
	print sess.run(D,feed_dict={A:a})
