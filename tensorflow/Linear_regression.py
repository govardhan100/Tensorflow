import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
train_x= np.linspace(-1, 1, 101)
m=train_x.shape[0]
train_x=train_x.reshape(m,1)

train_y= 2 * train_x + np.random.randn(*train_x.shape) * 0.33 # create a y value which is approximately linear but with some random noise
train_y.reshape(m,1)
X=tf.placeholder(dtype="float",shape=[m,1],name="input")
Y=tf.placeholder(dtype="float",shape=[m,1],name="out")
w=tf.Variable([random.random()],name="weight")

b=tf.Variable(0.0,name="bias")
y=tf.add(tf.multiply(w,X),b)
C=(Y-y)*(Y-y)
cost=tf.reduce_sum(C)/m
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(10000):
		Cost,_=sess.run([cost,train],feed_dict={X:train_x,Y:train_y})
		print "i:",i,"cost:",Cost
		#k=input()
	
	W=sess.run(w)
	b=sess.run(b)
	print W,b



plt.figure("data_show")
plt.plot(train_x,train_y,'ro')
plt.plot(train_x,train_x*W[0])
plt.show()

