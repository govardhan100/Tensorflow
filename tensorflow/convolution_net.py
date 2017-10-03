import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/1072025/Desktop/python/MNIST_data/", one_hot=True)
class con_net:
	def __init__(self,x,y,lr):
		self.x=x
		self.y=y
		self.lr=lr
		self.y_=self.model(self.x)
		self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_,labels=self.y))
		self.train=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
		self.accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y,1),tf.argmax(self.y_,1)),tf.float32))
		self.initialise=tf.global_variables_initializer()

	def Layer(self,input_x,kernel_shape,bias_shape):
		weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1),"weight")
		bias=tf.Variable(initial_value=tf.random_normal(bias_shape, stddev=0.01),name="bais")
		conv=tf.nn.conv2d(input_x,weights,strides=[1,1,1,1],padding="SAME")
		out=tf.nn.relu(conv+bias)
		return out

	def net(self,x):	
		weights = tf.Variable(tf.truncated_normal([7*7*64,1024], stddev=0.1))
		bias=tf.Variable(tf.truncated_normal([1024], stddev=0.1))
		f1=tf.matmul(x,weights)+bias
		f1=tf.nn.dropout(f1,0.5)
		weights2 = tf.Variable(tf.truncated_normal([1024,10], stddev=0.1))
		bias2=tf.Variable(tf.truncated_normal([10],stddev=0.1))
		f2=tf.matmul(f1,weights2)+bias2
		return f2

	def model(self,x):
		x_image = tf.reshape(x, [-1, 28, 28, 1])
		with tf.variable_scope("con1") as scope:
			kernel_shape=[5, 5, 1, 32]
			bias_shape=[32]
			con1=self.Layer(x_image,kernel_shape,bias_shape)
			out1=tf.nn.max_pool(con1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
		with tf.variable_scope("con2") as scope:
			kernel_shape=[7, 7, 32, 64]
			bias_shape=[64]
			con2=self.Layer(out1,kernel_shape,bias_shape)
			out2=tf.nn.max_pool(con2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
		out2=tf.reshape(out2,[-1,7*7*64])
		with tf.variable_scope("Dense_layer") as scope:
			Net_out=self.net(out2)
		return Net_out

if __name__=="__main__":
	lr=0.0001
	batch_size=200
	step_size=20000
	x=tf.placeholder(dtype=tf.float32,shape=[None,28*28])
	y=tf.placeholder(dtype=tf.float32,shape=[None,10])
	Net=con_net(x,y,lr)
	with tf.Session() as sess:
		sess.run(Net.initialise)
		for i in range(step_size):
			batch=mnist.train.next_batch(batch_size)
			v={Net.x:batch[0],Net.y:batch[1]}
			_,accuracy,cost=sess.run([Net.train,Net.accuracy,Net.cost],feed_dict=v)
			#k={Net.x:mnist.test.images,Net.y:mnist.test.labels}
			#test_accuracy=sess.run(Net.accuracy,feed_dict=k)
			print "i:",i," Cost:",cost," accuracy:",accuracy#," Test Acc:",test_accuracy










