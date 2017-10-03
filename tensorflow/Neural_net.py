import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import random
mnist = input_data.read_data_sets("/home/1072025/Desktop/python/MNIST_data/", one_hot=True)
train_X,train_Y,test_X,test_Y=mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
class Neural_net:
	def __init__(self,x,y,lr):
		self.x=x
		self.y=y
		self.lr=lr
		self.y_=self.model(self.x)
		self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_,labels=self.y))
		#self.train=tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost)
		self.train=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
		self.accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y,1),tf.argmax(self.y_,1)),tf.float32))
		self.initial=tf.global_variables_initializer()
		


	def model(self,x):
		self.W1=tf.Variable(initial_value=tf.random_normal([784,600], stddev=0.01),name="W1")
		self.b1=tf.Variable(initial_value=tf.random_normal([1,600], stddev=0.01),name="b1")
		self.W2=tf.Variable(initial_value=tf.random_normal([600,10], stddev=0.01),name="W2")
		self.b2=tf.Variable(initial_value=tf.random_normal([1,10], stddev=0.01),name="b2")
		z1=tf.matmul(x,self.W1)+self.b1
		a1=tf.nn.relu(z1)
		#a1=tf.nn.dropout(a1,0.5)
		a2=tf.matmul(a1,self.W2)+self.b2
		return a2
if __name__=="__main__":
	no_epoch=50
	Batch_size=10000
	lr=.01
	x=tf.placeholder(dtype=tf.float32,shape=(None,28*28),name="x")
	y=tf.placeholder(dtype=tf.float32,shape=(None,10),name="y")
	NN_obj=Neural_net(x,y,lr)
	with tf.Session() as sess:
		sess.run(NN_obj.initial)
		for i in range(no_epoch):
			for start, end in zip(range(0, len(train_X), Batch_size), range(Batch_size, len(train_X)+1, Batch_size)):
				var={NN_obj.x:train_X[start:end,:],NN_obj.y:train_Y[start:end,:]}						
				_,acc,cost=sess.run([NN_obj.train,NN_obj.accuracy,NN_obj.cost],feed_dict=var)
				print "I:",i,"cost:",cost,"Accuracy:",acc
		var={NN_obj.x:test_X,NN_obj.y:test_Y}
		acc=sess.run(NN_obj.accuracy,feed_dict=var)
		print "Test Accuracy:",acc




