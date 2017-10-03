import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/1072025/Desktop/python/MNIST_data/", one_hot=True)
train_X,train_Y,test_X,test_Y=mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

class logistic:
	def __init__(self,x,y,lr):
		self.x=x
		self.y=y
		self.lr=lr
		self.y_estimated=self.model(self.x)
		self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_estimated,labels=self.y))
		self.train=tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost)
		correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_estimated,1))
		self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		self.initial=tf.global_variables_initializer()
	def model(self,x):
		self.w=tf.Variable(initial_value=np.random.rand(28*28,10),name="w")
		self.b=tf.Variable(initial_value=np.zeros((1,10)),name="b")
		return tf.matmul(x,self.w)+self.b			
if __name__=="__main__":
	lr=2
	x=tf.placeholder(dtype=tf.float64,shape=(None,28*28),name="x")
	y=tf.placeholder(dtype=tf.float64,shape=(None,10),name="y")
	Logistic_object=logistic(x,y,lr)
	no_epoch=100
	with tf.Session() as sess:
		sess.run(Logistic_object.initial)
		var={Logistic_object.x:train_X,Logistic_object.y:train_Y}
		for i in range(no_epoch):
			_,cost,acc=sess.run([Logistic_object.train,Logistic_object.cost,Logistic_object.accuracy],feed_dict=var)
			print "I:",i,"cost:",cost,"  Accuracy:",acc
		#print "cost:",cost
		var={Logistic_object.x:test_X,Logistic_object.y:test_Y}
		Acc=sess.run([Logistic_object.accuracy],feed_dict=var)
		print "Test accuracy:",Acc









				
						
