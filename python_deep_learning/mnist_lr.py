from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math

def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
	s=sigmoid(x)*(1-sigmoid(x));
	return s
def softmax(x):
	s=np.exp(x)
	p=np.sum(s,axis=0,keepdims=True)
	p=s/p
	return p

def initialise_parameter(n_x,n_y):
	 W = np.random.randn(n_y,n_x)*0.001
	 b=np.zeros((n_y,1));
	 parameters = {"W": W,"b": b}
	 return parameters

def normalizeRows(x):
	x_norm = np.linalg.norm(x,axis=1,ord=2,keepdims=True)
	x = x/x_norm;
	return x;



def compute_cost(A2, Y):
	m =200 #Y.shape[1] # number of example
	A2=np.maximum(A2,1E-15)
	var=np.multiply(np.log(1-(0.8*A2)),1-Y)
	
	logprobs = np.multiply(np.log(A2),Y)
	cost = - np.sum(logprobs/m)  
	cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
		                       # E.g., turns [[17]] into 17 
	assert(isinstance(cost, float))

	return cost
def predict(z):
	predictions = np.argmax(z,axis=0)

	return predictions




def normalizeRows(x):
	x_norm = np.linalg.norm(x,axis=1,ord=2,keepdims=True)
	x=x/x_norm;
	return x

def accuracy(Y_hat,Y):
	Y_pre1=predict(Y_hat)
	Y_pre2=predict(Y)
	g=Y_pre1==Y_pre2
	M=sum(g);
	K=Y.shape[1];
	l=(M*100)/K
	return l
	


if __name__=="__main__":
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	# data 	
	X=trX.T;
	Y=trY.T
	n_x=X.shape[0]
	n_y=10
	training_size=X.shape[1]
	#parameter
	lr=0.00001;
	batch_size=50000;
	step=X.shape[1]/batch_size;
	epoch=100
	parameters=initialise_parameter(n_x,n_y)
	print parameters["W"].shape
	print parameters["b"].shape
	W=parameters["W"]
	b=parameters["b"]
	for i in range(1000):
		x=X[:,0:batch_size]
		
		
		y=Y[:,0:batch_size]	
		#print "x:",x.shape,"y:",y.shape
		Z=np.dot(W,x)+b;
		#print "Z:",Z.shape
		Y_=softmax(Z);	
		#print "Y_ ",Y_.shape
		dz=Y_-y
		dw=np.dot(dz,x.T)
		db=np.sum(dz,axis=1,keepdims=True)
		W=W-(lr*dw)
		b=b-(lr*db)
		#cost=	compute_cost(Y_,y)
		Acc=accuracy(Z,y)
		print "i:",i," accuracy:",Acc		
	test_X=teX.T
	test_Y=teY.T
	Z=np.dot(W,test_X)+b;
	Acc=accuracy(Z,test_Y)
	print "test accuracy:",Acc
		
	

		
		
		




