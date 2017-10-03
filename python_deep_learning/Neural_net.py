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
	 W1 = np.random.randn(600,n_x)*0.01
	 W2= np.random.randn(10,600)*0.01
	 b1=np.zeros((600,1));
	 b2=np.zeros((10,1))
	 parameters = {"W1": W1,"b1": b1,"W2":W2,"b2":b2}
	 return parameters

def normalizeRows(x):
	x_norm = np.linalg.norm(x,axis=1,ord=2,keepdims=True)
	x = x/x_norm;
	return x;

def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = (-1 / m) * (np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))))
	cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ())
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
	l=(M*100.0)/K
	return l
	
def leaky_relu(x):
	return np.maximum(x,0.01*x)
def Derivative_leaky_relu(x):
	gradient=1.0*(x>=0)
	other=1.0*(x<0)
	other=other*0.01
	gradient+=other
	return gradient
	

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
	lr=01.0;
	batch_size=10000;
	step=int(X.shape[1]/batch_size);
	epoch=1000
	parameters=initialise_parameter(n_x,n_y)
	W1=parameters["W1"]
	b1=parameters["b1"]
	W2=parameters["W2"]
	b2=parameters["b2"]
	m=batch_size
	test_X=teX.T;
	test_Y=teY.T;
	for j in range(epoch):
		if j+1%200==0:
			lr=lr/5
		for i in range(step-1):
			x=X[:,batch_size*i:batch_size*(i+1)]
		
		
			y=Y[:,batch_size*i:batch_size*(i+1)]	
			#print "x:",x.shape,"y:",y.shape
			#forward progagation
			#first layer
			Z1=np.dot(W1,x)+b1;
			A1=leaky_relu(Z1)
			Z2=np.dot(W2,A1)+b2
			A2=softmax(Z2)
			#backward propagation
			dz2=A2-y
			dw2=(np.dot(dz2,A1.T))/m

			db2=np.sum(dz2,axis=1,keepdims=True)/m
			#print "derivative shape:",Derivative_leaky_relu(Z1).shape
			#print "2 shape",np.dot(W2.T,dz2).shape
			dz1=np.dot(W2.T,dz2)*Derivative_leaky_relu(Z1)
			dw1=np.dot(dz1,x.T)/m
			db1=np.sum(dz1,axis=1,keepdims=True)/m
			W1=W1-(lr*dw1)
			W2=W2-(lr*dw2)
			b2=b2-(lr*db2)
			b1=b1-(lr*db1)
			#print b2
			#print 'DW2:',np.sum(np.abs(dw2))
			#print "Dw1",np.sum(np.abs(dw1))
			#print "dz2: error:",np.sum(np.abs(dz2))
			#print dz2
			#l=input()
			Acc=accuracy(A2,y)
			pre=predict(A2)
			cost=compute_cost(A2,y)
			
			Z1=np.dot(W1,test_X)+b1;
			A1=leaky_relu(Z1)
			Z2=np.dot(W2,A1)+b2
			A2=softmax(Z2)
			
			print "I:",i+(step*j)," accuracy:",Acc,"cost:",cost,"testing accuracy:",accuracy(A2,test_Y)
		
		




