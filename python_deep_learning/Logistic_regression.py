from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
#from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#tr_x, tr_y, te_X, te_Y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
	s=sigmoid(x)*(1-sigmoid(x));
	return s
def softmax(x):
	s=np.exp(x)
	p=np.sum(k,axis=0,keepdims=True)
	p=s/p
	return p
def initialise_parameter(n_x,n_y):
	 W = np.random.randn(1,n_x)*0.001
	 b=np.zeros((1,1));
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
	predictions = (z>0.5)

	return predictions




def normalizeRows(x):
	x_norm = np.linalg.norm(x,axis=1,ord=2,keepdims=True)
	x=x/x_norm;
	return x




if __name__=="__main__":
	#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	print "hi";
	np.random.seed(0)
	X, Y =datasets.make_moons(10000, noise=0.20)
	Y=Y.reshape(Y.shape[0],1)
	print(X.shape)
	print Y.shape
	
	#x = np.array([1, 2, 3])
	x = np.array([[0, 3, 4],[1, 6, 4]])
	print normalizeRows(x)	
	
	#plt.scatter(X[:,0], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)
	#plt.show()
	# here forward_back back_propagation
	X=X.T
	Y=Y.T	
	X=normalizeRows(X)
	parameters=initialise_parameter(2,1)
	W=parameters["W"];
	b=parameters["b"]
	lr=01.0;
	num_epoch=100
	m=X.shape[1];
	print "shape:",m;
	for i in range(0,1000000):
		z=np.dot(W,X)+b
		#print "size of Z:",z.size
		a=sigmoid(z);
	
		dz=a-Y
		dw=np.dot(dz,X.T)/m		
		db=np.sum(dz)/m
		cost=compute_cost(z,Y)
		W=W-(lr*dw)
		b=b-(lr*db)
		A=predict(z)
		if(lr%10000==0):
			lr/=10;
		B=predict(Y)
		B=1*B
		A=1*A
		if(i%100==0):
			#print "new iteration:"	
			#print "size of A",A.shape;
			#print "size of Y",Y.shape;			
			#print "accuracy:",np.sum(A==B)
			total_count=float(np.sum(A==B))
			#print Y
			#print "here is predict:\n",A
			#print "here is W",W
			print "i:",i," cost :",cost," ","accuracy:",(total_count*100)/m;	
			#O=input(":")


	#plot_decision_boundary(lambda x: predict(z), X, Y)
	#plt.title("Decision Boundary for hidden layer size " + str(4))
	#f, axarr = plt.subplots(1, 2)	
	#axarr[1, 1].scatter(X[0,:], X[1,:], s=40, c=Y, cmap=plt.cm.Spectral)
	
	#plt.draw()
	#plt.figure(1)
	#axarr[1,2].scatter(X[0,:],X[1,:],s=50,c=B,cmap=plt.cm.Spectral)
	
	#plt.show()

	
