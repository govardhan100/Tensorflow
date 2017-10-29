import numpy as np 
import random
#two coin 
bais_A=0.6
bais_B=0.4
actual_A_head=0.6
actual_B_head=0.5
Number_data_point=100000
size_vector=10

def data_point(size_vector,bais):
	temp=0
	for i in range(size_vector):
		if random.random()<bais:
			temp+=1
	return temp

data_total=[]
total_A=0;
total_B=0
for i in range(Number_data_point):
	#vector_binary=[]
	if(random.random()<bais_A):
		data_total.append(data_point(size_vector,actual_A_head))
		total_A+=1
	else:
		data_total.append(data_point(size_vector,actual_B_head))
		total_B+=1




print "total_A",total_A
print "total_B",total_B
#input()

data_H=np.array(data_total)#np.array([5,9,8,4,7])
data_T=size_vector-data_H
data=np.array([data_H,data_T],dtype=np.float32).T
#print(data_H)
#print(data_T)
#print "Head Tail\n",data


Theta=np.array([0.5,0.6])
def Likelyhood_probability(data,Theta):
	p1=np.power(Theta,data)
	p2=np.power(1-Theta,size_vector-data)
	temp=np.multiply(p1,p2)
	return (temp/np.sum(temp)).reshape((1,2))[0]

#print "probability of coin:",Likelyhood_probability(9,Theta)



epsilon=np.exp(-20)


count_Head=np.zeros([2,2])
while True:
	for i in range(data.shape[0]):
		count_Head[0,]+=Likelyhood_probability(data[i],Theta)[0]*data[i,]
		count_Head[1,]+=Likelyhood_probability(data[i],Theta)[1]*data[i,]
		#print(Likelyhood_probability(data[i],Theta)[0]*data[i,])
		#count_Head[2,:]=Likelyhood_probability(data[i],Theta)[1]*data[i]

		#print(Likelyhood_probability(data[i],Theta)[0]*data[i])
	update=(count_Head.T)/np.sum(count_Head,1)
	if(np.max(np.abs(update[0,]-Theta))<epsilon):
		break;


	Theta=update[0,]
	print(Theta)

