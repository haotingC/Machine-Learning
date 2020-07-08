import matplotlib.pyplot as plt
import numpy as np 
import math
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def setdata():
    x = []
    y = []
    f = open('input.data')
    line = f.readline()
    cnt = 0
    while line :
        line=line.strip('\n')
        row_data = line.strip().split(' ')
        x.append(float(row_data[0]))  
        y.append(float(row_data[1])) 
        line = f.readline()
    x = np.array(x)
    y = np.array(y)
    f.close()
    return x,y

def Kernel(Xn,Xm,beta,alpha,l):
    # rational quardratic kernel
    Krq = beta*((1+cdist(Xn,Xm,'sqeuclidean')/(2.0*alpha*l*l))**(-alpha))
    return Krq

def GP(x,y,testX,alpha,l):
    x= x.reshape(-1,1)
    y= y.reshape(-1,1)
    testX = testX.reshape(-1,1)
    K0 = Kernel(x,x,beta,alpha,l)
    K1 = Kernel(testX,testX,beta,alpha,l)
    K2 = Kernel(x,testX,beta,alpha,l)
    C = K0+np.eye(len(x))/beta
    Kstar = K1+1/beta
    mean = np.dot(np.dot(np.transpose(K2),np.linalg.inv(C)),y)
    var2 = Kstar-np.dot(np.dot(np.transpose(K2),np.linalg.inv(C)),K2)
    return mean.flatten(),var2

def ConfidenceInterval(mean,var2):
    plusSD = mean + 1.96*np.sqrt(np.diag(var2))
    minusSD = mean - 1.96*np.sqrt(np.diag(var2))
    return plusSD,minusSD

def neg_log_likihood(theta,x):
    #minimizing negative marginal log-likelihood
    x = x.reshape(-1,1)
    C = Kernel(x,x,beta,alpha=theta[0],l=theta[1])+np.eye(len(x))/beta
    neg_lnP = 0.5*math.log(np.linalg.norm(C))+0.5*np.dot(np.dot(np.transpose(y),np.linalg.inv(C)),y)+(x.shape[0]/2)*math.log(2*math.pi)
    return neg_lnP

def visualization(mode,x,y,testX,mean,plusSD,minusSD):
    plt.subplot(2, 1, mode + 1)
    plt.plot(x,y,'.',color='red')
    plt.plot(testX,mean,color='blue')
    plt.fill_between(testX,plusSD,minusSD,facecolor='green',alpha=0.3)
    plt.xlim(-60,60)

if __name__ == '__main__':
    x,y = setdata()
    testX = np.linspace(-60,60, 500) # test point cnt = 500
    beta = 5.0
    alpha,l = 10.0, 1.0
    
    mean,var2 = GP(x,y,testX,alpha,l)
    plusSD,minusSD = ConfidenceInterval(mean,var2)
    visualization(0,x,y,testX,mean,plusSD,minusSD)
    
    #optimal kernel
    theta = [1.0, 1.0]
    result = minimize(neg_log_likihood,theta,args=(x))
    alpha,l=result.x
    mean,var2 = GP(x,y,testX,alpha,l)
    plusSD,minusSD = ConfidenceInterval(mean,var2)
    visualization(1,x,y,testX,mean,plusSD,minusSD)

    plt.show()
    