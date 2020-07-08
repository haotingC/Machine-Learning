import numpy as np
import random
import math
import matplotlib.pyplot as plt

# def visualization():
#     # Ground truth function (from linear model generator)
#     # Final predict result
#     # At the time that have seen 10 data points
#     # At the time that have seen 50 data points

def univar_gaussian_data_gen(m,s):
    #  Marsaglia polar method 
    U = random.uniform(-1,1)
    V = random.uniform(-1,1)
    S = U**2+V**2
    while S>=1 :
        U = random.uniform(-1,1)
        V = random.uniform(-1,1)
        S = U**2+V**2
    temp =((-2/S)*math.log(S))**(1/2)
    z1 = U*temp   # standard normal random variable1
    z2 = V*temp   # standard normal random variable2

    data_point_x =  m + (s**(1/2))*z1    # X = μ + σZ

    return data_point_x
    
def Sequential_Estimator(m,s):
    # Welford's online algorithm 

    #n=0
    M = m
    S = s
    n = 0
    oldM = 0
    oldS = 0

    #n=1
    x = univar_gaussian_data_gen(m,s)
    n = 1 
    oldM = M
    M = M + (x-M)/n

    #n=2
    x = univar_gaussian_data_gen(m,s)
    n = 2 
    oldM = M
    M = M + (x-M)/n
    
    while (abs(oldM-M)>0.001) or (abs(oldS-S)>0.001):
        x = univar_gaussian_data_gen(m,s)
        n+=1
        oldM = M
        M = M + (x-M)/n
        oldS = S
        S = (S*(n-2) + (x-M)*(x-oldM))/(n-1)
        print("Add data point: ",x)
        print("Mean = ", M, " Variance = ", S)
    
def poly_basis_linear_data_gen(n,a,w):
    y = 0
    x = random.uniform(-1,1)
    for i in range(n):
        y += w[i]*(x**i)
    
    # e ~ N(0,a)
    e = univar_gaussian_data_gen(0,a)
    return x, y+e


def Baysian_Linear_Regression(b,n,a,w):
    Alist = []
    Ylist = []
    oldMean = np.zeros((n,1))
    oldVariance = np.zeros((n,n))
    I = np.identity(n)
    mean = np.zeros((n,1))
    variance = np.zeros((n,n))
    B = b*I
    cnt=0
    
    while (np.linalg.norm(oldMean-mean)>0.001 or np.linalg.norm(oldVariance-variance)>0.001 or cnt<10):
        cnt+=1
        x,y = poly_basis_linear_data_gen(n,a,w)
        print("n=",cnt)
        print("Add data point (", x, ", ", y, "):")
        print()

        oldMean = mean
        oldVariance = variance
        

        X = np.array([x**i for i in range(n)])
        Alist.append(X)
        A = np.array(Alist)
        Ylist.append([y])
        Y = np.array(Ylist)

        # variance = inverse(aAtA+bI)
        temp = a*np.dot(np.transpose(A),A)+B
        variance = np.linalg.inv(temp)
        # mean = [inverse(AtA+(b/a)I)]AtY = [inverse((1/a)*(aAtA+bI))]AtY = a*inverse(aAtA+bI)*At*Y
        # mean = a*np.dot(np.dot(variance, np.transpose(A)), Y)
        mean = np.dot(variance,a*np.dot(np.transpose(A), Y)+np.dot(B,oldMean))

        # new prior = old posterior
        # predict mean = (oldMean)t*X
        p_mean = np.dot(np.transpose(oldMean),X)
        # predict variance = (1/a)+Xt*(bI)*X
        p_variance = 1/a+np.dot(np.dot(np.transpose(X), B), X)
        
        
        B = temp # = aAtA+bI
        
        
        print("Posterior mean:")
        print(mean)
        print()
        print("Posterior variance:")
        print(variance)
        print()
        print("Predictive distribution ~ N(", p_mean, ", ", p_variance, ")")
        print(np.linalg.norm(oldMean-mean), np.linalg.norm(oldVariance-variance))
        print("--------------------------------------------------")
        if cnt == 10:
            mean10 = mean
            variance10 = variance
        elif cnt == 50:
            mean50 = mean
            variance50 = variance
   
    meanP = mean
    varianceP = variance 

    # visualization()

    # Ground truth
    plt.subplot(2,2,1)
    plt.title("Ground truth")
    x_d = np.array([])
    y_d = np.array([])
    for x in np.arange(-2,2,0.1):
        x_d = np.append(x_d,x) 
        X = np.array([x**i for i in range(n)])
        W = np.array(w)
        y = np.dot(W.transpose(),X)
        y_d = np.append(y_d,y)

    plt.plot(x_d, y_d, '-', color='black')
    plt.plot(x_d, y_d+a, '-', color='red')
    plt.plot(x_d, y_d-a, '-', color='red')
    plt.xlim(-2,2)
    plt.ylim(-15,25)



    x_p = np.array([A[i][1] for i in range(cnt)])
    y_p = np.array([Y[i][0] for i in range(cnt)])
    x_10 = np.array([A[i][1] for i in range(10)])
    y_10 = np.array([Y[i][0] for i in range(10)])
    x_50 = np.array([A[i][1] for i in range(50)])
    y_50 = np.array([Y[i][0] for i in range(50)])
    
    x_mean = np.array([])
    xlist = []
    # print(n)
    for x in np.arange(-2,2,0.1):
        x_mean = np.append(x_mean,x) 
        X = np.array([x**i for i in range(n)])
        xlist.append(X)
    x_predict = np.array(xlist)

    # Predict result
    plt.subplot(2,2,2)
    plt.title("Predict result")

    y_predict = []
    var = []
    # print(meanP.shape)
    for k in range(40):
        ## mean
        y_predict.append(np.dot(meanP.transpose(), x_predict[k]))
        ## variance
        var = np.append(var, (1/a)+np.dot(np.dot(np.array(x_predict[k]).transpose(), varianceP), x_predict[k]))
    y_predict = np.array(y_predict)
    var = np.array(var).reshape((40,1))
    plt.plot(x_p, y_p,'o',markersize =3)
    plt.plot(x_mean, y_predict, '-', color='black')
    plt.plot(x_mean, y_predict+var, '-', color='red')
    plt.plot(x_mean, y_predict-var, '-', color='red')
    plt.plot()
    plt.xlim(-2,2)
    plt.ylim(-15,25)
    
    # After 10 incomes
    plt.subplot(2,2,3)
    plt.title("After 10 incomes")

    y_predict = []
    var = []
    for k in range(40):
        ## mean
        y_predict.append(np.dot(mean10.transpose(), x_predict[k]))
        ## variance
        var = np.append(var, (1/a)+np.dot(np.dot(np.array(x_predict[k]).transpose(), variance10), x_predict[k]))
    y_predict = np.array(y_predict)
    var = np.array(var).reshape((40,1))
    plt.plot(x_10, y_10,"o",markersize =3)
    plt.plot(x_mean, y_predict, '-', color='black') ## mean
    plt.plot(x_mean, y_predict+var, '-', color='red') ## variance
    plt.plot(x_mean, y_predict-var, '-', color='red') ## variance
    plt.xlim(-2,2)
    plt.ylim(-15,25)
    
    # After 50 incomes
    plt.subplot(2,2,4)
    plt.title("After 50 incomes")
    y_predict = []
    var = []
    for k in range(40):
        ## mean
        y_predict.append(np.dot(mean50.transpose(), x_predict[k]))
        ## variance
        var = np.append(var, (1/a)+np.dot(np.dot(np.array(x_predict[k]).transpose(), variance50), x_predict[k]))
    y_predict = np.array(y_predict)
    var = np.array(var).reshape((40,1))
    plt.plot(x_50, y_50,"o",markersize =3)
    plt.plot(x_mean, y_predict, '-', color='black') ## mean
    plt.plot(x_mean, y_predict+var, '-', color='red') ## variance
    plt.plot(x_mean, y_predict-var, '-', color='red') ## variance
    plt.xlim(-2,2)
    plt.ylim(-15,25)
    plt.show()


if __name__ == "__main__":
    # sequential estimator
    m = 3
    s = 5
    print("Data point source function: N(", m, ",", s, ")")
    # Sequential_Estimator(m,s)
 
    # Baysian Linear Regression
    # b=1
    # n=4
    # a=1
    # w=[1,2,3,4]

    # b=100
    # n=4
    # a=1
    # w=[1,2,3, 4]

    b=1
    n=3
    a=3
    w=[1,2,3]
    Baysian_Linear_Regression(b,n,a,w)