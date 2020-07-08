import math, random
import numpy as np
import matplotlib.pyplot as plt

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

def Data_point(N,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2):
    x = np.zeros((N*2,3))
    for i in range(N):
        x[i][0] = (univar_gaussian_data_gen(mx1,vx1))
        x[i][1] = (univar_gaussian_data_gen(my1,vy1))
        x[i][2] = 1
    for i in range(N):
        x[N+i][0] = (univar_gaussian_data_gen(mx2,vx2))
        x[N+i][1] = (univar_gaussian_data_gen(my2,vy2))
        x[N+i][2] = 1
    return x

def printGraph(no,D,L):
    plt.subplot(1, 3, no + 1)
    
    x = D[:,0].reshape(N*2,1)
    y = D[:,1].reshape(N*2,1)
    cl_x_1 = x[L<=0.5]
    cl_y_1 = y[L<=0.5]
    cl_x_2 = x[L>0.5]
    cl_y_2 = y[L>0.5]
    plt.plot(cl_x_1, cl_y_1, '.', color='red')
    plt.plot(cl_x_2, cl_y_2, '.', color='blue')
    if no==0:
        plt.title("Ground truth")
    elif no==1:
        plt.title("Gradient descent")
    elif no==2:
        plt.title("Newton's method")

def printResult(no,Data,Label,w):
    # confusion matrix
    truth = Label
    predict = np.dot(Data,w)
    N = predict.shape[0]
    conf_mtrx = np.zeros(4) #confusion matrix
    for i in range(Label.shape[0]):
        if(predict[i] <= 0.5 and truth[i]<=0.5):
            conf_mtrx[0] +=1
        elif(predict[i] > 0.5 and truth[i]<=0.5):
            conf_mtrx[1] +=1
        elif(predict[i] <= 0.5 and truth[i]>0.5):
            conf_mtrx[2] +=1
        elif(predict[i] > 0.5 and truth[i]>0.5):
            conf_mtrx[3] +=1

    if no==1:
        print("Gradient descent:")
    elif no==2:
        print("Newton's method:")
    print("")
    print("w:")
    print(w)
    print("")
    print("Confusion Matrix:")
    print("              Predict cluster 1 Predict cluster 2")
    print("Is cluster 1             ", conf_mtrx[0],"      ", conf_mtrx[1])
    print("Is cluster 2             ", conf_mtrx[2],"      ", conf_mtrx[3])
    print("")
    print("Sensitivity (Successfully predict cluster 1): ", conf_mtrx[0]/(conf_mtrx[0]+conf_mtrx[1]))
    print("Sensitivity (Successfully predict cluster 2): ", conf_mtrx[3]/(conf_mtrx[2]+conf_mtrx[3]))



def Hessian(A,Y,w):
    pivot = np.exp(-np.dot(A,w))/((1+np.exp(-np.dot(A,w)))**2)
    # print (np.shape(A),np.shape(Y),np.shape(w))
    # print (np.shape(pivot))
    pivot = pivot.reshape(A.shape[0])
    D = np.diag(pivot)
    return np.dot(np.dot(np.transpose(A),D),A)

def logisticFunc(X,w):
    return 1/(1+np.exp(-np.dot(X,w)))

def GradientDescent(N,X,Y):
    # Wn+1 = Wn + r*Gradient_F(Xn)
    r = 0.001 # learning rate
    w = np.zeros((3,1), dtype='float64')
    w_past = np.zeros((3,1), dtype='float64')
    while True:
        # gradient = Xt(Y-1/(1+e^(-WtX)))
        # print(np.shape(Y))
        G = np.dot(np.transpose(X),Y-(1/(1+np.exp(-np.dot(X,w))))) 
        w += r*G
        if np.linalg.norm(w-w_past) <= 0.01:
            break
        w_past = w.copy()
    return w

def NewtonsMethod(N,X,Y):
    w = np.zeros((3,1), dtype='float64')
    w_past = np.zeros((3,1), dtype='float64')
    while True:
        # gradient = Xt(Y-1/(1+e^(-WtX)))
        G = np.dot(np.transpose(X),Y-(1/(1+np.exp(-np.dot(X,w)))))  
        try:
            H = Hessian(X,Y,w)
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            w += G
        else:
            w += np.dot(H_inv,G)
        if np.linalg.norm(w-w_past) <= 0.01:
            break
        w_past = w.copy()
    return w

def LogisticReg(N,Data,Label):
    # Xn+1 = Xn - [H_f(Xn)^(-1)]*Gradient_F(Xn)
    w_G = GradientDescent(N,Data,Label)
    predict_G = np.dot(Data,w_G)
    printGraph(1,Data,predict_G)
    printResult(1,Data,Label,w_G)
    print("")
    print('----------------------------------------')
    w_N = NewtonsMethod(N,Data,Label)
    predict_N = np.dot(Data,w_N)
    printGraph(2,Data,predict_N)
    printResult(2,Data,Label,w_N)
    

if __name__ == "__main__":
    N = 50
    (mx1,my1,mx2,my2,vx1,vy1,vx2,vy2) = 1,1,10,10,2,2,2,2
    # (mx1,my1,mx2,my2,vx1,vy1,vx2,vy2) = 1,1,3,3,2,2,4,4
    # ground truth
    Data = Data_point(N,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2) 
    Label = np.ones((N*2,1))
    for i in range(N):
        Label[i][0] = 0
    # print(Data)
    # print(Label)
    printGraph(0,Data,Label)
    LogisticReg(N,Data,Label)
    plt.show()
    # print(Data)
    # print(Label)
    print("---------------------------------------------------")
    print("")
    print("Test II:")
    N = 50
    (mx1,my1,mx2,my2,vx1,vy1,vx2,vy2) = 1,1,3,3,2,2,4,4
    # ground truth
    Data = Data_point(N,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2) 
    Label = np.ones((N*2,1))
    for i in range(N):
        Label[i][0] = 0
    # print(Data)
    # print(Label)
    printGraph(0,Data,Label)
    LogisticReg(N,Data,Label)
    plt.show()
