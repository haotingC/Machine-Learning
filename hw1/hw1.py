import numpy as np
import math as mt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def invrsLU(A):
    dim = A.shape[0]#3
    L = np.zeros(A.shape)
    U = np.zeros(A.shape)
    U[0][0] = A[0][0]
    L[0][0] = 1
    for i in range(1,dim):
        L[i][i] = 1
        U[0][i] = A[0][i]

    for i in range(dim-1):
        #L
        for j in range(i+1,dim):
            zigma = 0
            for k in range(dim):
                if k!=i:
                    zigma += L[j][k]*U[k][i]
            L[j][i] = (A[j][i]-zigma)/U[i][i]
        #U
        for j in range(i+1,dim):
            zigma = 0
            for k in range(dim):
                if k!=i+1:
                    zigma += L[i+1][k]*U[k][j]
            U[i+1][j] = A[i+1][j]-zigma

    # A^(-1)=(LU)^(-1)=x  LUx=I  Ux=y  Ly=I 
    iMatrix = np.identity(dim,dtype=float)
    x = np.zeros(L.shape)
    y = np.zeros(L.shape)
    # y from Ly=I
    for i in range(dim):
        for j in range(dim):
            zigma = 0
            for k in range(j):
                zigma += L[j][k]*y[k][i]
            y[j][i] = iMatrix[i][j]-zigma
    # print("y")
    # print(y)

    # x from Ux=y
    for i in range(dim):
        for j in range(dim-1,-1,-1):
            # summation of u[i][j]*x[j][i]
            zigma = 0.0
            for k in range(j + 1, dim):
                zigma += U[j][k] * x[k][i]
            # Evaluating x[i][i]
            x[j][i] = (y[j][i] - zigma) / U[j][j]
    # print("x")
    # print(x)
    return x

def LSE(A, b, lamda):
    #fx_coefficient: (A^TA+lamdaI)^(-1)(A^T)b
    A_transpose = A.transpose()
    iMatrix = np.identity(A.shape[1],dtype=float)
    invrs = np.dot(A_transpose,A)+lamda*iMatrix
    invrs = invrsLU(invrs)
    fx_coef = np.dot(np.dot(invrs,A_transpose),b)
    
    Ax_b = np.dot(A,fx_coef)-b
    loss = np.dot(Ax_b.transpose(),Ax_b)
    return fx_coef,loss[0][0]

def Newton_Method(A,b,x,base):
    # x(n+1) = x(n) - H^(-1)*G
    # thrd: H^(-1)*G
    fx_coef = np.zeros((base,1))
    Ax = np.dot(A,fx_coef)
    # gradient : f' = 2(A^T)Ax-2(A^T)b
    G = 2*np.dot(A.transpose(),Ax) - 2*np.dot(A.transpose(),b)
    # hessian : f" = 2(A^T)A
    H = 2*np.dot(A.transpose(),A)
    invrsH = invrsLU(H)
    thrd = np.dot(invrsH,G)
    
    while np.dot(thrd.transpose(),thrd) >= 0.00000001:
        Ax = np.dot(A,fx_coef)
        G = 2*np.dot(A.transpose(),Ax) - 2*np.dot(A.transpose(),b)
        H = 2*np.dot(A.transpose(),A)
        invrsH = invrsLU(H)
        thrd = np.dot(invrsH,G)
    
        fx_coef = fx_coef-thrd

    Ax_b = np.dot(A,fx_coef)-b
    loss = np.dot(Ax_b.transpose(),Ax_b)
    return fx_coef,loss[0][0]

def setdata(filename,base):
    f = open(filename, 'r')
    x = []
    y = []
    A = []
    b = []
    for line in f:
        row_data = line.strip().split(',')
        x.append(float(row_data[0]))  
        y.append(float(row_data[1]))  
        A.append([
            mt.pow(float(row_data[0]),base-i-1) for i in range(base)
        ])
        b.append([float(row_data[1])])
    f.close()
    x = np.array(x)
    y = np.array(y)
    A = np.array(A)
    b = np.array(b)
    
    return x,y,A,b

if __name__ == '__main__':
    n = int(input("n= "))#3
    lamda = int(input("lambda= "))#10000

    x,y,A,b = setdata("testfile.txt",n)
    # print(x)
    # print(y)
    # print(A)
    # print(b)

    print("LSE:")
    fx_coef,loss = LSE(A,b,lamda)
    fitting_line_LSE = "Fitting line: "
    for i in range(n):
        fitting_line_LSE += str(fx_coef[i][0])
        if i != n-1:
            fitting_line_LSE += "x^" + str(n-i-1)
            if fx_coef[i+1][0] >= 0:
                fitting_line_LSE += " + "
            else:
                fitting_line_LSE += " "
    print(fitting_line_LSE)
    print("Total error: "+str(loss))
    #print(loss)
    print("")

    plt.subplot(2,1,1)
    plt.plot(x,y,"ro")
    x1 = np.arange(np.min(x)-1,np.max(x)+1,0.1)
    y1 = 0
    for i in range(n):
        y1 += fx_coef[i] * np.power(x1,n-i-1)
    plt.plot(x1,y1)
    plt.xlim(np.min(x1),np.max(x1))


    print("Newton's Method:")
    fx_coef,loss = Newton_Method(A,b,x,n)
    fitting_line_NM = "Fitting line: "
    for i in range(n):
        fitting_line_NM += str(fx_coef[i][0])
        if i != n-1:
            fitting_line_NM += "x^" + str(n-i-1) 
            if fx_coef[i+1][0] >= 0:
                fitting_line_NM += " + "
            else:
                fitting_line_NM += " "
    print(fitting_line_NM)
    print("Total error: "+str(loss))
    
    plt.subplot(2,1,2)
    plt.plot(x,y,"ro")
    
    x2 = np.arange(np.min(x)-1,np.max(x)+1,0.1)
    y2 = 0
    for i in range(n):
        y2 += fx_coef[i] * np.power(x2,n-i-1)
    plt.plot(x2,y2)
    plt.xlim(np.min(x2),np.max(x2))

    plt.show()