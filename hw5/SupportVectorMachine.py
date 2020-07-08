import numpy as np
from libsvm.svmutil import *
# from libsvm.svm import *
from scipy.spatial.distance import cdist
import math

def setdata():
    def read(filename):
        return np.genfromtxt(filename, delimiter=',', dtype="float64")        
    Xtrain = read('X_train.csv')
    Ytrain = read('Y_train.csv')
    Xtest = read('X_test.csv')
    Ytest = read('Y_test.csv')
    return Xtrain,Ytrain, Xtest,Ytest

def trainModel(x, y, xt, yt):
    prob  = svm_problem(y, x)
    paramLin  = svm_parameter("-t 0 -q")
    paramPoly = svm_parameter("-t 1 -q")
    paramRBF  = svm_parameter("-t 2 -q")
    modelLin  = svm_train(prob, paramLin)
    modelPoly = svm_train(prob, paramPoly)
    modelRBF  = svm_train(prob, paramRBF)
    print("linear")
    resultLin  = svm_predict(yt, xt, modelLin)
    print("")
    print("Polynomial")
    resultPoly = svm_predict(yt, xt, modelPoly)
    print("")
    print("RBF")
    resultRBF  = svm_predict(yt, xt, modelRBF)
    print("")

def trainModelwithGrid(x, y, xt, yt):
    def train(parameters):
        prob  = svm_problem(y, x)
        param = svm_parameter(parameters)
        model = svm_train(prob, param)        
        return model
    g = [2**i for i in range(-15,4,2)] 
    c = [2**i for i in range(-5,16,2)] 
    # linear parameter:c
    best_acc = 0.0
    for c_ele in range(len(c)):
        param = '-t 0 -c {} -v 3 -q'.format(c[c_ele])
        acc = train(param)
        print("c = ",c[c_ele])
        if best_acc < acc:
            best_acc = acc
            best_param = param
            best_c_ele = c_ele
    print("linear best accuracy:",best_acc)
    print("linear best parameter:",best_param)
    print("Test accuracy:")
    paramBest = '-t 0 -c {} -q'.format(c[best_c_ele])
    modelBest = train(paramBest)
    svm_predict(yt, xt, modelBest)
    print("============================")

    # polynomial parameter:c,g
    for c_ele in range(len(c)):
        for g_ele in range(len(g)):
            param = '-t 1 -c {} -g {} -v 3 -q'.format(c[c_ele], g[g_ele])
            acc = train(param)
            print("c = {}, g = {}".format(c[c_ele], g[g_ele]))
            if best_acc < acc:
                best_acc = acc
                best_param = param
                best_c_ele = c_ele
                best_g_ele = g_ele
    print("Polynomial best accuracy:",best_acc)
    print("Polynomial best parameter:",best_param)
    print("Test accuracy:")
    paramBest = '-t 1 -c {} -g {} -q'.format(c[best_c_ele], g[best_g_ele])
    modelBest = train(paramBest)
    svm_predict(yt, xt, modelBest)
    print("============================")

    # RBF parameter: c g
    for c_ele in range(len(c)):
        for g_ele in range(len(g)):
            param = '-t 2 -c {} -g {} -v 3 -q'.format(c[c_ele], g[g_ele])
            acc = train(param)
            print("c = {}, g = {}".format(c[c_ele], g[g_ele]))
            if best_acc < acc:
                best_acc = acc
                best_param = param
                best_c_ele = c_ele
                best_g_ele = g_ele
    print("RBF best accuracy:",best_acc)
    print("RBF best parameter:",best_param)
    print("Test accuracy:")
    paramBest = '-t 1 -c {} -g {} -q'.format(c[best_c_ele], g[best_g_ele])
    modelBest = train(paramBest)
    svm_predict(yt, xt, modelBest)
    print("============================")


def trainModelplusLinRBF(x, y, xt, yt):
    def LinearRBFKernel(x1,x2,g):
        linear_part = np.dot(x1,np.transpose(x2))
        RBF_part = np.exp(-g*cdist(x1,x2,'sqeuclidean'))
        mix = linear_part+RBF_part
        return np.hstack((np.arange(1, x1.shape[0]+1).reshape(-1, 1), mix))

    def train(k, parameters):
        prob  = svm_problem(y, k, isKernel=True)
        param = svm_parameter(parameters)
        model = svm_train(prob, param)
        return model

    g = [2**i for i in range(-15,4,2)]
    c = [2**i for i in range(-5,16,2)]
    # linear+RBF parameter:c g
    best_acc = 0.0
    for c_ele in range(len(c)):
        for g_ele in range(len(g)):
            param = '-t 4 -c {} -g {} -v 3 -q'.format(c[c_ele], g[g_ele])
            new_k = LinearRBFKernel(x,x,g[g_ele])
            acc = train(new_k, param)
            print("c = {}, g = {}".format(c[c_ele], g[g_ele]))
            if best_acc < acc:
                best_acc = acc
                best_param = param
                best_new_k = new_k
                best_c_ele = c_ele
                best_g_ele = g_ele
    print("LinRBF best accuracy:",best_acc)
    print("LinRBF best parameter:",best_param)

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = setdata()
    trainModel(x_train, y_train, x_test, y_test)

    trainModelwithGrid(x_train, y_train, x_test, y_test)

    trainModelplusLinRBF(x_train, y_train, x_test, y_test)
    
    
    # print(np.shape(x_train)) #(5000, 784)
    # print(np.shape(y_train)) #(5000,)
    # print(np.shape(x_test))  #(2500, 784)
    # print(np.shape(y_test))  #(2500,)
