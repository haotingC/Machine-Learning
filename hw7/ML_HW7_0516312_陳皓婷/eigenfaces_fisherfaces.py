import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from PIL import Image
import os
from itertools import chain
from scipy.misc import imread

def read_IMG(dirpath,subjects):
    ims = os.listdir(dirpath)
    num_ims = subjects*15
    pixels = [[] for i in range(num_ims)]
    label = [[] for i in range(num_ims)]
    i = 0
    for im in ims:
        pixels[i] = list(chain.from_iterable(imread(dirpath+'/'+im)))
        label[i] = int(i/subjects) 
        i += 1
    pixels = np.array(pixels)
    label = np.array(label)
    return pixels, label

def kernel(x,kernel_mode):
    if kernel_mode == "linear":
        return x@x.T
    elif kernel_mode == "Polynomial":
        g = 0.01
        c = 0
        d = 3
        return (g*x@x.T+c)**d
    elif kernel_mode == "RBF":
        g = 0.01
        return np.exp(-g*cdist(x,x,'sqeuclidean'))

def pca(x,y,t_x,t_y, k, kernel_mode):
    mean = np.mean(x, axis=0) 
    delt = x-mean             
    print("kernel_mode: ",kernel_mode)
    if kernel_mode == 0:
        S = np.cov(x, bias=True)  
    else:
    	S = kernel(x,kernel_mode)
    	one_N = np.ones((x.shape[0],x.shape[0]))/x.shape[0]
    	S = S - one_N@S - S@one_N + one_N@S@one_N
    eigenValues, eigenVectors = np.linalg.eig(S) 
    ### sorted largest->smallest ###
    eigenValues = np.abs(eigenValues)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    # count important ones
    total_Val = np.sum(eigenValues)
    sumup = 0
    for i in range(len(idx)):
        sumup += eigenValues[i]
        if sumup/total_Val >= 0.97: # set threshold = 97%
            eigenValues = eigenValues[0:i]
            eigenVectors = eigenVectors[:,:i]
            break
    # print(eigenValues.shape,eigenVectors.shape)

    transform = delt.T@eigenVectors
    transform = np.real(transform)
    if kernel_mode == 0:
        showout_transform(transform)
    
    reconstruct = transform@(transform.T@delt.T) + mean.reshape(-1,1)
    if kernel_mode == 0:
        showout_reconstruct(x, reconstruct)
    
    t_z = transform.T@(t_x-mean).T   # projected faces (low_dim)
    z = transform.T@delt.T           # projected faces (low_dim)
    dist = np.zeros(x.shape[0])          # size = 135
    predicted_y = np.zeros(t_x.shape[0]) # size = 30

    for i in range(predicted_y.size):   # 0-30
        for j in range(dist.size):      # 0-135
            dist[j] = cdist(t_z[:,i].reshape(1,-1),z[:,j].reshape(1,-1),'sqeuclidean')
        nn = y[np.argsort(dist)[:k]] # closest k-labels
        sort_uniq_nn, sort_nn_appearcounts = np.unique(nn, return_counts=True) 
        idx_pred = np.argmax(sort_nn_appearcounts)
        predicted_y[i] = sort_uniq_nn[idx_pred]

    error_rate=len(np.argwhere(t_y-predicted_y))/predicted_y.size
    print("predicted_y:",predicted_y)
    if kernel_mode == 0:
        print("pca accuarcy rate: ", 1.-error_rate)
    else:
        print("kpca accuarcy rate: ", 1.-error_rate)

    return transform, reconstruct, z

def lda(pca_transform,pca_x,x,y,t_x,t_y, k):
    # mean function
    allmean = np.mean(pca_x, axis=1) 

    # within class scatter
    within_class_S = np.zeros((pca_x.shape[0],pca_x.shape[0]))  
    for i in range(15):
        within_class_S += np.cov(pca_x[:,i*9:i*9+9], bias=True) 
    
    # in between class scatter 
    in_btw_class_S = np.zeros((pca_x.shape[0],pca_x.shape[0]))  
    for i in range(15):
        cl_mean = np.mean(pca_x.T[i*9:i*9+9,:], axis=0)         
        in_btw_class_S += 9*(cl_mean-allmean).reshape(-1,1)@(cl_mean-allmean).reshape(1,-1) 
   
    S = np.linalg.inv(within_class_S)@in_btw_class_S  
    eigenValues, eigenVectors = np.linalg.eig(S)      
    ### sorted largest->smallest ###
    eigenValues = np.abs(eigenValues)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    # count important ones
    total_Val = np.sum(eigenValues)
    sumup = 0
    for i in range(len(idx)):
        sumup += eigenValues[i]
        if sumup/total_Val >= 0.97: # set threshold = 99%
            if i<25:
                i=25
            eigenValues = eigenValues[0:i]
            eigenVectors = eigenVectors[:,:i]
            break
    # print(eigenValues.shape,eigenVectors.shape) (25,)(50,25)

    transform = pca_transform@eigenVectors # (45045,25)
    transform = np.real(transform)
    if kernel_mode == 0:
        showout_transform(transform)

    mean = np.mean(x, axis=0)
    delt = x-mean
    reconstruct = transform@(transform.T@delt.T) + mean.reshape(-1,1)
    if kernel_mode == 0:
        showout_reconstruct(x, reconstruct)
    
    t_z = transform.T@(t_x-mean).T   # projected faces (low_dim)
    z = transform.T@delt.T           # projected faces (low_dim)
    dist = np.zeros(x.shape[0])          # size = 135
    predicted_y = np.zeros(t_x.shape[0]) # size = 30

    for i in range(predicted_y.size):   # 0-30
        for j in range(dist.size):      # 0-135
            dist[j] = cdist(t_z[:,i].reshape(1,-1),z[:,j].reshape(1,-1),'sqeuclidean')
        nn = y[np.argsort(dist)[:k]] # closest k-labels
        sort_uniq_nn, sort_nn_appearcounts = np.unique(nn, return_counts=True) 
        idx_pred = np.argmax(sort_nn_appearcounts)
        predicted_y[i] = sort_uniq_nn[idx_pred]

    error_rate=len(np.argwhere(t_y-predicted_y))/predicted_y.size
    print("predicted_y: ",predicted_y)
    if kernel_mode == 0:
        print("lda accuarcy rate: ", 1.-error_rate)
    else:
        print("klda accuarcy rate: ", 1.-error_rate)
        
    return transform, reconstruct

def showout_transform(transform):
    # eigenface
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(transform[:,i].reshape(231,195), cmap="gray")
    plt.show()

def showout_reconstruct(x, reconstruct):
    # random pick 10
    fig, axes = plt.subplots(2, 10)
    idx = np.random.choice(135, 10, replace=False)
    for i, random_idx in enumerate((idx)):
        axes[0, i].imshow(x[random_idx].reshape(231,195), cmap="gray")
        axes[1, i].imshow(reconstruct[:,random_idx].reshape(231,195), cmap="gray")
    plt.show()


if __name__ == "__main__":
    train_img, train_label = read_IMG("./Yale_Face_Database/Training",9)  
    test_img, test_label = read_IMG("./Yale_Face_Database/Testing",2)
    k_knn = 3
    kernel_modes = [0,"linear","Polynomial","RBF"]
    for kernel_mode in kernel_modes:
        pca_transform, pca_reconstruct, pca_x= pca(train_img,train_label, test_img, test_label, k_knn, kernel_mode) # x_mean(45045,) eigenVal(135,) eigenVec(135,25)
        lda_transform, lda_reconstruct= lda(pca_transform, pca_x,train_img,train_label, test_img, test_label, k_knn)

    # kernel
    