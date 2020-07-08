import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
import os.path
import random

def setdata(filename):
    image = Image.open(filename) # .format(png) .size(100*100) .mode(RGB)
    data = np.array(image) # rows/columns/RGB .size(30000) .shape(100 100 3)
    dataC = data.reshape((data.shape[0]*data.shape[1],data.shape[2])) # color data
    dataS = np.array([(i,j) for i in range(data.shape[0]) for j in range(data.shape[1])]) # spatial data
    image_size = image.size
    return dataC, dataS, image_size


def SimilarityGraph(dataC,dataS,Gs,Gc):
    Gram = np.exp(-Gs*cdist(dataS,dataS,'sqeuclidean'))*np.exp(-Gc*cdist(dataC,dataC,'sqeuclidean'))
    return Gram

def Laplacian(Gram,mode_S,testImage):
    if (os.path.exists('Laplacian_modeS{}_Img{}.npy'.format(mode_S,testImage))):
        L = np.load('Laplacian_modeS{}_Img{}.npy'.format(mode_S,testImage))
        return L
    else:
        W = Gram
        D = np.diag(np.sum(W,axis=1))
        L = D-W    # Graph laplacian
        if mode_S == 0:
            np.save('Laplacian_modeS{}_Img{}.npy'.format(mode_S,testImage), L)
            return L
        elif mode_S == 1:
            # Lsym = D^(-1/2)*L*D^(-1/2)
            Lsym = np.dot(np.dot(np.diag(1/np.diag(np.sqrt(D))),L),np.diag(1/np.diag(np.sqrt(D))))
            np.save('Laplacian_modeS{}_Img{}.npy'.format(mode_S,testImage), Lsym)
            return Lsym

def Eigen(L,k,mode_S,testImage):
    # Compute the first k eigenvectors u1, . . . ,uk of L. 
    if (os.path.exists('eigenValue_modeS{}_Img{}.npy'.format(mode_S,testImage)) and
        os.path.exists('eigenVector_modeS{}_Img{}.npy'.format(mode_S,testImage))):
        eigenValue = np.load('eigenValue_modeS{}_Img{}.npy'.format(mode_S,testImage))
        eigenVector = np.load('eigenVector_modeS{}_Img{}.npy'.format(mode_S,testImage))
    else:
        eigenValue, eigenVector = np.linalg.eig(L) 
        np.save('eigenValue_modeS{}_Img{}.npy'.format(mode_S,testImage), eigenValue)
        np.save('eigenVector_modeS{}_Img{}.npy'.format(mode_S,testImage), eigenVector)

    sortIdx = eigenValue.argsort() # find k smalleat eigenvalues
    # U (n,k)
    U = eigenVector.T[sortIdx[1:k+1]].T
    if mode_S == 0:
        return U
    elif mode_S == 1:
        # T (n,k) from U by normalizing the norm to row 1
        # t_ij = u_ij/sqrt(sigmak(u_ik**2))
        divisor = np.sqrt(np.sum(U**2, axis=1)).reshape(-1,1)
        T = U/divisor
        return T

def firstMean(eigen,k,mode_K):
    center = random.sample(range(0,10000),k)
    center = np.array(center)
    if mode_K == 0:  # random pick
        mean = eigen[center,:] # (k,k)
    elif mode_K == 1: # kmeans++
        mean = np.zeros((k, eigen.shape[1])) # (k, k)
        mean[0] = eigen[center[0],:]
        for t in range(1,k):
            D = np.zeros((eigen.shape[0],t))
            for i in range(eigen.shape[0]):
                for j in range(t):
                    D[i][j] = np.linalg.norm(eigen[i]-mean[j])
            D_list = np.min(D,axis=1)
            randomNum = np.random.rand()
            R = np.sum(D_list)*randomNum
            for i in range(eigen.shape[0]):
                R -= D_list[i]
                if R<=0:
                    mean[t] = eigen[i]
                    break
    return mean

def Kmeans(eigen,k,mode_K):
    data_cluster_record = np.zeros((10000,eigen.shape[0]))
    data_cluster = np.zeros(eigen.shape[0])
    # initial center pick
    mean = firstMean(eigen,k,mode_K) # (k,k)
    old_mean = np.zeros(mean.shape)
    cnt = 0
    while np.linalg.norm(mean-old_mean) > 1e-9:
        print('iter: ',cnt)
        # E-step: keep mu_k fixed, minimize J with respect to rnk
        for i in range(eigen.shape[0]):
            J = []
            for j in range(k):
                J.append(np.linalg.norm(eigen[i]-mean[j]))
            data_cluster[i] = np.argmin(J)
        data_cluster_record[cnt] = data_cluster
        
        # M-step: keep rnk fixed, minimize J with respect to mu_k 
        old_mean = mean
        shape = mean.shape
        mean=np.zeros(shape)
        for i in range(k):
            sumEigen=np.zeros(eigen.shape[1])
            r_nk=np.argwhere(data_cluster==i)
            for j in r_nk:
                sumEigen = sumEigen + eigen[j]
            if len(r_nk)>0:
                divisor = len(r_nk)
            else :
                divisor = 1
            mean[i] = sumEigen/divisor

        cnt += 1
    return data_cluster_record, cnt

def visualization(record,iteration,k,image_size,output_dir):
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            raise OSError("Can't create destination directory (%s)!" % (output_dir))
    
    gifs = []
    color = [(200,0,0,100),(0,200,0,100),(0,0,200,100),(200,200,200,100)]
    for i in range(iteration):
        pic = Image.new('RGB', image_size, (0, 0, 0)) 
        for j in range(record.shape[1]): # 10000 pixel
            rgba = color[int(record[i][j])]
            pic.putpixel((int(j%100),int(j/100)), rgba) # pic.putpixel((x,y), rgba)
        pic.save(os.path.join(output_dir,'k{}_{}.png'.format(k,i)))
        gifs.append(Image.open(os.path.join(output_dir,'k{}_{}.png'.format(k,i))))
    
    # Save into a GIF file that loops forever
    gifs[0].save(output_dir+'.gif', format='GIF',
               append_images=gifs[1:],
               save_all=True,
               duration=300, loop=0)

def drawCoordinates(eigen,finalCluster,k):
    # whether the points within the same cluster do have the same coordinates in the eigenspace of graph Laplacian or not.
    # eigen (n k)
    # finalCluster (n,)
    plt.figure()
    x = eigen[:,0]
    y = eigen[:,1]
    color_list = ['red','blue']
    # print(eigen.shape)
    for i in range(k):
        plt.plot(x[finalCluster==i],y[finalCluster==i],'.',color=color_list[i])
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.title("the coordinates of different clustering points")
    plt.show()
    

if __name__ == '__main__':
    mode_S = 0 # 0:unnormalized, 1:normalized
    mode_K = 1 # 0:kmeans, 1:kmean++
    testImage = 2 # image 1 or image 2
    k = 2 # number of clusters
    Gs,Gc = 0.001,0.001   # gamma of s and c for new kernel
    output_dir = "output_Spectral_normal{}_Kmeans{}_Img{}_k{}".format(mode_S,mode_K,testImage,k)

    dataC, dataS, image_size = setdata('image{}.png'.format(testImage)) # .shape (10000 3), (10000 2)
    GramMatrix = SimilarityGraph(dataC,dataS,Gs,Gc) #.shape (10000 10000)
    print("Similarity done!")
    L = Laplacian(GramMatrix,mode_S,testImage)
    print("Laplacian done!")
    eigen = Eigen(L,k,mode_S,testImage) #.shape (10000 k)
    print("Eigen done!")
    record, iteration = Kmeans(eigen,k,mode_K)
    print("Kmeans done!")
    visualization(record,iteration,k,image_size,output_dir)
    if k==2:
        drawCoordinates(eigen,record[iteration-1],k)