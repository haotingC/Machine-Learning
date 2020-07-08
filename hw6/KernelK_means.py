import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def KernelSpace(dataC,dataS,Gs,Gc):
    Gram = np.exp(-Gs*cdist(dataS,dataS,'sqeuclidean'))*np.exp(-Gc*cdist(dataC,dataC,'sqeuclidean'))
    return Gram

def firstMean(Gram,k,mode):
    mean = np.zeros((k,Gram.shape[0]))
    center = random.sample(range(0,10000),k)
    center = np.array(center)
    if mode == 0:  # random pick    
        mean = Gram[center,:]
    elif mode == 1: # kmeans++
        mean[0] = Gram[center[0],:]
        for t in range(1,k):
            D = np.zeros((Gram.shape[0],t))
            for i in range(Gram.shape[0]):
                for j in range(t):
                    D[i][j] = np.linalg.norm(Gram[i]-mean[j])
            D_list = np.min(D,axis=1)
            randomNum = np.random.rand()
            R = np.sum(D_list)*randomNum
            for i in range(Gram.shape[0]):
                R -= D_list[i]
                if R<=0:
                    mean[t] = Gram[i]
                    break
    return mean

def Kmeans(Gram,k,mode):
    data_cluster_record = np.zeros((10000,Gram.shape[0]))
    data_cluster = np.zeros(Gram.shape[0])
    # initial center pick
    mean = firstMean(Gram,k,mode)
    old_mean = np.zeros(mean.shape)
    cnt = 0
    while np.linalg.norm(mean-old_mean) > 1e-9: 
        print('iter: ',cnt)
        # E-step: keep μk fixed, minimize J with respect to rnk
        for i in range(Gram.shape[0]):
            J = []
            for j in range(k):
                J.append(np.linalg.norm(Gram[i]-mean[j]))
            data_cluster[i] = np.argmin(J)
        data_cluster_record[cnt] = data_cluster
        
        # M-step: keep rnk fixed, minimize J with respect to μk
        old_mean = mean
        shape = mean.shape
        mean=np.zeros(shape)
        for i in range(k):
            sumGram=np.zeros(Gram.shape[0])
            r_nk=np.argwhere(data_cluster==i)
            for j in r_nk:
                sumGram = sumGram + Gram[j]
            if len(r_nk)>0:
                divisor = len(r_nk)
            else :
                divisor = 1
            mean[i] = sumGram/divisor
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


if __name__ == '__main__':
    testImage = 2 # image 1 or image 2
    k = 4 # number of clusters
    mode = 1 # 0:kmeans, 1:kmean++
    output_dir = "output_KernelKmeans_Kmeans{}_Img{}_k{}".format(mode,testImage,k)
    Gs,Gc = 0.001,0.001   # gamma of s and c for new kernel
    dataC, dataS, image_size = setdata('image{}.png'.format(testImage)) # .shape (10000 3), (10000 2)

    GramMatrix = KernelSpace(dataC,dataS,Gs,Gc) #.shape (10000 10000)
    
    record, iteration = Kmeans(GramMatrix,k,mode)
    # print(record.shape)
    visualization(record,iteration,k,image_size,output_dir)