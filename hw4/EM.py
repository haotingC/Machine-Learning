import numpy as np
import os.path
import math, random
import pandas
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

def setImage(filename):
    f = open(filename, 'rb')
    array = []
    magic_num = int.from_bytes(f.read(4), 'big')
    image_cnt = int.from_bytes(f.read(4), 'big')
    row_cnt = int.from_bytes(f.read(4), 'big')
    col_cnt = int.from_bytes(f.read(4), 'big')
    # print(magic_num)
    # print(image_cnt)
    # print(row_cnt)
    # print(col_cnt)
    pixel_cnt = row_cnt*col_cnt
    # print(pixel_cnt)
    array = np.zeros((image_cnt,pixel_cnt))
    for i in range(image_cnt):
        for j in range(pixel_cnt):
            array[i][j] = int.from_bytes(f.read(1), 'big')
    f.close()
    # print(np.shape(array))
    array = np.floor(array/128) #tally into 2 bins
    return array

def setLabel(filename):
    f = open(filename, 'rb')
    array = []
    magic_num = int.from_bytes(f.read(4), 'big')
    label_cnt = int.from_bytes(f.read(4), 'big')
    # print(magic_num)
    # print(label_cnt)
    array = np.zeros((label_cnt))
    for i in range(label_cnt):
        array[i] = int.from_bytes(f.read(1), 'big')
    f.close
    return array

def setdata():
    if (os.path.exists('trainImage.npy') and
        os.path.exists('trainLabel.npy') and
        os.path.exists('testImage.npy') and
        os.path.exists('testLabel.npy')):
        # print("save1")
        trainImage = np.load('trainImage.npy')
        trainLabel = np.load('trainLabel.npy')
        testImage = np.load('testImage.npy')
        testLabel = np.load('testLabel.npy')
        # print("save2")
    else:
        trainImage = setImage('train-images.idx3-ubyte')
        trainLabel = setLabel('train-labels.idx1-ubyte')
        testImage = setImage('t10k-images.idx3-ubyte')
        testLabel = setLabel('t10k-labels.idx1-ubyte')
        np.save('trainImage', trainImage)
        np.save('trainLabel', trainLabel)
        np.save('testImage', testImage)
        np.save('testLabel', testLabel)
    return trainImage, trainLabel, testImage, testLabel


def setBernoulli(Image):
    pixel_cnt = Image.shape[1] # = 784
    lamd = np.full((10),0.1) # 0.1 0.1 0.1 0.1 0.1 0.1....
    p_pixels = np.random.rand(10, pixel_cnt) # random(0~1) matrix: 10*784
    return lamd, p_pixels
    
def E_step(image_cnt,Image,lamd,p):
    #  responsibility: w(x1|0)...w(x1|9), w(x2|0)...w(x2|9),....., w(x60000|0)...w(x60000|9)
    #  wi = exp(Zi)/sum((Zi))
    z = np.zeros((image_cnt,10)) # = 60000*10
    w = np.zeros((image_cnt,10)) # = 60000*10 
    for i in range((image_cnt)): # 60000
        unitImage = Image[i] # 784 pixels
        for j in range(10):
            z[i][j] = np.log(lamd[j])+np.sum(p[j]*unitImage)+np.sum((1-p[j])*(1-unitImage))
        w[i] = np.exp(z[i]-max(z[i]))
        w[i] = w[i]/np.sum(w[i])
    return w

def M_step(image_cnt,Image,w):
    wSum = np.sum(w,axis=0) # 10*1
    lamd = wSum/image_cnt
    p = np.dot(np.transpose(w),Image)
    for i in range(10):
        p[i]/=wSum[i]
    return lamd,p

def EM(Image,lamd,p_pixels):  
    image_cnt = Image.shape[0] # = 60000
    pixel_cnt = Image.shape[1] # = 784
    p = p_pixels.copy() # = 10*784

    lamd_past = lamd.copy()
    p_past = p.copy()
    iterate_cnt = 0
    while True:
        iterate_cnt+=1
        # E step
        w = E_step(image_cnt,Image,lamd,p)
        # M step
        lamd, p = M_step(image_cnt,Image,w)
        # print("lamd:",lamd," p_pixels:")
        # print(" iterate:",iterate_cnt," del_lamd:",np.linalg.norm(lamd-lamd_past)," del_p:",np.linalg.norm(p-p_past))
        
        if np.linalg.norm(lamd-lamd_past)<=0.1 and np.linalg.norm(p-p_past)<=0.1:
            break
        lamd_past = lamd.copy()
        p_past = p.copy()
        
    return lamd, p, iterate_cnt

def LabelClassMapping(Image, Label, lamd, p_pixels):
    # label[i] = class[j]
    p = p_pixels.copy() # = 10*784
    labl_cl_mapping = np.full((10),-1)
    num_probility = np.zeros((10)) 
    for i in range(10): #10
        Image_is_num_i = Image[Label == i]
        predict_num_appear_times = np.zeros((10))
        for j in Image_is_num_i: # j: single image, 784 pixels
            for k in range(10):
                num_probility[k] = lamd[k]*np.prod(p[k][j==1])
            predict_num = np.argmax(num_probility)
            predict_num_appear_times[predict_num] += 1
        
        predict_sort_index = np.argsort(predict_num_appear_times)
        
        index = 9
        while np.any(labl_cl_mapping == predict_sort_index[index]):
            index -= 1
        labl_cl_mapping[i] = predict_sort_index[index]
    return labl_cl_mapping

def ConfusionMatrix(Image, Label, lamd, p_pixels, labl_cl_mapping):
    conf_mtrx = np.zeros((10,4))
    for i in range(10):
        num_y = Image[Label == i]
        num_n = Image[Label != i]
        for r in num_y:
            predict_P = np.zeros((10))
            for j in range(10):
                predict_P[j] = lamd[j]*np.prod(p_pixels[j][r == 1])
            predict = np.argmax(predict_P)
            mapping_index = np.argwhere(labl_cl_mapping == predict)
            predict_label = mapping_index.item()
            if predict_label==i:
                conf_mtrx[i][0] +=1
            else:
                conf_mtrx[i][1] +=1
        for r in num_n:
            predict_P = np.zeros((10))
            for j in range(10):
                predict_P[j] = lamd[j]*np.prod(p_pixels[j][r == 1])
            predict = np.argmax(predict_P)
            mapping_index = np.argwhere(labl_cl_mapping == predict)
            predict_label = mapping_index.item()
            if predict_label==i:
                conf_mtrx[i][2] +=1
            else:
                conf_mtrx[i][3] +=1
    return conf_mtrx

def printResult(p, iterate_cnt, label_mapping, conf_mtrx, error_rate):
# Print out the result
    for i in range(10):
        print("class ", i, ":")
        lbl = label_mapping[i]
        pixels = p[lbl].copy()
        pixels[p[lbl].copy()>0.5] = 1
        pixels[p[lbl].copy()<=0.5] = 0
        pixels = pixels.reshape(28,28)
        for j in range(28):
            for k in range(28):
                if pixels[j][k] == 1:
                    print(1,end='')
                else:
                    print(0,end='')
                if (k+1)%28 == 0:
                    print('')
        print("")
    
    for i in range(10):
        print("------------------------------------------------------------")
        print("")
        print("Confusion Matrix ", i,":")
        print("                 Predict number ", i," Predict not number ", i)
        print("Is number ",i,"       ",conf_mtrx[i][0],"                 ", conf_mtrx[i][1])
        print("Isn't number ",i,"    ",conf_mtrx[i][2],"                ", conf_mtrx[i][3])
        print("Sensitivity (Successfully predict number ", i, ") : ", conf_mtrx[i][0]/(conf_mtrx[i][0]+conf_mtrx[i][1]))
        print("Specificity (Successfully predict not number ", i, "): ", conf_mtrx[i][3]/(conf_mtrx[i][2]+conf_mtrx[i][3]))
        print("")
    
    print('Total Iteration to converge: ', iterate_cnt)
    print('Total error rate: ', error_rate)


    

if __name__ == '__main__':
    testDataCnt = 500
    
    trainImage, trainLabel, testImage, testLabel = setdata()

    lamd, p_pixels = setBernoulli(trainImage) # lamb=1*10, p_pixel = 10*784
    lamd, p_pixels, iterate_cnt = EM(trainImage,lamd,p_pixels)
    
    # label[i] = class[j]
    labl_cl_mapping = LabelClassMapping(trainImage, trainLabel, lamd, p_pixels)
    print(labl_cl_mapping)

    # confusion matrix
    conf_mtrx = ConfusionMatrix(trainImage, trainLabel, lamd, p_pixels, labl_cl_mapping)
    
    # error rate
    correct_cnt = 0
    for i in range(testDataCnt):
        predict_P = np.zeros((10))
        for j in range(10):
            predict_P[j] = lamd[j]*np.prod(p_pixels[j][testImage[i]==1])
        predict = np.argmax(predict_P)
        mapping_index = np.argwhere(labl_cl_mapping == predict)
        predict_label = mapping_index.item()
        if predict_label == testLabel[i]:
            correct_cnt += 1
    error_rate = 1- correct_cnt/testDataCnt

    printResult(p_pixels, iterate_cnt, labl_cl_mapping, conf_mtrx, error_rate)
    
