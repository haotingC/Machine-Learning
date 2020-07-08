import numpy as np
import os.path
import math
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

def color(Image, Label, label_appear_times):
    image_cnt = Image.shape[0] # = label_cnt
    pixel_cnt = Image.shape[1]
    # print(image_cnt) = 60000
    # print(pixel_cnt) = 784
    pixel_color = np.zeros((10,pixel_cnt))
    bin_value = np.zeros((10,pixel_cnt))
    for i in range(image_cnt):
        for j in range(pixel_cnt):            
            bin_value[int(Label[i])][j] += int(Image[i][j])

    for i in range(10):
        for j in range(pixel_cnt):
            avg_bin_value = (bin_value[i][j]/label_appear_times[i])
            if avg_bin_value >= 128:
                pixel_color[i][j] = 1
            else:
                pixel_color[i][j] = 0
    return pixel_color

def labelAppearTimes(Image, Label):
    image_cnt = Image.shape[0] # = label_cnt
    pixel_cnt = Image.shape[1]
    # print(image_cnt) = 60000
    # print(pixel_cnt) = 784
    label_appear_times = np.zeros((10))
    for i in range(image_cnt):
        label_appear_times[int(Label[i])] += 1

    return label_appear_times

def mean(Image, Label, label_appear_times):
    image_cnt = Image.shape[0] # = label_cnt
    pixel_cnt = Image.shape[1]
    # print(image_cnt) = 60000
    # print(pixel_cnt) = 784

    # every label's every pixel's value sum(adding from the 60000 image)
    pixel_sum = np.zeros((10,pixel_cnt)) 
    for i in range(image_cnt):
        for j in range(pixel_cnt):
            pixel_sum[int(Label[i])][j] += Image[i][j]
    
    # every label's every pixel's mean value
    mean_value = np.zeros((10,pixel_cnt))
    for i in range(10):
            mean_value[i] = pixel_sum[i] / label_appear_times[i]

    return mean_value

def variance(Image, Label, mean_value, label_appear_times):
    image_cnt = Image.shape[0] # = label_cnt
    pixel_cnt = Image.shape[1] 
    # print(image_cnt) = 60000
    # print(pixel_cnt) = 784

    # every label's every pixel's sum((x-u)^2)
    sum_pow2 = np.zeros((10,pixel_cnt))
    for i in range(image_cnt):
        for j in range(pixel_cnt):
            sum_pow2[int(Label[i])][j] += (Image[i][j]-mean_value[int(Label[i])][j])**2
    
    vari_value = np.zeros((10,pixel_cnt))
    for i in range(10):
        vari_value[i] = sum_pow2[i] / label_appear_times[i]

    return vari_value

def posterior_continuous(testImage_unit, testLabel_unit, mean_value, vari_value, label_appear_times):
    likelihood = np.zeros((10))
    prior = np.zeros((10))
    post_value = np.zeros((10))

    pixel_cnt = testImage_unit.shape[0]
    # posterior of every pixel
    # likelihood
    post_pixel = np.zeros((10,pixel_cnt))
    for i in range(10):
        for j in range(pixel_cnt):
            vari_value[i][j] += 1e-6
            post_pixel[i][j] = math.log(1/math.sqrt(2*math.pi*vari_value[i][j])) - (math.pow((testImage_unit[j]-mean_value[i][j]),2)/(2*vari_value[i][j]))
            likelihood[i] += post_pixel[i][j]
            vari_value[i][j] -= 1e-6
    # prior 
    for i in range(10):
        prior[i] = label_appear_times[i]/sum(label_appear_times)
    # posterior 
    for i in range(10):
        post_value[i] = likelihood[i]+math.log(prior[i])

    post_value /= sum(post_value)
    prediction = np.argmin(np.array(post_value))
    return prediction, post_value

def continuous(trainImage, trainLabel, testImage, testLabel, testDataCnt, label_appear_times):
    if (os.path.exists('mean.npy')
        and os.path.exists('variance.npy')
        and os.path.exists('color.npy')):
        mean_value = np.load('mean.npy')
        vari_value = np.load('variance.npy')
        pixel_color = np.load('color.npy')
    else:
        mean_value = mean(trainImage, trainLabel, label_appear_times)
        vari_value = variance(trainImage, trainLabel, mean_value, label_appear_times)
        pixel_color = color(trainImage, trainLabel, label_appear_times)
        np.save('mean',mean_value)
        np.save('variance', vari_value)   
        np.save('color',pixel_color) 

    prediction = np.zeros((testDataCnt))
    post_value = np.zeros((testDataCnt,10))
    for i in range(testDataCnt):
        prediction[i], post_value[i] = posterior_continuous(testImage[i],testLabel[i],mean_value,vari_value,label_appear_times)
        # print("Posterior (in log scale):")
        # for j in range(10):
        #     print(j, ":", post_value[j])
        # print('Prediction: ', prediction, ', Ans: ',testLabel[i])
        # print('')
        # if prediction == testLabel[i]:
        #     correct_cnt += 1

    return prediction, post_value
    

def probability(Image,Label):
    image_cnt = Image.shape[0] # = label_cnt
    pixel_cnt = Image.shape[1] 
    # print(image_cnt) = 60000
    # print(pixel_cnt) = 784
    # every pixel is classified into 32bins
    # every label's every pixel's different bin_value appearence times
    Label_pixel_bins_appear_times = np.zeros((10,pixel_cnt,32))
    for i in range(image_cnt):
        for j in range(pixel_cnt):
            if(int(Image[i][j])>=32):
                bin_value = 31
            else:
                bin_value = int(Image[i][j])
            Label_pixel_bins_appear_times[int(Label[i])][j][bin_value] += 1
    
    # every label's all pixels bins appearence probability
    Label_pixel_bins_appear_prob = np.zeros((10,pixel_cnt,32))
    for i in range(10):
        Label_pixel_bins_appear_prob[i] = Label_pixel_bins_appear_times[i] / sum(Label_pixel_bins_appear_times[i])
        
    # log of Label_pixel_bins_appear_prob
    prob_log_value = np.zeros((10,pixel_cnt,32))
    for i in range(10):
        for j in range(pixel_cnt):
            for k in range(32):
                # make sure no zero
                if Label_pixel_bins_appear_prob[i][j][k] == 0:
                    prob_log_value[i][j][k] = math.log(1e-9)
                else:
                    prob_log_value[i][j][k] = math.log(Label_pixel_bins_appear_prob[i][j][k])

    return prob_log_value

def posterior_discrete(testImage_unit,testLabel_unit,prob_log_value):
    likelihood = np.zeros((10))
    prior = np.zeros((10))
    post_value = np.zeros((10))
    
    pixel_cnt = testImage_unit.shape[0]
    # posterior of every pixel
    # likelihood
    post_pixel = np.zeros((10,pixel_cnt))
    for i in range(10):
        for j in range(pixel_cnt):
            if(int(testImage_unit[j])>=32):
                bin_value = 31
            else:
                bin_value = int(testImage_unit[j])
            likelihood[i] += prob_log_value[i][j][bin_value]
    # prior 
    for i in range(10):
        prior[i] = np.sum(np.exp(prob_log_value[i]))
    prior /= sum(prior)
    # posterior 
    for i in range(10):
        post_value[i] = likelihood[i]+prior[i]
    post_value /= sum(post_value)
    prediction = np.argmin(np.array(post_value))
    
    return prediction, post_value

def discrete(trainImage, trainLabel, testImage, testLabel, testDataCnt):
    if (os.path.exists('probability.npy')):
        prob_log_value = np.load('probability.npy')
    else:
        prob_log_value = probability(trainImage, trainLabel)
        np.save('probability',prob_log_value) 
    
    prediction = np.zeros((testDataCnt))
    post_value = np.zeros((testDataCnt,10))
    for i in range(testDataCnt):
        prediction[i], post_value[i] = posterior_discrete(testImage[i],testLabel[i],prob_log_value)
    
    return prediction, post_value
    

if __name__ == '__main__':
    # 0(discrete),1(continuous)
    print("discrete mode:0 ; continuous mode:1")
    mode = int(input("mode= "))
    # you want to test how much test_images
    testDataCnt = 500
    
    trainImage, trainLabel, testImage, testLabel = setdata()
    if (os.path.exists('color.npy')
        and os.path.exists('label_appear_times.npy')):
        pixel_color = np.load('color.npy')
        label_appear_times = np.load('label_appear_times.npy')
    else:
        label_appear_times = labelAppearTimes(trainImage, trainLabel)
        pixel_color = color(trainImage, trainLabel, label_appear_times)
        np.save('label_appear_times',label_appear_times) 
        np.save('color',pixel_color) 

    # Naive Bayes Classifier
    if mode == 0:
        # Tally gray value to 32 bins
        prediction, post_value = discrete(trainImage/8, trainLabel, testImage/8, testLabel, testDataCnt)
    else:
        prediction, post_value = continuous(trainImage, trainLabel, testImage, testLabel, testDataCnt, label_appear_times)

    # Print out the result
    correct_cnt = 0
    for i in range(testDataCnt):
        print("Postirior (in log scale):")
        for j in range(10):
            print(j, ":", post_value[i][j])
        print('Prediction: ', prediction[i], ', Ans: ',testLabel[i])
        print('')
        if prediction[i] == testLabel[i]:
            correct_cnt += 1

    pixel_cnt = trainImage.shape[1]
    print("Imagination of numbers in Bayesian classifier:")
    for i in range(10):
        print('')
        print(i,':')
        for j in range(pixel_cnt):
            if pixel_color[i][j] == 1:
                print(1,end='')
            else:
                print(0,end='')
            if (j+1)%28 == 0:
                print('')

    print("Error rate: ", 1.0-float(correct_cnt/testDataCnt))
