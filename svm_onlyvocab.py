# -*- coding: utf-8 -*-
"""
Created on Sun May 01 16:28:41 2016
@author: Prasanna D. Lalingkar

Performs the experiment of predicting the classes of the images using SVM.
This particular instance of code contains features generated using a combination of all three 
of feature extraction approaches as described in the readme file.
The parameters of the algorithm here are a snapshot of many different parameters used for SVM.

"""



import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import skimage
import cv2
import os
import winsound
from os import listdir, path
from os.path import isfile, join
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cluster import KMeans
import gc
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.cluster import vq
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs



# A bag of features is built in this section from all the images
print("Reading images to build vocabulary")


# Change the paths in this section to the appropriate location of training data and train.csv
mypathTrain='C:\\Users\\plalingk\\Downloads\\train'
onlyfiles = [ int(f.rstrip(".jpg")) for f in listdir(mypathTrain) if isfile(join(mypathTrain,f)) ]
onlyfiles.sort()

my_data = np.genfromtxt('C:\\Users\\plalingk\\Downloads\\train.csv', delimiter=',', dtype=int)
my_data = my_data[1:,:]

count, count1 = 0,0
features = []

for image in onlyfiles:
    count1 = count1+1
    if(count%2000 == 0):
        print('Sift done on: ',count)
    try:
        img = skimage.io.imread(mypathTrain+str("\\")+str(image)+".jpg")    
        sft = cv2.SIFT(50)        
        kp,fd = sft.detectAndCompute(img,None)
        for f in fd:
            features.append(f)
        count = count+1
    except:
        pass

data1 = np.array(features)


print("Doing Kmeans")


k = 300

mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=1000,
                       max_no_improvement=15, verbose=0)               
cent = mbk.fit(data1)
cluster_centers = cent.cluster_centers_


cluster_centers.shape

np.save("cluster",cent)
np.savetxt("centers.csv",cluster_centers,delimiter=',')


print("Vocabulary built")
print("Finding features on input data")



''' 
Finding features for all input images using a bag of features created by SIFT.
This section reads all the images and extracts the feature from them.
'''
# Finding features for input data
mypathTrain='C:\\Users\\plalingk\\Downloads\\train'
onlyfiles = [ int(f.rstrip(".jpg")) for f in listdir(mypathTrain) if isfile(join(mypathTrain,f)) ]
onlyfiles.sort()

my_data = np.genfromtxt('C:\\Users\\plalingk\\Downloads\\train.csv', delimiter=',', dtype=int)
my_data = my_data[1:,:]

count, count1 = 0,0
validId = []
validLabel = []
features = []

for image in onlyfiles:
    count1 = count1+1
    if(count%2000 == 0):
        print('Images done: ',count)
    try:
        img = skimage.io.imread(mypathTrain+str("\\")+str(image)+".jpg")
        sft = cv2.SIFT(100)        
        kp,feat = sft.detectAndCompute(img,None)
        code = cent.predict(feat)
        vocab = np.array(np.bincount(code,minlength = k))
        features.append(vocab)
        validId.append(my_data[count1-1,0])
        validLabel.append(np.argmax(my_data[count1-1,1:])+1)
        count = count+1        
    except:
        pass



X = np.array(features)
trainingLabels = np.array(validLabel)
trainingIds = np.array(validId)

trainingLabels.shape
trainingIds.shape
X.shape

print("Features extracted")


'''
For a few high dimension tasks I experimented with PCA to see how correct the outputs are.
Below is the PCA code which was used and can be uncommented in case it needs to be used.

If PCA is used comment the appropriate lines in the for loop below - to create TrainX1. 
'''


# pca on the training data
#pca = RandomizedPCA(n_components=300)
#TrainX = pca.fit_transform(X)
#TrainX.shape

# Cross validation on the Training model, parameters can be varied according to choice
kf = KFold(trainingIds.size, n_folds = 2, shuffle = True)


print("Starting CV SVM")
#for SVM

cnt = 0
for train_index, test_index in kf:
    TrainX1 = np.array(X[train_index])
    TrainY = np.array(trainingLabels[train_index])
    trainingIds1 = np.array(trainingIds[train_index])
    print(TrainX1.shape)
    print(TrainY.shape)
    print(trainingIds1.shape)
    clf = svm.NuSVC(nu=0.02, probability = True, cache_size = 24000)
    clf.fit(TrainX1, TrainY)  
    TestX = np.array(X[test_index])
    TestY = np.array(trainingLabels[test_index])
    trainingIds1 = np.array(trainingIds[test_index])
    print(TestX.shape)
    print(TestY.shape)
    print(trainingIds1.shape)
    mas = clf.score(TestX, TestY)
    print("Cross validation mean accuracy: ",mas)
    cnt=cnt+1
    if cnt == 1:
        maxMas = mas
        model = clf
    elif mas>maxMas:
        maxMas = mas
        model = clf



#gc.collect()


'''
Testing the data with the model built
Change the path in mypathTrain1 to the location of the test folder
'''
print("Preparing test Data")
# prepare testing data
mypathTrain1='C:\\Users\\plalingk\\Downloads\\test'
onlyfiles1 = [ int(f.rstrip(".jpg")) for f in listdir(mypathTrain1) if isfile(join(mypathTrain1,f)) ]
onlyfiles1.sort()


count, count1 = 0,0
validId = []
invalidId = []
images = []
features1 = []


for image in onlyfiles1:
    count1 = count1+1
    if(count%500 == 0):
        print('Images done: ',count)
    try:
        img = skimage.io.imread(mypathTrain1+str("\\")+str(image)+".jpg")
        sft = cv2.SIFT(200)        
        kp,feat = sft.detectAndCompute(img,None)
        code = cent.predict(feat)
        vocab = np.array(np.bincount(code,minlength = k))
        features1.append(vocab)
        validId.append(image)
        count = count+1        
    except:
        invalidId.append(image)
        pass


test = np.array(features1)
Ids = np.array(validId)
missedIds = np.array(invalidId)

test.shape
Ids.shape
missedIds.shape

'''
If PCA id used on the train data then use the same model to reduce the dimensionality of the test data.
Use appropriate TestX in case PCA is used.
'''
# pca on test data
#TestX = pca.transform(test)
#TestX = np.array(TestX)



print("Predicting on test data")
TestX = np.array(test)

# prediction using the test data for SVM
output = model.predict_proba(TestX)


# for SVM
op = np.column_stack((Ids.astype(int),output))


#missedIds
opMissed = np.array([np.array([0.125]*8)]*len(missedIds))
missedOp = np.column_stack((missedIds.astype(int),opMissed))


TestY = np.row_stack((op,missedOp))
print("Writing output to file")
# For SVM
np.savetxt("SVMonlyvocab.csv", TestY, delimiter=",")






