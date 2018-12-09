#read image and label
#call detect to get the face only
#feature extraction
#train the model
#save it

from tkinter import Label

import numpy as np
import cv2
import os
from joblib import dump, load
import matplotlib.pyplot as plt
import glob
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import datetime
# Read dataset
#print('reading data set')
#print(datetime.datetime.now())

Labels = []
Descriptor = []
folder = 'images'
winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        Desc = hog.compute(img)
        Desc = Desc.ravel()  # Flattening histogram into a feature vector
        Descriptor.append(Desc)
        # (1 = dog, 0 = cat)
        if filename[0] == 'm':
            Labels.append(0)
        elif filename[0] == 'f':
            Labels.append(1)
print('start training')
print(datetime.datetime.now())

knn = cv2.ml.KNearest_create()
knn.train(np.array(Descriptor),cv2.ml.ROW_SAMPLE,np.array(Labels))

print(Desc)
print(np.array(Descriptor).shape)
knn.save("train.yml")
#############for test
ret, results, neighbours ,dist = model.findNearest(Descriptor[0].reshape(1, -1), 3)
print(results)
####################
print('done')
print(datetime.datetime.now())

