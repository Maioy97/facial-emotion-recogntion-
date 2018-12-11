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

Labels = []
Descriptor = []

folders = './images/'
for foldername in os.listdir(folders):
    newpath = folders + foldername
    list_images = os.listdir(newpath)
    count_images = np.array(list_images).shape
    count = 0

    while (count < count_images[0] - 1):
        img1 = cv2.imread(os.path.join(newpath, list_images[count]))
        img2 = cv2.imread(os.path.join(newpath, list_images[count + 1]))
        count = count + 1

        img1 = cv2.resize(img1, (64, 64), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (64, 64), interpolation=cv2.INTER_AREA)

        hsv = np.zeros_like(img1)
        hsv[..., 1] = 255

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        features = []
        h = []
        v = []
        # using HOF to extract Features
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        for i in range(8):
            for j in range(8):
                histH = cv2.calcHist(hsv[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, 0], [1], None, [10], [0, 100])
                histV = cv2.calcHist(hsv[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, 2], [1], None, [10], [0, 100])
                h.append(histH)
                v.append(histV)

        features = [h, v]
        Descriptor.append(features)

        print (np.array(features).shape)

        # frrame lable
        # (0=neutral , 1=happy , 2=sad , 3=surprised , 4=angry , 5=Fear , 6=Disgust)

print('start training')
print(datetime.datetime.now())

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
# svm.setDegree(5.0)
# svm.setGamma(0.0001)
# svm.setCoef0(0.0)
# svm.setC(1)
# svm.setNu(0.21)
# svm.setP(0.0)
# svm.setClassWeights(None)
# svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
print(Desc)
print(np.array(Descriptor).shape)
svm.train(np.array(Descriptor), cv2.ml.ROW_SAMPLE, np.array(Labels))
svm.save('svmCatDogbdt.xml')
#########for test
print(svm.predict(Descriptor[0].reshape(1, -1))[1][0][0])
#################
print('done')
print(datetime.datetime.now())

