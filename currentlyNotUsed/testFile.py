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
import train_HOF
import train_HOG

trainType=input("Train Type   "
                "1= HOG     "
                "2= HOF     "
                "3= HOG&HOF3")

folders = './images/'

# test using HOG
if trainType=='1':


# test using HOF
elif trainType=='2':
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
            feature = train_HOF.trainHOF(img1,img2)
            lable =

# test using HOG & HOF
elif trainType=='3':
    print ('2')