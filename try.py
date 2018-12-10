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

folders = './images/'
for foldername in os.listdir(folders):
    newpath = folders+foldername
    list_images = os.listdir(newpath)
    count_images = np.array(list_images).shape
    count = 0
    while(count < count_images[0] - 1):
        img1 = cv2.imread(os.path.join(newpath, list_images[count]))
        img2 = cv2.imread(os.path.join(newpath, list_images[count + 1]))
        count = count + 1

        img1 = cv2.resize(img1, (64, 64), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (64, 64), interpolation=cv2.INTER_AREA)

        hsv = np.zeros_like(img1)
        hsv[..., 1] = 255

        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        #using HOF to extract Features
        flow = cv2.calcOpticalFlowFarneback(img1, img2,None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
