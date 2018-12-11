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

path = ('../Data_test/images/')
list_txtFiles = os.listdir('../Data_test/emotion/')
count_Of_txtFiles = np.array(list_txtFiles).shape[0]
Actor_folder , emotion_folder , last_img_num , extention = list_txtFiles[0].split('_')
path_txt=str(list_txtFiles)+str(list_txtFiles[0])
path_images=(path+Actor_folder+'/'+emotion_folder)
with open (list_txtFiles) as f:
    next(f)
    print (f)
for filename in os.listdir(path_images):
    img = cv2.imread(os.path.join(path_images, filename))
    cv2.imshow('img',img)
    cv2.waitKey(0)

#for i in range(count_Of_txtFiles)



