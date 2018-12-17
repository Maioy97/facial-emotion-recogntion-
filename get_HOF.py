from tkinter import Label

import numpy as np
import cv2
import os
from joblib import dump, load
import matplotlib.pyplot as plt
import glob
import datetime



def getHOFfeatures(img1 , img2):
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
    # dense optical flow. It computes the optical flow for all the points in the frame
    #prev, next,flow(out), pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    for i in range(8):
        for j in range(8):
            histH = cv2.calcHist(hsv[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, 0], [1], None, [10], [0, 180])
            histV = cv2.calcHist(hsv[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, 2], [1], None, [10], [0, 255])
            h.append(histH)
            v.append(histV)

    features = [h, v]

    return np.array(features).reshape(1280,1)
