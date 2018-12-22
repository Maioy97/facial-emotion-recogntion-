'''from tkinter import Label

import numpy as np

import os
from joblib import dump, load
import matplotlib.pyplot as plt
import glob
import datetime
'''
import cv2


def getHOGfeatures( img ):

    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 10
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # using HOG to extract Features

    #print(img.shape)
    #print(features.shape)
    try:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        features = hog.compute(img)
        features = features.ravel()
    except:#in case more than one image is passed take only one 
        img = cv2.resize(img[0], (64, 64), interpolation=cv2.INTER_AREA)
        features = hog.compute(img)
        features = features.ravel()
    #print(np.array(features).shape)
    return features


def getHOGfeatures128(img):
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 10
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # using HOG to extract Features

    #print(img.shape)
    #print(features.shape)
    try:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        features = hog.compute(img)
        features = features.ravel()
    except:#in case more than one image is passed take only one
        img = cv2.resize(img[0], (64, 64), interpolation=cv2.INTER_AREA)
        features = hog.compute(img)
        features = features.ravel()
    #print(np.array(features).shape)
    return features
