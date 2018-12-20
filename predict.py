import numpy as np
import cv2
import os
import sys
from joblib import dump, load
import matplotlib.pyplot as plt
import glob
import get_HOG
import get_HOF
import facedetecion
import random
import collections

def PredictEmo(vid,start,end):
    print("here")
    predictedlabelsHOG = []
    predictedlabelsHOF = []
    # get random "pairs" of frames
    # get only the face of the frame
    # extract HoG & HOF Features
    # reload svm
    # predict label

    HOGsvm = cv2.ml.SVM_load('HOGsvm.xml')
    HOFsvm = cv2.ml.SVM_load('HOFsvm.xml')

    fps = vid.get(cv2.CAP_PROP_FPS)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = float(frameCount) / float(fps)  # in seconds
    else:
        print("can't render video")
        return -1,-1,-1
    FramestoCapturePS = 1  #fps / 3  # number of (frames)/(pair of frames) to capture in each second

    NumFrames = FramestoCapturePS * duration
    counter=0
    if NumFrames > 30 :
        NumFrames = 30
    print(NumFrames)
    for f in range(int(NumFrames)):
        while True:# to ensure the captured frame contains a face
            counter = counter + 1
            currentFrameTime = random.randint(fps * start, fps * end)
            vid.set(1, currentFrameTime)	#"1" the proberty index to get the spicified frame
            ret, frame1 = vid.read()		# ret indicates success of read process(true/false) (ex: returns false when end of 																						#video reached)
            if not ret:						# to ensure the frame was read correctly
                print(str(frameCount)+" err1")
                continue
            vid.set(1, currentFrameTime + 1)
            ret, frame2 = vid.read()
            if not ret:
                print("err2")
                continue
            face1, img1, x, y = facedetecion.detect(frame1) # get only the face of each frame
            face2, img2 = facedetecion.detect(frame2)
            if counter == 100:
                break
            if face1 is not None:			# check if either images does not contain a face go and get an other frame
                if face2 is not None:
                    break
        if counter == 100:
            print("SKIP******************* Can't find a face")
            break
        hoffeature = get_HOF.getHOFfeatures(face1, face2)
        hogfeature = get_HOG.getHOGfeatures(face1)
        predictedlabelsHOF.append(int(HOFsvm.predict(hoffeature.reshape(1, -1))[1].ravel()[0]))
        predictedlabelsHOG.append(int(HOGsvm.predict(hogfeature.reshape(1, -1))[1].ravel()[0]))
    #print(predictedlabelsHOG)
    #print(predictedlabelsHOF)
    vid.release() #same as closing a file after reading #release software resource & release hardware resource(ex:camera)
    # do majority voting and append respectively
    predictedlabelsHOGcounter=collections.Counter(predictedlabelsHOG)
    predictedlabelsHOFcounter=collections.Counter(predictedlabelsHOF)
    predictedlabelsBOTHcounter=collections.Counter(predictedlabelsHOF+predictedlabelsHOG)

    labelHog = predictedlabelsHOGcounter.most_common(1)[0][0]
    labelHof = predictedlabelsHOFcounter.most_common(1)[0][0]
    labelboth = predictedlabelsBOTHcounter.most_common(1)[0][0]
    return labelHof, labelHog, labelboth, x, y

def PredictGender(vid,start,end):
    print("here")
    predictedlabelsHOG = []

    # get random "pairs" of frames
    # get only the face of the frame
    # extract HoG & HOF Features
    # reload svm
    # predict label

    HOGsvm = cv2.ml.SVM_load('content%2FgenderDetectionModel.xml')


    fps = vid.get(cv2.CAP_PROP_FPS)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = float(frameCount) / float(fps)  # in seconds
    else:
        print("can't render video")
        return -1,-1,-1
    FramestoCapturePS = 1  #fps / 3  # number of (frames)/(pair of frames) to capture in each second

    NumFrames = FramestoCapturePS * duration
    counter=0
    if NumFrames > 30 :
        NumFrames = 30
    print(NumFrames)
    for f in range(int(NumFrames)):
        while True:# to ensure the captured frame contains a face
            counter = counter + 1
            currentFrameTime = random.randint(fps * start, fps * end)
            vid.set(1, currentFrameTime)	#"1" the proberty index to get the spicified frame
            ret, frame1 = vid.read()		# ret indicates success of read process(true/false) (ex: returns false when end of 																						#video reached)
            if not ret:						# to ensure the frame was read correctly
                print(str(frameCount)+" err1")
                continue
            vid.set(1, currentFrameTime + 1)
            ret, frame2 = vid.read()
            if not ret:
                print("err2")
                continue
            face1, img1 = facedetecion.detect(frame1) # get only the face of each frame
            face2, img2 = facedetecion.detect(frame2)
            if counter == 100:
                break
            if face1 is not None:			# check if either images does not contain a face go and get an other frame
                if face2 is not None:
                    break
        if counter == 100:
            print("SKIP******************* Can't find a face")
            break

        hogfeature = get_HOG.getHOGfeatures(face1)

        predictedlabelsHOG.append(int(HOGsvm.predict(hogfeature.reshape(1, -1))[1].ravel()[0]))
    #print(predictedlabelsHOG)
    #print(predictedlabelsHOF)
    vid.release() #same as closing a file after reading #release software resource & release hardware resource(ex:camera)
    # do majority voting and append respectively
    predictedlabelsHOGcounter=collections.Counter(predictedlabelsHOG)

    labelHog = predictedlabelsHOGcounter.most_common(1)[0][0]


    return  labelHog