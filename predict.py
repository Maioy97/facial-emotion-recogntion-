'''
import os
import sys
from joblib import dump, load
import matplotlib.pyplot as plt
import glob
'''
import numpy as np
import cv2
import get_HOG
import get_HOF
import facedetecion
import random
import collections



def predictEmoHOF(vid, start, end, hof_svm_emo):
    print("PredictEmoHOF")
    predicted_labels_hof_emo = []
    # get random "pairs" of frames
    # get only the face of the frame
    # extract HoG & HOF Features
    # reload svm
    # predict label

    fps = vid.get(cv2.CAP_PROP_FPS)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = end - start  # in seconds
    else:
        print("can't render video")
        return -1, -1, -1
    FramestoCapturePS = fps / 3  # number of (frames)/(pair of frames) to capture in each second

    NumFrames = FramestoCapturePS * duration
    counter = 0
    if NumFrames > 30:
        NumFrames = 30
    print(NumFrames)
    for f in range(int(NumFrames)):
        while True:  # ensuring the captured frame contains a face
            counter = counter + 1
            currentFrameTime = random.randint(int(fps * start), int(fps * end))
            vid.set(1, currentFrameTime)  # "1" the property index to get the specified frame
            ret, frame1 = vid.read()  # ret indicates success of read process(true/false)(ex: false > end of video)
            if not ret:  # to ensure the frame was read correctly
                print(str(frameCount) + " err1")
                continue
            vid.set(1, currentFrameTime + 1)
            ret, frame2 = vid.read()
            if not ret:
                print("err2")
                continue
            face1, img1, x, y = facedetecion.detect(frame1)  # get only the face of each frame
            face2, img2, x, y = facedetecion.detect(frame2)
            if counter == 100:
                break
            if face1 is not None:  # check if either images does not contain a face go and get an other frame
                if face2 is not None:
                    break
        if counter == 100:
            print("SKIP******************* Can't find a face")
            break
        hoffeature = get_HOF.getHOFfeatures(face1, face2)
        predicted_labels_hof_emo.append(int(hof_svm_emo.predict(hoffeature.reshape(1, -1))[1].ravel()[0]))
    # do majority voting and append respectively
    predicted_labels_hof_emocounter = collections.Counter(predicted_labels_hof_emo)
    labelHof=0
    try:
        labelHof = predicted_labels_hof_emocounter.most_common(1)[0][0]
    except IndexError:
        print ("there are no faces")
        return -1
    return labelHof
def predictEmoHOG(vid, start, end, hog_svm_emo):
    print("PredictEmoHOG")
    predicted_labels_hog_emo = []
    # get random "pairs" of frames
    # get only the face of the frame
    # extract HoG & HOF Features
    # reload svm
    # predict label
    fps = vid.get(cv2.CAP_PROP_FPS)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = end - start  # in seconds
    else:
        print("can't render video")
        return -1, -1, -1
    FramestoCapturePS =  fps / 3  # number of (frames)/(pair of frames) to capture in each second

    NumFrames = FramestoCapturePS * duration
    counter = 0
    if NumFrames > 30:
        NumFrames = 30
    print(NumFrames)
    for f in range(int(NumFrames)):
        while True:  # ensuring the captured frame contains a face
            counter = counter + 1
            currentFrameTime = random.randint(int(fps * start), int(fps * end))
            vid.set(1, currentFrameTime)  # "1" the property index to get the specified frame
            ret, frame1 = vid.read()  # ret indicates success of read process(true/false)(ex: false > end of video)
            if not ret:  # to ensure the frame was read correctly
                print(str(frameCount) + " err1")
                continue
            vid.set(1, currentFrameTime + 1)
            ret, frame2 = vid.read()
            if not ret:
                print("err2")
                continue
            face1, img1, x, y = facedetecion.detect(frame1)  # get only the face of each frame
            face2, img2, x, y = facedetecion.detect(frame2)
            if counter == 100:
                break
            if face1 is not None:  # check if either images does not contain a face go and get an other frame
                if face2 is not None:
                    break
        if counter == 100:
            print("SKIP******************* Can't find a face")
            break
        hogfeature = get_HOG.getHOGfeatures(face1)
        predicted_labels_hog_emo.append(int(hog_svm_emo.predict(hogfeature.reshape(1, -1))[1].ravel()[0]))
    # do majority voting and append respectively
    predicted_labels_hog_emocounter = collections.Counter(predicted_labels_hog_emo)
    labelHog=0
    try:
        labelHog = predicted_labels_hog_emocounter.most_common(1)[0][0]
    except IndexError:
        print ("there are no faces")
        return -1
    return labelHog
def PredictEmoBoth(vid, start, end, hog_svm_emo ,hof_svm_emo):
    print("PredictEmoBoth")
    predicted_labels_hog_emo = []
    predicted_labels_hof_emo = []
    # get random "pairs" of frames
    # get only the face of the frame
    # extract HoG & HOF Features
    # reload svm
    # predict label
    fps = vid.get(cv2.CAP_PROP_FPS)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = end - start  # in seconds
    else:
        print("can't render video")
        return -1, -1, -1
    FramestoCapturePS =  fps / 3  # number of (frames)/(pair of frames) to capture in each second

    NumFrames = FramestoCapturePS * duration
    counter = 0
    if NumFrames > 30:
        NumFrames = 30
    print(NumFrames)
    for f in range(int(NumFrames)):
        while True:  # ensuring the captured frame contains a face
            counter = counter + 1
            currentFrameTime = random.randint(int(fps * start), int(fps * end))
            vid.set(1, currentFrameTime)   # "1" the property index to get the specified frame
            ret, frame1 = vid.read()	   # ret indicates success of read process(true/false)(ex: false > end of video)
            if not ret:					   # to ensure the frame was read correctly
                print(str(frameCount)+" err1")
                continue
            vid.set(1, currentFrameTime + 1)
            ret, frame2 = vid.read()
            if not ret:
                print("err2")
                continue
            face1, img1, x, y = facedetecion.detect(frame1)  # get only the face of each frame
            face2, img2, x, y = facedetecion.detect(frame2)
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
        predicted_labels_hof_emo.append(int(hof_svm_emo.predict(hoffeature.reshape(1, -1))[1].ravel()[0]))
        predicted_labels_hog_emo.append(int(hog_svm_emo.predict(hogfeature.reshape(1, -1))[1].ravel()[0]))

    # do majority voting and append respectively

    predictedlabelsBOTHcounter =collections.Counter(predicted_labels_hof_emo+predicted_labels_hog_emo)
    labelboth=0
    try:
        labelboth = predictedlabelsBOTHcounter.most_common(1)[0][0]
    except IndexError:
        print ("there are no faces")
        return -1
    return labelboth

def PredictGender(vid, start, end):

    print("Predict Gender")
    predicted_labels_hog_emo = []
    folder_path = "../modules/"
    hog_svm_file = "genderDetectionModelcrop.xml"
    hog_svm_emo = cv2.ml.SVM_load(folder_path+hog_svm_file)

    fps = vid.get(cv2.CAP_PROP_FPS)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = float(frameCount) / float(fps)  # in seconds
    else:
        print("can't render video")
        return -1
    counter =0
    while True:  # to ensure the captured frame contains a face
        counter = counter + 1
        currentFrameTime = random.randint(int(fps * start), int(fps * end))
        vid.set(1, currentFrameTime)  # "1" the property index to get the specified frame
        ret, frame = vid.read()      # ret indicates success of read process(true/false)(ex: false > end of video)
        if not ret:					  # to ensure the frame was read correctly
            print(str(frameCount) + "err1")
            continue
        face, img, x, y = facedetecion.detect(frame)  # get only the face of each frame
        if counter > 100:
            print("can't find faces in video")
            return -1
        if face is not None:  # check if either images does not contain a face go and get an other frame
            break

    hog_feature = get_HOG.getHOGfeatures128(face)
    final_label = int(hog_svm_emo.predict(hog_feature.reshape(1, -1))[1].ravel()[0])
    return final_label

def PredictGender1(vid, start, end):

    print("Predict Gender")
    predicted_labels_hog_emo = []
    # get random frames
    # get only the face of the frame
    # extract HoG Features
    # reload svm
    # predict label
    folder_path = "../modules/"
    hog_svm_file = "genderDetectionModelcrop.xml"
    hog_svm_emo = cv2.ml.SVM_load(folder_path+hog_svm_file)

    fps = vid.get(cv2.CAP_PROP_FPS)
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = end-start  # in seconds
    else:
        print("can't render video")
        return -1, -1, -1
    FramestoCapturePS = 1  # fps / 3  # number of (frames)/(frames to capture in each second)

    NumFrames = FramestoCapturePS * duration  # frames to get from the whole vid
    counter = 0
    if NumFrames > 30:
        NumFrames = 30
    print(NumFrames)
    for f in range(int(NumFrames)):
        while True:  # to ensure the captured frame contains a face
            counter = counter + 1
            currentFrameTime = random.randint(int(fps * start), int(fps * end))
            vid.set(1, currentFrameTime)  # "1" the property index to get the specified frame
            ret, frame1 = vid.read()      # ret indicates success of read process(true/false)(ex: false > end of video)
            if not ret:					  # to ensure the frame was read correctly
                print(str(frameCount) + "err1")
                continue
            if not ret:
                print("err2")
                continue
            face1, img1, x, y = facedetecion.detect(frame1)  # get only the face of each frame
            if counter == 100:
                break
            if face1 is not None:  # check if either images does not contain a face go and get an other frame
                    break
        if counter == 100:
            print("SKIP******************* Can't find a face")
            break

        hog_feature = get_HOG.getHOGfeatures128(face1)
        predicted_labels_hog_emo.append(int(hog_svm_emo.predict(hog_feature.reshape(1, -1))[1].ravel()[0]))
    # do majority voting and append respectively
    predicted_labels_hog_emocounter = collections.Counter(predicted_labels_hog_emo)
    try:
        final_label = predicted_labels_hog_emocounter.most_common(1)[0][0]
        return final_label
    except:
        print("no faces")
        return -1

"""
def predict_both(vid, start, end):
    print("Predict Both")
    predicted_labels_hog_emo = []
    predicted_labels_hof_emo = []
    predicted_labels_hog_gender = []

    # get random "pairs" of frames
    # get only the face of the frame
    # extract HoG & HOF Features
    # reload svm
    # predict label
    folder_path = "../modules/"
    hof_file = "HOFsvm1.xml"
    hog_file = 'HOGsvm1.xml'
    hog_file_gender = 'genderDetectionModel.xml'
    hog_svm_emo = cv2.ml.SVM_load(folder_path+hog_file)
    hof_svm_emo = cv2.ml.SVM_load(folder_path+hof_file)
    hog_svm_gender = cv2.ml.SVM_load(folder_path + hog_file_gender)

    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = float(frame_count) / float(fps)  # in seconds
    else:
        print("can't render video")
        return -1, -1, -1,-1
    FramestoCapturePS = 1  # fps / 3  # number of (frames)/(pair of frames) to capture in each second

    NumFrames = FramestoCapturePS * duration
    counter = 0
    if NumFrames > 30:
        NumFrames = 30
    #print(NumFrames)
    for f in range(int(NumFrames)):
        while True:  # ensuring the captured frame contains a face
            counter = counter + 1
            currentFrameTime = random.randint(int(fps * start), int(fps * end))
            vid.set(1, currentFrameTime)   # "1" the property index to get the specified frame
            ret, frame1 = vid.read()	   # ret indicates success of read process(true/false)(ex: false > end of video)
            if not ret:					   # to ensure the frame was read correctly
                print(str(frame_count)+" err1")
                continue
            vid.set(1, currentFrameTime + 1)
            ret, frame2 = vid.read()
            if not ret:
                print("err2")
                continue
            face1, img1, x, y = facedetecion.detect(frame1)  # get only the face of each frame
            face2, img2, x, y = facedetecion.detect(frame2)
            if counter == 100:
                break
            if face1 is not None:			# check if either images does not contain a face go and get an other frame
                if face2 is not None:
                    break
        if counter == 100:
            print("SKIP******************* Can't find a face")
            break
        hof_feature = get_HOF.getHOFfeatures(face1, face2)
        hog_feature = get_HOG.getHOGfeatures(face1)
        hog_feature2 = get_HOG.getHOGfeatures128(face1)
        hog_f2 = np.float32(hog_feature2)
        # print("array shape", hog_f2.shape)

        predicted_labels_hof_emo.append(int(hof_svm_emo.predict(hof_feature.reshape(1, -1))[1].ravel()[0]))
        predicted_labels_hog_emo.append(int(hog_svm_emo.predict(hog_feature.reshape(1, -1))[1].ravel()[0]))
        predicted_labels_hog_gender.append(int(hog_svm_gender.predict(hog_f2.reshape(1, -1))[1].ravel()[0]))
    #print(predicted_labels_hog_emo)
    vid.release()  # same as closing a file :release software resource & release hardware resource(ex:camera)
    # do majority voting and append respectively
    predicted_labels_hog_emo_counter = collections.Counter(predicted_labels_hog_emo)
    predicted_labels_hof_emo_counter = collections.Counter(predicted_labels_hof_emo)
    predicted_labels_hof_gender_counter = collections.Counter(predicted_labels_hof_emo)

    predicted_labels_both_counter = collections.Counter(predicted_labels_hof_emo+predicted_labels_hog_emo)

    label_hog = predicted_labels_hog_emo_counter.most_common(1)[0][0]
    label_hof = predicted_labels_hof_emo_counter.most_common(1)[0][0]
    label_both = predicted_labels_both_counter.most_common(1)[0][0]
    label_gender = predicted_labels_hof_gender_counter.most_common(1)[0][0]
    return label_hog, label_hof, label_both, label_gender
"""
