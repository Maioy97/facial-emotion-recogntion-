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
import datetime
import argparse
import collections

def test(file, target_dir):
    print("here")
    labelsHog = []
    labelsHof = []
    labelsboth = []
    labelsReal = []
    HOGsvm = cv2.ml.SVM_load('HOGsvm.xml')
    HOFsvm = cv2.ml.SVM_load('HOFsvm.xml')
    print("doneloading")

    # fill features and labels
    with open(file) as f:
        next(f)
        for l in f:
            l = l.strip()
            if len(l) > 0:
                link, start, end, video, utterance, arousal, valence, Emotion = l.split(',')[:8]
                video_dir = os.path.join(os.path.join(target_dir, video))
                current_video_path = os.path.abspath(os.path.join(video_dir, utterance))

                if os.path.exists(current_video_path):
                    print(str(current_video_path))
                    vid = cv2.VideoCapture(current_video_path)
                    fps = vid.get(cv2.CAP_PROP_FPS)
                    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                    if fps ==0:
                        vid.release()#can't render video
                        continue
                    else:
                        duration = float(frameCount) / float(fps) # in seconds
                    FramestoCapturePS = 1  #fps / 3  # number of (frames)/(pair of frames) to capture in each second

                    NumFrames = FramestoCapturePS * duration
                    counter=0
                    #if NumFrames > 15 :
                    #    NumFrames = 15
                    #print(NumFrames)
                    predictedlabelsHOG = []
                    predictedlabelsHOF = []
                    for f in range(int(NumFrames)):
                        while True:# to ensure the captured frame contains a face
                            counter = counter + 1
                            currentFrameTime = random.randint(0, frameCount - 2)
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
                            print("SKIP******************* "+str(current_video_path))
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
                    #print(predictedlabelsHOGcounter.most_common(1)[0][0])
                    #print(predictedlabelsHOFcounter.most_common(1)[0][0])
                    #print(predictedlabelsBOTHcounter.most_common(1)[0][0])
                    try:
                        labelsHog.append(predictedlabelsHOGcounter.most_common(1)[0][0])
                        labelsHof.append(predictedlabelsHOFcounter.most_common(1)[0][0])
                        labelsboth.append(predictedlabelsBOTHcounter.most_common(1)[0][0])
                        labelsReal.append(int(Emotion))
                    except IndexError:
                        print(IndexError)
                        continue
    a = np.asarray([labelsReal, labelsHog, labelsHof, labelsboth])
    a = a.transpose()
    np.savetxt("results.csv", a, delimiter=",", fmt='%10.5f')

    #use zip to compare predicted vs real labels to calculate the accuracy
    #a = [5,1,1,2,1]  b = [0,1,1,2,6] zip(a,b)=[(5, 0), (1, 1), (1, 1), (2, 2), (1, 6)]
    HogAccuracy = sum(1 for x,y in zip(labelsReal,labelsHog) if x == y) / float(len(labelsReal))
    print("Hog Accuracy: "+str(HogAccuracy))
    	  
    HOFAccuracy = sum(1 for x,y in zip(labelsReal,labelsHof) if x == y) / float(len(labelsReal))
    print("HOF Accuracy: "+str(HOFAccuracy))
    
    BothAccuracy = sum(1 for x,y in zip(labelsReal,labelsboth) if x == y) / float(len(labelsReal))
    print("Hog & HOF Accuracy: "+str(BothAccuracy))

if __name__ == "__main__":# to start using cmd and pass parameters 
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file")
    parser.add_argument("--video-dir")

    opt = parser.parse_args()
    if not os.path.exists(opt.split_file):
        print("Cannot find split file")
        sys.exit(-1)
    if not os.path.exists(opt.video_dir):
        print("Cannot find video Directory")
        sys.exit(-1)
    print(opt.split_file)
    test(opt.split_file, opt.video_dir)
