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

def Train(file, target_dir):
    print ("here")
    labels = []
    hogDescriptors = []
    hofDescriptors = []
	
	#fill features and labels
    with open(file) as f:
        next(f)
        for l in f:
            l = l.strip()
            if len(l) > 0:
                link, start, end, video, utterance, arousal, valence, Emotion = l.split(',')[:8]
                video_dir = os.path.join(os.path.join(target_dir, video))
                current_video_path = os.path.abspath(os.path.join(video_dir, utterance))
                
                if os.path.exists(current_video_path):
                    # read vid
                    # get random "pairs" of frames
					# get only the face of the frame
					# extract HoG & HOF Features 
					# Append label
                    vid = cv2.VideoCapture(current_video_path)
                    fps = vid.get(cv2.CAP_PROP_FPS)
                    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = float(num_frames) / float(fps) # in seconds
                    FramestoCapturePS = fps/3 #number of (frames)/(pair of frames) to capture in each second
                    
                    NumFrames = FramestoCapturePS * duration
                    
                    for f in range(NumFrames):
                    	labels.append(Emotion)
						
                        currentFrameTime = random.randint(0,frameCount-1)
						
                        vid.set(1, currentFrameTime)
                        ret, frame1 = vid.read()
                        vid.set(1, currentFrameTime+1)
                        ret, frame2 = vid.read()
						
						face1 ,img1 = detect(frame1)
						face2 ,img2 = detect(frame2)
						
                        hoffeature = getHOFfeatures(face1,face2)
                        hofDescriptors.append(hoffeature)
                        hogfeature = getHOGfeatures(face1)
                        hogDescriptors.append(hogfeature)
	#train with HOG features
	
	#train with HOG features
	HOGsvm = cv2.ml.SVM_create()
	HOGsvm.setType(cv2.ml.SVM_C_SVC)
	HOGsvm.setKernel(cv2.ml.SVM_RBF)
	
	HOGsvm.train(np.array(hogDescriptors), cv2.ml.ROW_SAMPLE, np.array(labels))
	HOGsvm.save('HOGsvm.xml')

	#train with HOF features
	HOFsvm = cv2.ml.SVM_create()
	HOFsvm.setType(cv2.ml.SVM_C_SVC)
	HOFsvm.setKernel(cv2.ml.SVM_RBF)

	HOFsvm.train(np.array(hofDescriptors), cv2.ml.ROW_SAMPLE, np.array(labels))
	HOFsvm.save('HOFsvm.xml')
