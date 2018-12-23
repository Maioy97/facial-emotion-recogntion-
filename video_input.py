import sys
import facedetecion
import cv2
import predict
import labeldetection
import tkinter as tk
from tkinter import filedialog

video_path = '../videos/short/'
#root = tk.Tk()
#root.withdraw()
#print("start")
'''video_capture = cv2.VideoCapture(file_path)
videoType = input("videoType: ")

if videoType == 'vid':
    file_path = filedialog.askopenfilename()  # to get video path
    video_capture = cv2.VideoCapture(file_path)
elif videoType == 'cam':
    video_capture = cv2.VideoCapture(0)
'''
Vote = input("HOG or HOF or BOTH: ")
Vote = Vote.upper()
folder_path = "../modules/"
hog_file = 'HOGsvmRBF.xml'
hof_file = "HOFsvmLinear.xml"

if Vote == "HOG":
    hog_svm_emo = cv2.ml.SVM_load(folder_path + hog_file)
elif Vote == "HOF":
    hof_svm_emo = cv2.ml.SVM_load(folder_path + hof_file)
elif Vote == "BOTH":
    hog_svm_emo = cv2.ml.SVM_load(folder_path + hog_file)
    hof_svm_emo = cv2.ml.SVM_load(folder_path + hof_file)
print("done loading")
y="y"
#set start and end for function predict
while(y=="y"):
    file_path = filedialog.askopenfilename()  # to get video path
    video_capture = cv2.VideoCapture(file_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frameCount = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        duration = int(float(frameCount) / float(fps))  # in seconds
    else:
        print("Can't render video")

    gender_vote=predict.PredictGender1(video_capture, 0, duration)

    i=0
    while i < duration:

        if duration - i < 10:
            # if the remaining part of the video is shorter than 10, predict labels in the range i and the video end
            begin = i
            end = duration
            i =i+ 10
        else:
            # if it's longer, predict labels in a 10 second range that starts at i,range = 10 ,skip the labeled 10 secs
            begin = i
            end = i + 10
            i =i+ 10
        if Vote == "HOG":
            hogVote = predict.predictEmoHOG(video_capture, begin, end, hog_svm_emo)
            labeldetection.write_labels(video_capture, hogVote, gender_vote, begin, end)
        elif Vote == "HOF":
            hofVote = predict.predictEmoHOF(video_capture, begin, end, hof_svm_emo)
            labeldetection.write_labels(video_capture, hofVote, gender_vote, begin, end)
        elif Vote == "BOTH":
            bothVote = predict.PredictEmoBoth(video_capture, begin, end,hog_svm_emo,hof_svm_emo)
            labeldetection.write_labels(video_capture, bothVote, gender_vote, begin, end)
        print("prediction done ")
        print(i)

    #hogVote, hofVote, bothVote = predict.PredictEmo(video_capture)
    #print("hog "+str(hogVote)+" hof "+ str(hofVote)+" Both " +str(bothVote))
    ''''#Milestone 1 read video & detect faces
    while True:
        # Capture frame-by-frame
        #video_capture.set(1,50.0)
        ret, frame = video_capture.read()
        if ret == 0:
            break
        height, width, channels = frame.shape
        if width>1000:
            frame = cv2.resize(frame, (int(width/3), int(height/3)))
        face, frame = facedetecion.detect(frame)
        # Display the resulting frame
        cv2.imshow('Video', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    '''
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    y=input("to continue enter y").lower()