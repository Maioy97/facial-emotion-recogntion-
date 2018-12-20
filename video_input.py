import cv2
import sys
import facedetecion
import predict
import labeldetection
import tkinter as tk
from tkinter import filedialog

videopath = '../videos/short/'
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()#to get video path 
videoType=input("videoType")
if videoType=='vid':
    video_capture = cv2.VideoCapture(file_path)
elif videoType=='cam':
    video_capture = cv2.VideoCapture(0)

Vote = input("HOG or HOF or BOTH")

#set start and end for function predict

fps = video_capture.get(cv2.CAP_PROP_FPS)
frameCount = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

if fps > 0:
    duration = float(frameCount) / float(fps)  # in seconds

for i in range(duration):
    if duration - i < 10:
        hogVote, hofVote, bothVote = predict.PredictEmo(video_capture,i,duration)
        x = duration - i
    else:
        end = i + 10
        hogVote, hofVote, bothVote, x, y = predict.PredictEmo(video_capture,i,end)
        genderVote = predict.PredictGender(video_capture,i,end)
        x = 10
        i += 10
    for j in range(x):
        video_capture.set(1, j)
        ret, frame = video_capture.read()
        if Vote == "HOG":
            labeldetection.HOG(frame,hogVote,genderVote,x,y)
        elif Vote == "HOF":
            labeldetection.HOG(frame, hofVote, genderVote, x, y)
        elif Vote == "BOTH":
            labeldetection.HOG(frame, bothVote, genderVote, x, y)





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
