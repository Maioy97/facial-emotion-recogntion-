import cv2
import sys
import facedetecion
import predict
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

#hogVote, hofVote, bothVote = predict.PredictEmo(video_capture)

#Milestone 1 read video & detect faces
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

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
