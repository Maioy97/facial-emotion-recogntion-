import sys
import facedetecion
import cv2
import predict
import labeldetection
import tkinter as tk
from tkinter import filedialog

video_path = '../videos/short/'
root = tk.Tk()
root.withdraw()
print("start")
file_path = filedialog.askopenfilename()  # to get video path
video_capture = cv2.VideoCapture(file_path)
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
#set start and end for function predict

fps = video_capture.get(cv2.CAP_PROP_FPS)
frameCount = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

if fps > 0:
    duration = int(float(frameCount) / float(fps))  # in seconds

gender_vote=predict.PredictGender(video_capture, begin, end)

for i in range(duration):
    if duration - i < 10:
        # if the remaining part of the video is shorter than 10, predict labels in the range i and the video end
        begin = i
        end = duration

    else:
        # if it's longer, predict labels in a 10 second range that starts at i,range = 10 ,skip the labeled 10 secs
        begin = i
        end = i + 10
        i += 10

    if hogVote>-1:
        print("prediction done ")
        if Vote == "HOG":
            hogVote = predict.predictEmoHOG(video_capture, begin, end)
            labeldetection.write_labels(video_capture, hogVote, gender_vote, begin, end)
        elif Vote == "HOF":
            hofVote = predict.predictEmoHOF(video_capture, begin, end)
            labeldetection.write_labels(video_capture, hofVote, gender_vote, begin, end)
        elif Vote == "BOTH":
            bothVote = predict.PredictEmoBoth(video_capture, begin, end)
            labeldetection.write_labels(video_capture, bothVote, gender_vote, begin, end)
    
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
