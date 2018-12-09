import cv2
import sys
import facedetecion

videopath = '../videos/short/'
videoType=input("videoType")
if videoType=='cam':
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(videopath+videoType)

while True:
    # Capture frame-by-frame
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