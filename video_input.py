import cv2
import sys
import facedetecion


cascPath = 'HAAR_xmls/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
video = '../videos/short/sad Female.mp4'
video_capture = cv2.VideoCapture(video)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if ret == 0:
        break
    frame = facedetecion.detect(frame)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()