import cv2
import sys
import facedetecion


cascPath = 'HAAR_xmls/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
video = 'videos/short/angry male 2.mp4'
video_capture = cv2.VideoCapture(video)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    frame = facedetecion.detect(frame)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()