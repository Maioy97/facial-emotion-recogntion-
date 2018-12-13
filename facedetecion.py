import cv2 as cv


def detect(img):
    face_cascade = cv.CascadeClassifier('HAAR_xmls/haarcascade_frontalface_alt.xml')
    if (len(img.shape) < 3):
        gray = img
    else:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    roi_color=None
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    return roi_color ,img
