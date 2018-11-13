import cv2 as cv


def detect(img):
    face_cascade = cv.CascadeClassifier('HAAR_xmls/haarcascade_frontalface_alt.xml')
    '''righteye_cascade = cv.CascadeClassifier('HAAR_xmls/haarcascade_righteye_2splits.xml')
    #lefteye_cascade = cv.CascadeClassifier('HAAR_xmls/haarcascade_lefteye_2splits.xml')
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        '''roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        righteyes = righteye_cascade.detectMultiScale(roi_gray)
        lefteyes = lefteye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in righteyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        for (lex, ley, lew, leh) in lefteyes:
            cv.rectangle(roi_color, (ex, ey), (lex + lew, ley + leh), (0, 255, 0), 2)'''
    '''#cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''
    return img
'''plt.imshow(img)
plt.show()#print(x)
'''