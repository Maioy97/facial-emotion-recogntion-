import cv2 as cv

def HOG(img,emoVote,genderVote,x,y):
    font = cv.FONT_HERSHEY_SIMPLEX
    if(emoVote == 0):
        cv.putText(img,"Angry",(x+50,y+50),font,0.8, (255, 255, 255), 2, cv.LINE_AA)
    elif(emoVote == 1):
        cv.putText(img,"Disgusted",(x+50,y+50),font,0.8, (255, 255, 255), 2, cv.LINE_AA)
    elif (emoVote == 2):
        cv.putText(img, "Afraid", (x + 50, y + 50), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    elif (emoVote == 3):
        cv.putText(img, "Happy", (x + 50, y + 50), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    elif (emoVote == 4):
        cv.putText(img, "Neutral", (x + 50, y + 50), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    elif (emoVote == 5):
        cv.putText(img, "Sad", (x + 50, y + 50), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    elif (emoVote == 6):
        cv.putText(img, "Surprised", (x + 50, y + 50), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    if(genderVote == 0):
        cv.putText(img,"Man",(x+80,y),font,0.8, (255, 255, 255), 2, cv.LINE_AA)
    elif(genderVote == 1):
        cv.putText(img, "Woman", (x + 80, y), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)