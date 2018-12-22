import cv2 as cv
import facedetecion 


def write_labels(vid, emo_vote, gender_vote, start, end):
    print("write_labels")
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*' DIVX')
    out = cv.VideoWriter("../output/"+'output.mp4', fourcc, 20.0, (640, 480))

    font = cv.FONT_HERSHEY_SIMPLEX
    fps = vid.get(cv.CAP_PROP_FPS)
    frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    label_string = ""
    if emo_vote == 0:
        label_string = "Angry"
    elif emo_vote == 1:
        label_string = "Disgusted"
    elif emo_vote == 2:
        label_string = "Afraid"
    elif emo_vote == 3:
        label_string = "Happy"
    elif emo_vote == 4:
        label_string = "Neutral"
    elif emo_vote == 5:
        label_string = "Sad"
    elif emo_vote == 6:
        label_string = "Surprised"

    if gender_vote == 0:
        label_string += " Man"
    elif gender_vote == 1:
        label_string += " Woman"

    # loop over frames and write labels for each frame
    for f in range(start_frame, end_frame):
        vid.set(1, f)            # "1" the property index to get the specified frame
        ret, frame = vid.read()  # ret indicates success of read process(true/false)(ex: false > end of video)
        if not ret:              # to ensure the frame was read correctly
            print(str(frame_count) + " err1")
            continue
        face, img, x, y = facedetecion.detect(frame)  # get only the face of each frame
        # writing labels
        n_frame = cv.putText(img, label_string, (x - 50, y - 50), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)
        out.write(n_frame)

