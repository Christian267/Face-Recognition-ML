import cv2 as cv
def detectFace():
    cap = cv.VideoCapture(0)
    haar_cascade = cv.CascadeClassifier('data/haar_face.xml')
    while True:
        ret, frame = cap.read()

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        
        faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=2)

        for(x,y,w,h) in faces_rect:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)


        cv.imshow('Capture - Face detection', frame)
        if cv.waitKey(10) == 32:    # use 'space' to capture current frame
            return frame


    