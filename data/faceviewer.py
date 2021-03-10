import cv2 as cv
import os

img = cv.imread('myFaces/RawImages/IMG_06.jpg')
# cv.imshow('faceImage', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

myDir = os.getcwd()
haar_face = os.path.join(myDir, 'haar_face.xml')
haar_cascade = cv.CascadeClassifier(haar_face)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

for(x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

x,y,w,h = faces_rect[0]
crop = gray[y:y + h, x:x + w]

# x1, y1, w1, h1 = 0,0,0,0
# for(x,y,w,h) in faces_rect:
#     if w < 500 or h < 500:
#         continue
#     cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
#     x1, y1, w1, h1 = x, y, w ,h


# crop = gray[y1:y1 + h1, x1:x1 + w1]


crop = cv.resize(crop, (160, 160))
img = cv. resize(img, (500, 500))
cv.imshow('Cropped Face', crop)
cv.imshow('Detected Faces', img)
cv.waitKey(0)