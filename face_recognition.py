from os import confstr
import cv2 as cv 
import numpy as np 

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread("test_images/2.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Person",gray)

# Detect faces

face_rect = haar_cascade.detectMultiScale(gray,1.1,8)
for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h,x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    print("Name of the person in the image", label," with confidence of",confidence)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv.imshow("Detected image", img)

cv.waitKey(0)