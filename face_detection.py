import cv2 as cv

img = cv.imread("test_images/4.png")
cv.imshow("Original",img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

rect_cord = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8)
print("number of face found",len(rect_cord))

# Drawing the rectange on the original image 
for (x,y,w,h) in rect_cord:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
cv.imshow("Face Detected",img)
cv.waitKey(0)