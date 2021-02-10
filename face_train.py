import os 
import cv2 as cv
import numpy as np 

people =[]
for i in os.listdir(r'face'):
    people.append(i)
print(people)
DIR = r'face'

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = []
labels = []
# Croping the face from the image 
def collect_feature():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)
        if os.path.isdir(path):
            for img in os.listdir(path):
                img_path = os.path.join(path,img)
                img_array = cv.imread(img_path)
                gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
                rect_cord = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8)

                for(x,y,w,h) in rect_cord:
                    face_roi = gray[y:y+h,x:x+w]
                    features.append(face_roi)
                    labels.append(label)

collect_feature()
print("Training done")
#print("Numbers of Features", len(features))
#print("Number of Labels",len(labels))

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and the label
features = np.array(features,dtype='object')
labels = np.array(labels)
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml') # Saving the face trained model so that we can use it.
# Saving the features and the labels
np.save("features.npy",features)
np.save("labels.npy",labels)