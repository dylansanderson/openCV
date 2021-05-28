import numpy as np
import cv2
import time
"""
This program uses openCV to detect faces, smiles, and eyes. It uses haarcascades which are public domain. Haar cascades rely on
xml files which contain model training data. An xml file can be generated through training many positive and negative images. 
Try your built-in camera with 'cap = cv2.VideoCapture(0)' or use any video. cap = cv2.VideoCapture("videoNameHere.mp4")
"""

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

prev_frame_time, new_frame_time = 0,0
while 1:
    ret, img = cap.read()
    img = cv2.resize(img,(640,480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.09, 5)
    eyes = eye_cascade.detectMultiScale(gray,1.09,4)
    smiles = smile.detectMultiScale(gray,1.09,15)

    new_frame = time.time()
    fps = 1/(new_frame_time-prev_frame_time + 1)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(img,"FPS: "+str(fps),(10,450), font, 3, (0,0,0), 5, cv2.LINE_AA)

    for (x,y,w,h) in smiles:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,69,255),2)
        cv2.putText(img,"smile",(int(x-.1*x),int(y-.1*y)),font,1,(255,255,255),2)
  
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,"FACE",(int(x-.1*x),int(y-.1*y)),font,1,(255,255,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
