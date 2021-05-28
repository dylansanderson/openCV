import numpy as np
import cv2
import time

"""__
This program uses Haar-Cascades to detect cars. Uses Video6.mp4 as sample input. 
Detection is based off data in an XML file.

"""


W,H = 1280,720
car_cascade = cv2.CascadeClassifier('cars.xml')
cap = cv2.VideoCapture('cars.mp4')
prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while 1:
    ret, img = cap.read()
    img = cv2.resize(img,(W,H))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.04, 7)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(img, "FPS: " + str(fps), (10,int(H-.1*H)), font, 3, (0, 0, 0), 5, cv2.LINE_AA)
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,69,255),2)
        cv2.putText(img,"car",(int(x-.1*x),int(y-.1*y)),font,1,(255,255,255),2)
 
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()