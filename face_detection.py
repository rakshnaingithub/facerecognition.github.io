import cv2
import os
import numpy as np
import faceRecognization as fr

test_img=cv2.imread('D:/python course/facedetection/image.jpg')
faces_detected,gray_img=fr.facedetection(test_img)
print("faces_detected : ", faces_detected)

for (x,y,w,h ) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)

#resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("FaceDetection",test_img)
cv2.waitKey(0)
