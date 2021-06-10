import cv2 as cv
import numpy as np
from retinaface import RetinaFace

path = "video-test/face1.mp4"
cap = cv.VideoCapture(0)

detector = cv.dnn.readNetFromCaffe("model/deploy.prototxt" , 
                                   "model/res10_300x300_ssd_iter_140000.caffemodel")
conf_t = 0.5

def draw(frame,x1,x2,y1,y2,width):
    len = int(width/4)
    t = 2
    if width < 100: t = 1

    cv.line(frame, (x1, y1), (x1, y1+len), (252, 222, 165), t)
    cv.line(frame, (x1, y1), (x1+len, y1), (252, 222, 165), t)

    cv.line(frame, (x2, y1), (x2-len, y1), (252, 222, 165), t)
    cv.line(frame, (x2, y1), (x2, y1+len), (252, 222, 165), t)

    cv.line(frame, (x1, y2), (x1, y2-len), (252, 222, 165), t)
    cv.line(frame, (x1, y2), (x1+len, y2), (252, 222, 165), t)

    cv.line(frame, (x2, y2), (x2-len, y2), (252, 222, 165), t)
    cv.line(frame, (x2, y2), (x2, y2-len), (252, 222, 165), t)

    return frame

def face_detection(frame):
    # h, w = frame.shape[:2]
    # frame = cv.resize(frame,(640,int(640*h/w)))
    
    faces = RetinaFace.detect_faces(frame)

    for i in faces.keys():
        identity = faces[i]
        facial_area = identity["facial_area"]
        x1, y1, x2, y2 = facial_area

        frame = draw(frame,x1,x2,y1,y2,x2-x1)
        # frame = cv.rectangle(frame, (x1, y1), (x2, y2), (252, 222, 165), 1)

    return frame

# Video

"""
while True:
    ret, frame = cap.read()    

    frame = cv.resize(frame,(640,360))
    frame = face_detection(frame)
    cv.imshow('Face detector', frame)
 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()
#"""

# Image

#"""
input_image = cv.imread("image-test/face1.jpg")
output = face_detection(input_image)
cv.imshow('Face detector', output)
cv.waitKey(0)
cv.destroyAllWindows()
#"""