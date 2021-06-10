import cv2 as cv
import numpy as np

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
    h, w = frame.shape[:2]
    # frame = cv.resize(frame,(640,int(640*h/w)))
    
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    detector.setInput(blob)
    faces = detector.forward()

    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]

        if confidence < conf_t:
            continue

        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h]) 
        x1, y1, x2, y2 = box.astype("int")
        
        frame = draw(frame,x1,x2,y1,y2,x2-x1)

    return frame

# Video

#"""
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

"""
input_image = cv.imread("image-test/face1.jpg")
output = face_detection(input_image)
cv.imshow('Face detector', output)
cv.waitKey(0)
cv.destroyAllWindows()
"""