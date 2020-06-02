# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import face_recognition

# Get a reference to webcam 
video_capture = cv2.VideoCapture(0)

#Initialise the dlib's face detector (HOG-based) and then create
detector = dlib.get_frontal_face_detector()
#The facial landmark predictor
predictor = dlib.shape_predictor('path of the predictor file') # uploaded along with the source code
#Initialse face location list to capture the locations of the face being detected
face_locations = []

#Hit keyboard key 'q' to quit
while(True):
    ret, frame = video_capture.read()
    #cv2.imshow('Video', frame)
    #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #rgb_frame = frame[:, :, ::-1]
    rgb_frame = frame[:, :, [2, 1, 0]]
    #cv2.waitKey(1)
    #Finding all faces in the current frame of the video
    #face_locations = face_recognition.face_locations(rgb_frame)
    #print(face_locations)
    rects = detector(rgb_frame, 1) #To detect get the rectangle enclosed over a face
    #print(enumerate(rects))

    for (i,rect) in enumerate(rects):
        #print(i,rect)
        shape = predictor(rgb_frame, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        #print('x , y, w, h is ',x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 50), 2)
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow('Video', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;
      


video_capture.release()
cv2.destroyAllWindows()
        
