import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?) Exiting...")
            break

        flip = cv.flip(frame, 1)
        cv.imshow('frame', flip)
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()