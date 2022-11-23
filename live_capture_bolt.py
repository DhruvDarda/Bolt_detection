import cv2
import os
from trial import delta
import numpy as np

os.chdir("E:/Sem 7/Robotics/Final_Project/Live_capture")

# Opens the inbuilt camera of laptop to capture video.

cap = cv2.VideoCapture(0)


def bolt_location():
    while(cap.isOpened()):
        ret, frame = cap.read()

        # This condition prevents from infinite looping
        # incase video ends.
        if ret == False:
            break

        # Save Frame by Frame into disk using imwrite method
        #cv2.imwrite('Frame'+str(i)+'.jpg', frame)
        delta_coordinates, im2 = delta(frame)
        # print(delta_coordinates)
        if np.isnan(delta_coordinates).any():
            break
        cv2.imshow('frame', im2)
        if cv2.waitKey(1) == ord('q'):
            break
    return delta_coordinates


print(bolt_location())
cap.release()
cv2.destroyAllWindows()
