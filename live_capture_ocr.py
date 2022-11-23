import cv2
import os
from ocr_google import ocr_text
from trial import delta

os.chdir("E:/Sem 7/Robotics/Final_Project/Live_capture")

# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
i = 0


def ocr_location():
    is_A_plus = False

    while(cap.isOpened() and not is_A_plus):
        ret, frame = cap.read()

        # This condition prevents from infinite looping
        # incase video ends.
        if ret == False:
            break

        # Save Frame by Frame into disk using imwrite method
        #cv2.imwrite('Frame'+str(i)+'.jpg', frame)
        centroid, is_A_plus = ocr_text(frame)
        #i += 1
        if is_A_plus:
            return centroid
    return centroid


ocr_location()
cap.release()
cv2.destroyAllWindows()
