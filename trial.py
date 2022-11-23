import cv2
import os
import numpy as np

original_horizontal_distance = 15
original_vertical_distance = 12
maping_const = 1


def output(result, image):
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    centers = []
    '''
    ret, thresh = cv2.threshold(gray, 127, 255, 0)

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        centers.append((cX, cY))
    '''
    detected_circles = cv2.HoughCircles(gray,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=100,
                                        param2=30, minRadius=1, maxRadius=100)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(image, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (a, b), 1, (0, 0, 255), 3)

            centers.append((a, b))

    return centers, image


os.chdir("E:/Sem 7/Robotics/Final_Project/")

# Read image
image = cv2.imread("E:/Sem 7/Robotics/Final_Project/test1.jpg")


def delta(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold of blue in HSV space
    lower = np.array([70, 0, 5])

    upper = np.array([230, 40, 50])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower, upper)

    #result = cv2.bitwise_and(image, image, mask=mask)
    result = image.copy()
    #cv2.imshow('filter', result)
    centers, im2 = output(result, image)
    centroid_image = image.shape
    centroid_image = (centroid_image[1]//2, centroid_image[0]//2)
    print(centers)

    #tool_location = min(centers, key=lambda x: x[1])
    #bolt_location = max(centers, key=lambda x: x[1])
    #bolt_1_location = min(centers, key=lambda x: x[0])
    #bolt_2_location = max(centers, key=lambda x: x[0])
    if not centers:
        return np.array([np.NAN, np.NAN]), im2
    coordinates = maping_const * \
        (np.array(centers[0]) - np.array(centroid_image))
    return coordinates, im2


'''
cv2.imshow("Final Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
