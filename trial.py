import cv2
import os
import numpy as np
from math import abs

original_horizontal_distance = 15
original_vertical_distance = 12


def output(result):
    global image
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    centers = []

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

    return centers


os.chdir("E:/Sem 7/Robotics/Final_Project/")

# Read image
image = cv2.imread("E:/Sem 7/Robotics/Final_Project/test.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Threshold of blue in HSV space
lower_blue = np.array([60, 35, 140])
upper_blue = np.array([180, 255, 255])

lower_red = np.array([160, 50, 50])
upper_red = np.array([180, 255, 255])

# preparing the mask to overlay
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
red_mask = cv2.inRange(hsv, lower_red, upper_red)


coordinates = [0, 0]
#result = cv2.bitwise_and(image, image, mask=blue_mask)
result = image
centers = output(result)
centroid_image = result.shape
centroid_image = (centroid_image[1]//2, centroid_image[0]//2)
t = 0

centers = centers[1:]
print(centers)
while len(centers) > 4 and t < 15:
    vertical_dist_lst = [np.abs(centers[0][1] - centers[3][1]),
                         np.abs(centers[1][1] - centers[4][1])]

    horizontal_dist_lst = [np.abs(centers[0][0] - centers[1][0]),
                           np.abs(centers[3][0] - centers[4][0])]

    horizontal_dist = np.mean(horizontal_dist_lst)
    vertical_dist = np.mean(vertical_dist_lst)

    coordinates[0] = (
        original_horizontal_distance/horizontal_dist) * (centers[2][1]-centroid_image[1])
    coordinates[1] = (
        original_vertical_distance/vertical_dist) * (centers[2][0]-centroid_image[0])
    t += 1


cv2.imshow("Final Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(coordinates)

centers = output(image)
coordinates[0] = centers[0][0] - centroid_image[0]
while(abs(coordinates[0]) < 0.5 and abs(coordinates[1]) < 0.5):
    coordinates[0] = centers[0][0] - centroid_image[0]
