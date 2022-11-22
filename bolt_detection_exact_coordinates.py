import cv2
import os
import numpy as np

os.chdir("E:/Sem 7/Robotics/Final_Project/")

# Read image
image = cv2.imread("E:/Sem 7/Robotics/Final_Project/forlines.jpg")

# cv2.imshow('window name', image)
# cv2.waitKey(0)

# closing all open windows
# cv2.destroyAllWindows()
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
shape = gray.shape
shape = [shape[0]//2, shape[1]//2]

# Use canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply HoughLinesP method to
# to directly obtain line end points
lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi/180,  # Angle resolution in radians
    threshold=100,  # Min number of votes for valid line
    minLineLength=50,  # Min allowed length of line
    maxLineGap=10  # Max allowed gap between line for joining them
)

# Iterate over points
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])

# Save the result image
cv2.imwrite('detectedLines.png', image)


###################################################################################################################################
centers = []
gray = cv2.medianBlur(gray, 7)

rows = gray.shape[0]
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                           param1=100, param2=30,
                           minRadius=1, maxRadius=50)
'''
ret, thresh = cv2.threshold(gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
'''

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        centers.append(center)
        # circle center
        cv2.circle(image, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(image, center, radius, (255, 0, 255), 3)

'''
for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)

    # calculate x,y coordinate of center
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # circle center
    cv2.circle(image, center, 1, (0, 100, 100), 3)
    # circle outline
    cv2.circle(image, center, 5, (255, 255, 255), -1)
'''

cv2.imshow("detected circles", image)
cv2.waitKey(0)
print(centers)
