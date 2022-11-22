import cv2
import os
import numpy as np

os.chdir("E:/Sem 7/Robotics/Final_Project/")

# Read image
image = cv2.imread("E:/Sem 7/Robotics/Final_Project/forlines.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
#cv2.imshow("thresh", thresh)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
c = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = i
            image = cv2.drawContours(
                image, contours, c, (0, 255, 0), 3)
    c += 1

mask = np.zeros((gray.shape), np.uint8)
cv2.drawContours(mask, [best_cnt], 0, 255, -1)
cv2.drawContours(mask, [best_cnt], 0, 0, 2)
#cv2.imshow("mask", mask)

out = np.zeros_like(gray)
out[mask == 255] = gray[mask == 255]
#cv2.imshow("New image", out)

blur = cv2.GaussianBlur(out, (5, 5), 0)
#cv2.imshow("blur1", blur)

thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
#cv2.imshow("thresh1", thresh)

cv2.imwrite('maskedimage.png', thresh)


edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

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
    cv2.line(thresh, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])

# Save the result image
cv2.imwrite('detectedLines.png', thresh)

centers = []
gray = cv2.medianBlur(thresh, 7)

rows = gray.shape[0]
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                           param1=100, param2=30,
                           minRadius=1, maxRadius=50)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        centers.append(center)
        # circle center
        cv2.circle(thresh, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(thresh, center, radius, (255, 0, 255), 3)

cv2.imshow("Final Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(centers)

for i in lines_list:
    # Horizontal Lines
    if i[0][0] - i[1][0] < 5:
