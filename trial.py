import cv2
import numpy as np

image = cv2.imread("E:/Sem 7/Robotics/Final_Project/forlines.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

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
            image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
    c += 1

mask = np.zeros((gray.shape), np.uint8)
cv2.drawContours(mask, [best_cnt], 0, 255, -1)
cv2.drawContours(mask, [best_cnt], 0, 0, 2)

out = np.zeros_like(gray)
out[mask == 255] = gray[mask == 255]

blur = cv2.GaussianBlur(out, (5, 5), 0)

thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))

c = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 1000/2:
        cv2.drawContours(out, contours, c, (0, 255, 0), 3)
    c += 1


cv2.imshow("Final Image", out)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
        Parameters:
            test_video: Input video.
            output_video: A video file with detected lane lines.
    """
    input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(os.path.join('output_videos', output_video), audio=False)
'''
