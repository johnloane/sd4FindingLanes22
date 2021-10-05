import cv2
#import sys
import numpy as np
#numpy.set_printoptions(threshold=sys.maxsize)

image = cv2.imread("IMG_2095.jpg")
image = cv2.resize(image, (1280, 960))
lane_image = np.copy(image)
grey = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(grey, (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)
#print(image.shape)
#image = cv2.resize(image, (854, 1281))
#print(image.shape)
cv2.imshow("output", canny)
cv2.waitKey(0)
#image = cv2.resize(image, 1280, 960)
#

