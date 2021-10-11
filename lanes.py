import cv2
#import sys
import numpy as np
#numpy.set_printoptions(threshold=sys.maxsize)
#import matplotlib.pyplot as plt


def canny(image_of_interest):
    grey = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 150)
    return canny_image


def region_of_interest(image_of_interest):
    polygon = np.array([[(0, 700), (500, 375), (960, 1000)]])
    mask = np.zeros_like(image_of_interest)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image_of_interest, mask)
    return masked_image


def display_lines(image_of_interest, lines):
    line_image = np.zeros_like(image_of_interest)
    if lines is not None:
        for line in lines:
            #print(line)
            x1, y1, x2, y2 = line.reshape(4)
            #print(x1, y1, x2, y2)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average((left_fit), axis=0 )
        right_fit_average = np.average((right_fit), axis=0)
        #left_line = np.min(left_fit)
        left_line = make_coordinates(image, left_fit_average)
        print("Left line: ", left_line)
        right_line = make_coordinates(image, right_fit_average)
        print(right_line)
        return np.array([left_line, right_line])


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = 715
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])



#image = cv2.imread("IMG_2095.jpg")
#image = cv2.resize(image, (960, 1280))
#lane_image = np.copy(image)
#canny_image = canny(lane_image)
#roi_image = region_of_interest(canny_image)
#cropped_image = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 100)
#print(lines)
#averaged_lines = average_slope_intercept(lane_image, lines)
#print(averaged_lines)
#line_image = display_lines(lane_image, averaged_lines)
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

#print(image.shape)
#image = cv2.resize(image, (854, 1281))
#print(image.shape)
#cv2.imshow("output", combo_image)
#cv2.waitKey(0)
#image = cv2.resize(image, 1280, 960)



cap = cv2.VideoCapture("IMG_2094.mp4")


while(cap.isOpened()):
    i, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (960, 1280))
        lane_image = np.copy(frame)
        canny_image = canny(lane_image)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=100)
        line_image = display_lines(lane_image, lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        cv2.imshow("output", combo_image)
        cv2.waitKey(1)
