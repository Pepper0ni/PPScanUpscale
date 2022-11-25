from __future__ import print_function
import cv2 as cv
import argparse

import numpy as np

max_value = 255
max_value_H = 360 // 2
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def trimImage(img, fromTop, newBot, fromLeft, newRight):
    return img[fromTop:newBot, fromLeft:newRight]


parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument("-i", "--Input", help="Set Input")
parser.add_argument("-lh", "--LowHue", help="Set Input")
parser.add_argument("-hh", "--HighHue", help="Set Input")
parser.add_argument("-ls", "--LowSat", help="Set Input")
parser.add_argument("-hs", "--HighSat", help="Set Input")
parser.add_argument("-lv", "--LowVal", help="Set Input")
parser.add_argument("-hv", "--HighVal", help="Set Input")

args = parser.parse_args()
image = cv.imread(cv.samples.findFile(args.Input))
size = (round(image.shape[1]/2), round(image.shape[0]/2))
image = cv.resize(image, size, cv.INTER_AREA)
hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)


low_H = int(args.LowHue or 0) or 0
low_S = int(args.LowSat or 0) or 0
low_V = int(args.LowVal or 0) or 0
high_H = int(args.HighHue or max_value_H) or max_value_H
high_S = int(args.HighSat or max_value) or max_value
high_V = int(args.HighVal or max_value) or max_value

cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
halfImage = trimImage(hsvImage, round(hsvImage.shape[0]*0.65), hsvImage.shape[0]-15, 15, hsvImage.shape[1]-15)
print(np.percentile(halfImage, 2.25, (0, 1)))
print(np.percentile(halfImage, 55, (0, 1)))

while True:

    frame_threshold = cv.inRange(hsvImage, (low_H, low_S, low_V), (high_H, high_S, high_V))
    median = cv.medianBlur(frame_threshold, 3)
    #frame_threshold = np.bitwise_and(cv.cvtColor(frame_threshold, cv.COLOR_GRAY2BGR), image)
    cv.imshow(window_detection_name, frame_threshold)
    cv.imshow("median", median)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break