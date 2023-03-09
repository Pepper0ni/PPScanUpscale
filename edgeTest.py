from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import math
from skimage.filters import unsharp_mask
from skimage.util import img_as_ubyte

lowT = 3134
hiT = 11308
sobel = 3
space = 7
blur = 3

window_detection_name = 'Object Detection'

def lowTTrackbar(val):
    global lowT
    lowT = val

def hiTTrackbar(val):
    global hiT
    hiT = val

def sobelTrackbar(val):
    global sobel
    sobel = val
    
def spaceTrackbar(val):
    global space
    space = val

def blurTrackbar(val):
    global blur
    blur = val


def trimImage(img, fromTop, newBot, fromLeft, newRight):
    return img[fromTop:newBot, fromLeft:newRight]



parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument("-i", "--Input", help="Set Input")

args = parser.parse_args()
image = cv.imread(cv.samples.findFile(args.Input))
size = (round(image.shape[1]/2), round(image.shape[0]/2))
#image = cv.resize(image, size, cv.INTER_AREA)
H, S, V = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
S = S.astype(float) * 4
S = S - 40
S[S > 255] = 255
S[S < 0] = 0
clean = cv.cvtColor(cv.merge([H, S.astype(np.uint8), V]), cv.COLOR_HSV2BGR)

cv.namedWindow(window_detection_name, cv.WINDOW_NORMAL)
cv.namedWindow("bilin", cv.WINDOW_NORMAL)
cv.createTrackbar("lowT", window_detection_name, lowT, 10000, lowTTrackbar)
cv.createTrackbar("hiT", window_detection_name, hiT, 20000, hiTTrackbar)
cv.createTrackbar("sobel", window_detection_name, sobel, 5, sobelTrackbar)
cv.createTrackbar("space", window_detection_name, space, 30, spaceTrackbar)
cv.createTrackbar("blur", window_detection_name, blur, 30, blurTrackbar)

while True:
    biblur = cv.bilateralFilter(clean, space, blur, space)
    # biblur = cv.bilateralFilter(biblur, space, blur, space)
    # biblur = cv.bilateralFilter(biblur, space, blur, space)
    # biblur = cv.bilateralFilter(biblur, space, blur, space)
    #hsvImage = cv.cvtColor(biblur, cv.COLOR_BGR2HSV_FULL)
    if sobel == 0:
        sobel = 1
    edges = cv.Canny(biblur, lowT, hiT, False, sobel * 2 + 1)
    cv.imshow(window_detection_name, edges)
    cv.imshow("bilin", biblur)
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break