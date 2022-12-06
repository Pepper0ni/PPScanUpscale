from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import math
from skimage.filters import unsharp_mask
from skimage.util import img_as_ubyte

blur = 3
sigma = 141
iter = 1
addX = 20
addY = 20

window_detection_name = 'Object Detection'

def blurTrackbar(val):
    global blur
    blur = val

def sigmaTrackbar(val):
    global sigma
    sigma = val

def iterTrackbar(val):
    global iter
    iter = val

def XTrackbar(val):
    global addX
    addX = val

def YTrackbar(val):
    global addY
    addY = val

def trimImage(img, fromTop, newBot, fromLeft, newRight):
    return img[fromTop:newBot, fromLeft:newRight]

def pasteImage(base, sprite, posX, posY): #paste 1 image into another in the specified position
    base[round(posY):round(sprite.shape[0] + posY), round(posX):round(sprite.shape[1] + posX), :] = sprite
    return base

def addBlurredExtendBorder(src, top, bottom, left, right, blur, sigma): #add an extend border with a blur effect to smooth out varience
    blurred = cv.GaussianBlur(src, (blur[0], blur[1]), sigma)
    blurred = cv.copyMakeBorder(blurred, round(top), round(bottom), round(left), round(right), cv.BORDER_REPLICATE)
    blurred = pasteImage(blurred, sharpen(src, 3, 0.28, 1), left, top)
    return blurred

def sharpen(src, ksize, amount, iter):
    hsvSrc = cv.cvtColor(src, cv.COLOR_BGR2HSV_FULL)
    h, s, v = cv.split(hsvSrc)
    for _ in range(iter):
        v = img_as_ubyte(unsharp_mask(v, ksize, amount))
    return cv.cvtColor(cv.merge([h, s, v]), cv.COLOR_HSV2BGR_FULL)

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument("-i", "--Input", help="Set Input")

args = parser.parse_args()
image = cv.imread(cv.samples.findFile(args.Input))
size = (round(image.shape[1]/2), round(image.shape[0]/2))
#image = cv.resize(image, size, cv.INTER_AREA)

cv.namedWindow(window_detection_name, cv.WINDOW_NORMAL)
cv.createTrackbar("blur", window_detection_name, blur, 20, blurTrackbar)
cv.createTrackbar("sigma", window_detection_name, sigma, 400, sigmaTrackbar)
cv.createTrackbar("iter", window_detection_name, iter, 20, iterTrackbar)
cv.createTrackbar("addX", window_detection_name, addX, 50, XTrackbar)
cv.createTrackbar("addY", window_detection_name, addY, 50, YTrackbar)

while True:
    bordered = addBlurredExtendBorder(image, addY, addY, addX, addX, (blur*2+1, blur*2+1), sigma)
    cv.imshow(window_detection_name, bordered)
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break