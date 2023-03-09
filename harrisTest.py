from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import math
from skimage.filters import unsharp_mask
from skimage.util import img_as_ubyte

block = 1
ksize = 9
k = 31
thresh = 25

window_detection_name = 'Object Detection'

def ksizeTrackbar(val):
    global ksize
    ksize = val

def blockTrackbar(val):
    global block
    block = val

def kTrackbar(val):
    global k
    k = val

def threshTrackbar(val):
    global thresh
    thresh = val

def trimImage(img, fromTop, newBot, fromLeft, newRight):
    return img[fromTop:newBot, fromLeft:newRight]

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

cv.namedWindow(window_detection_name, cv.WINDOW_NORMAL)
cv.createTrackbar("block", window_detection_name, block, 10, blockTrackbar)
cv.createTrackbar("ksize", window_detection_name, ksize, 15, ksizeTrackbar)
cv.createTrackbar("k", window_detection_name, k, 100, kTrackbar)
cv.createTrackbar("thresh", window_detection_name, thresh, 1000, threshTrackbar)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
image = sharpen(image, 5, 0.28, 1)
gimg = np.float32(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
_, simg, _2 = cv.split(np.float32(cv.cvtColor(image, cv.COLOR_BGR2HSV)))


while True:
    cimg = np.copy(image)
    corners = cv.cornerHarris(simg, block*2+1, ksize*2+1, k/1000)
    cimg[corners > thresh/10000 * corners.max()] = [255, 0, 255]
    ret, dst = cv.threshold(corners, 0.01 * corners.max(), 255, 0)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst.astype(np.uint8))
    corners = cv.cornerSubPix(simg, np.float32(centroids), (5, 5), (-1, -1), criteria)
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    #cimg[res[:, 3], res[:, 2]] = [255, 0, 0]


    corners = cv.cornerHarris(gimg, block*2+1, ksize*2+1, (k+1)/1000)
    cimg[corners > thresh/10000 * corners.max()] = [0, 0, 255]
    ret, dst = cv.threshold(corners, 0.01 * corners.max(), 255, 0)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst.astype(np.uint8))
    subCorners = cv.cornerSubPix(gimg, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # for corner in subCorners:
    #     if (corner[0] < 50 or corner[0] > image.shape[1] - 50) and (corner[1] < 50 or corner[1] > image.shape[0] - 50):
    #         print(corner)
    # subCorners = np.int0(subCorners)
    # cimg[subCorners[:, 1], subCorners[:, 0]] = [0, 255, 0]



    #cv.imshow(window_detection_name, cv.cvtColor(image, cv.COLOR_BGR2GRAY))
    cv.imshow(window_detection_name, cimg)
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break