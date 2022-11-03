import main
import os
import cv2 as cv
import argparse


input, clean, deborder, border, trim, expand, edge = main.processArgs("folder")

if not input:
    input = os.path.join(os.getcwd(), "input/test.jpg")
if not clean:
    clean = os.path.join(os.getcwd(), "temp/test.png")
if not border:
    border = [0,0,0,0]

image, dst, cdst, cdstV, cdstH = main.processImage(os.path.join(input), clean, border, trim, expand, edge, True)
cv.imshow("edges", dst)
cv.imshow("Horizontal Lines - Standard Hough Line Transform", cdstH)
cv.imshow("Vertical Lines - Standard Hough Line Transform", cdstV)
cv.imshow("4 main lines - Probabilistic Line Transform", cdst)
cv.imshow("debordered", image)
cv.waitKey()