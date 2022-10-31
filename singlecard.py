import main
import os
import cv2 as cv
import argparse

input = os.path.join(os.getcwd(), "input/test.jpg")
clean = os.path.join(os.getcwd(), "temp/test.png")
#deborder = os.path.join(os.getcwd(), "debordered")

msg = "Adding description"
parser = argparse.ArgumentParser(description=msg)
parser.add_argument("-i", "--Input", help="Set Input folder")
#parser.add_argument("-d", "--Deborder", help="Set Deborder folder")
parser.add_argument("-c", "--Clean", help="Set Clean Images folder")
#parser.add_argument("-o", "--Output", help="Set Output folder")
args = parser.parse_args()

if args.Input:
    input = args.Input
if args.Clean:
    clean = args.Clean
# if args.Deborder:
#     deborder = args.Deborder
# if args.Output:
#     output = args.Output

image, dst, cdst, cdstV, cdstH = main.processImage(input, clean, True)
cv.imshow("edges", dst)
cv.imshow("Horizontal Lines - Standard Hough Line Transform", cdstH)
cv.imshow("Vertical Lines - Standard Hough Line Transform", cdstV)
cv.imshow("4 main lines - Probabilistic Line Transform", cdst)
cv.imshow("debordered", image)
cv.waitKey()