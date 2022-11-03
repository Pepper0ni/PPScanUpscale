import os
import argparse
import cv2 as cv

input = os.path.join(os.getcwd(), "resized")
output = os.path.join(os.getcwd(), "output")
mask = os.path.join(os.getcwd(), "cardmask.png")


msg = "Masks pokemon card scan corners out"
parser = argparse.ArgumentParser(description=msg)
parser.add_argument("-i", "--Input", help="Set Input folder")
parser.add_argument("-o", "--Output", help="Set Output folder")
parser.add_argument("-m", "--Mask", help="Set mask path")

args = parser.parse_args()

if args.Input:
    input = args.Input
if args.Output:
    output = args.Output
if args.Mask:
    mask = args.Mask


with os.scandir(input) as entries:
    for entry in entries:
        if entry.is_file() and entry.name != "Place Images Here":
            imgname, extension = os.path.splitext(os.path.basename(entry.name))
            src = cv.imread(cv.samples.findFile(os.path.join(input, entry.name)), cv.IMREAD_UNCHANGED)
            maskImg = cv.imread(cv.samples.findFile(mask), cv.IMREAD_UNCHANGED)
            #print(maskImg)
            #cv.imshow("mask", maskImg)
            src = cv.cvtColor(src, cv.COLOR_BGR2BGRA)
            maskImg = cv.cvtColor(maskImg, cv.COLOR_BGR2BGRA)
            result = cv.bitwise_and(src, maskImg)

            if result is not None:
               cv.imwrite(os.path.join(output, imgname + ".png"), result)






