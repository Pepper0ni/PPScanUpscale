import os
import argparse
import cv2 as cv
import numpy as np
from contextlib import suppress

def trimImage(img, fromTop, newBot, fromLeft, newRight):
    return img[fromTop:newBot, fromLeft:newRight]


def processImage(img, name):
    cutImg = trimImage(cv.cvtColor(img, cv.COLOR_BGR2HSV), round(img.shape[0] * 0.65), img.shape[0]-30, 30, img.shape[1]-30)
    #print(np.percentile(cutImg, 2, (0, 1)))
    if name.endswith("fe.png") and np.percentile(cutImg, 55, (0, 1))[2] < 70:
        return 2
    elif np.percentile(cutImg, 2.25, (0, 1))[2] <= 30:
        return 1
    else:
        return 0


def processArgs(inputText):
    input = os.path.join(os.getcwd(), "input/")

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input" + inputText)
    parser.add_argument("-d", "--Dark", help="Set Output for dark" + inputText)
    parser.add_argument("-b", "--Balanced", help="Set Output for balanced" + inputText)
    parser.add_argument("-l", "--Light", help="Set Output for light" + inputText)

    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if os.path.isfile(input):
        filename = os.path.basename(input)
    else:
        filename = ""
    if args.Dark:
        dark = args.Dark
    else:
        dark = os.path.join(os.getcwd(), "dark/" + filename)
    if args.Balanced:
        balanced = args.Balanced
    else:
        balanced = os.path.join(os.getcwd(), "balanced/" + filename)
    if args.Light:
        light = args.Light
    else:
        light = os.path.join(os.getcwd(), "light/" + filename)

    return input, dark, balanced, light


def resolveImage(input, dark, balanced, light):
    print("sorting " + input)
    img = cv.imread(cv.samples.findFile(input))
    if img is None:
        print('Image at ' + input + ' Not Found, skipping')
        return
    isLight = processImage(img, input)
    if isLight == 0:
        cv.imwrite(light, img)
    elif isLight == 1:
        cv.imwrite(balanced, img)
    else:
        cv.imwrite(dark, img)


def processFolder(input, dark, balanced, light):
    with suppress(FileExistsError):
        os.mkdir(dark)
        os.mkdir(light)
        os.mkdir(balanced)
    with os.scandir(input) as entries:
        for entry in entries:
            inputPath = os.path.join(input, entry.name)
            lightPath = os.path.join(light, entry.name)
            balancedPath = os.path.join(balanced, entry.name)
            darkPath = os.path.join(dark, entry.name)
            if os.path.isfile(inputPath):
                resolveImage(inputPath, darkPath, balancedPath, lightPath)
            elif os.path.isdir(inputPath):
                processFolder(inputPath, darkPath, balancedPath, lightPath)


def main():
    input, dark, balanced, light = processArgs("folder")
    if os.path.isfile(input):
        resolveImage(input, dark, balanced, light)
    elif os.path.isdir(input):
        processFolder(input, dark, balanced, light)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()
