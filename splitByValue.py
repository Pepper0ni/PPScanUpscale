import os
import argparse
import cv2 as cv
import numpy as np

def trimImage(img, fromTop, newBot, fromLeft, newRight):
    return img[fromTop:newBot, fromLeft:newRight]


def processImage(img):
    cutImg = trimImage(cv.cvtColor(img, cv.COLOR_BGR2HSV), round(img.shape[0] * 0.65), img.shape[0], 0, img.shape[1])
    #print(np.percentile(cutImg, 2, (0, 1)))
    if np.percentile(cutImg, 2, (0, 1))[2] > 30:
        return True
    else:
        return False


def processArgs(inputText):
    input = os.path.join(os.getcwd(), "input/")

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input" + inputText)
    parser.add_argument("-d", "--Dark", help="Set Output for dark" + inputText)
    parser.add_argument("-l", "--Light", help="Set Output for light" + inputText)

    args = parser.parse_args()

    if args.Input:
        input = args.Input
    filename = os.path.basename(input)
    if not filename:
        filename = ""
    if args.Dark:
        input = args.Dark
    else:
        dark = os.path.join(os.getcwd(), "dark/" + filename)
    if args.Light:
        input = args.Light
    else:
        light = os.path.join(os.getcwd(), "light/" + filename)

    return input, dark, light


def resolveImage(input, dark, light):
    print("sorting " + input)
    img = cv.imread(cv.samples.findFile(input))
    if img is None:
        print('Image at ' + input + ' Not Found, skipping')
        return
    isLight = processImage(img)
    if isLight:
        cv.imwrite(light, img)
    else:
        print(dark)
        cv.imwrite(dark, img)


def processFolder(input, dark, light):
    try:
        os.mkdir(dark)
        os.mkdir(light)
    except FileExistsError:
        pass
    with os.scandir(input) as entries:
        for entry in entries:
            inputPath = os.path.join(input, entry.name)
            lightPath = os.path.join(light, entry.name)
            darkPath = os.path.join(dark, entry.name)
            if os.path.isfile(inputPath) and entry.name != "Place Images Here":
                resolveImage(inputPath, darkPath, lightPath)
            elif os.path.isdir(inputPath):
                processFolder(inputPath, darkPath, lightPath)


def main():
    input, dark, light = processArgs("folder")
    if os.path.isfile(input):
        resolveImage(input, dark, light)
    elif os.path.isdir(input):
        processFolder(input, dark, light)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()
