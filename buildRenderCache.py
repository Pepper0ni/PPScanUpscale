import cv2 as cv
import numpy as np
import os
import argparse
import urllib.request
import utilities
from contextlib import suppress

targetRes = [734, 1024]
targetBorder = [targetRes[1] * 0.029296875, targetRes[1] * 0.029296875, targetRes[0] * 0.0408719346049,
                targetRes[0] * 0.0408719346049]

unownInput = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "!"]
unownOutput = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "exclamation-mark"]

def processImage(path, borders):
    resp = urllib.request.urlopen(path)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv.imdecode(img, cv.IMREAD_UNCHANGED)

    # borderRescale = np.divide(targetRes, [img.shape[1], img.shape[0]])
    # postScaleBorders = [borderRescale[1] * borders["top"], borderRescale[1] * borders["bottom"], borderRescale[0] * borders["left"],
    #                     borderRescale[0] * borders["right"]]
    # undersize = [int(max(0, targetBorder[0] - postScaleBorders[0] - 1)),
    #              int(max(0, targetBorder[1] - postScaleBorders[1] - 3)),
    #              int(max(0, targetBorder[2] - postScaleBorders[2] - 1)),
    #              int(max(0, targetBorder[3] - postScaleBorders[3] - 1))]
    # oversize = [int(max(0, postScaleBorders[0] - targetBorder[0] + 1)),
    #             int(max(0, postScaleBorders[1] - targetBorder[1] + 3)),
    #             int(max(0, postScaleBorders[2] - targetBorder[2] + 1)),
    #             int(max(0, postScaleBorders[3] - targetBorder[3] + 1))]
    # img = cv.resize(img, ([int(targetRes[0] - (undersize[2] + undersize[3]) + (oversize[2] + oversize[3])),
    #                            int(targetRes[1] - (undersize[0] + undersize[1]) + (oversize[0] + oversize[1]))]),
    #                   interpolation=cv.INTER_CUBIC)
    # img = cv.copyMakeBorder(img, undersize[0], undersize[1], undersize[2], undersize[3], cv.BORDER_REPLICATE)
    # img = utilities.trimImage(img, oversize[0], img.shape[0] - oversize[1], oversize[2],
    #                             img.shape[1] - oversize[3])
    img = cv.resize(img, targetRes, interpolation=cv.INTER_CUBIC)
    return img


def main():
    JSON, output = processArgs()
    JSON = utilities.loadJson(JSON)
    baseurl = "https://assets.pokemon.com/assets/cms2/img/cards/web/"
    for set in JSON:
        setSize = set["setSize"]
        setCode = set["setCode"]
        if "filename" in set:
            fileName = set["fileName"]
        else:
            fileName = setCode
        cacheName = os.path.join(output, set["folderName"])
        with suppress(FileExistsError):
            os.mkdir(cacheName)
        borders = set["borders"]
        partPath = baseurl + setCode + "/" + fileName + "_EN_"
        for count in range(setSize):
            savePath = os.path.join(cacheName, str(count + 1) + ".png")
            if os.path.isfile(savePath):
                print("skipping " + savePath + ": already exists")
            else:
                image = processImage(partPath + str(count + 1) + ".png", borders)
                cv.imwrite(savePath, image)
        if "unowns" in set:
            count = 0
            for letter in unownInput:
                savePath = os.path.join(cacheName, unownOutput[count] + ".png")
                if os.path.isfile(savePath):
                    print("skipping " + savePath + ": already exists")
                else:
                    image = processImage(partPath + "Unown_" + letter + ".png", borders)
                    cv.imwrite(savePath, image)
                count += 1


def processArgs():
    JSON = os.path.join(os.getcwd(), "RenderSetData.json")
    output = os.path.join(os.getcwd(), "cache/")

    msg = "sets up pokemon.com renders for color transfers"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-j", "--JSON", help="set JSON location")
    parser.add_argument("-o", "--output", help="set output location")
    args = parser.parse_args()

    if args.JSON:
        JSON = args.JSON
    if args.output:
        output = args.output

    return JSON, output

if __name__ == "__main__":
    main()