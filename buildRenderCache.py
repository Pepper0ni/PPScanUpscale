import cv2 as cv
import numpy as np
import os
import argparse
import urllib.request

def trimImage(img, fromTop, newBot, fromLeft, newRight): #crop the image based on the supplied values
    return img[fromTop:newBot, fromLeft:newRight]

def main():
    setsize = 102
    setcode = "EX5"
    cacheName = "./cache/ex-hidden-legends/"
    baseurl = "https://assets.pokemon.com/assets/cms2/img/cards/web/"
    borders = [10, 8, 9, 9]
    targetRes = [734, 1024]
    targetBorder = [targetRes[1]*0.029296875, targetRes[1]*0.029296875, targetRes[0]*0.0408719346049, targetRes[0]*0.0408719346049]

    for count in range(setsize):
        print(baseurl + setcode + "/" + setcode + "_EN_" + str(count + 1) + ".png")
        resp = urllib.request.urlopen(baseurl + setcode + "/" + setcode + "_EN_" + str(count + 1) + ".png")
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        #cv.imwrite(cacheName + str(count + 1) + "O.png", image)
        print(image.shape)

        borderRescale = np.divide(targetRes, [image.shape[1], image.shape[0]])
        print(borderRescale)
        postScaleBorders = [borderRescale[1] * borders[0], borderRescale[1] * borders[1], borderRescale[0] * borders[2], borderRescale[0] * borders[3]]
        print(postScaleBorders)
        print(targetBorder)
        undersize = [int(max(0, targetBorder[0] - postScaleBorders[0])),
                     int(max(0, targetBorder[1] - postScaleBorders[1] - 1)),
                     int(max(0, targetBorder[2] - postScaleBorders[2] - 1)),
                     int(max(0, targetBorder[3] - postScaleBorders[3] - 1))]
        oversize = [int(max(0, postScaleBorders[0] - targetBorder[0])),
                     int(max(0, postScaleBorders[1] - targetBorder[1] + 1)),
                     int(max(0, postScaleBorders[2] - targetBorder[2] + 1)),
                     int(max(0, postScaleBorders[3] - targetBorder[3] + 1))]
        print(undersize)
        print(oversize)
        #image = cv.resize(image, ([image.shape[0], image.shape[1]]), 2, 2, interpolation=cv.INTER_NEAREST)
        image = cv.resize(image, ([int(targetRes[0] - (undersize[2] + undersize[3]) + (oversize[2] + oversize[3])), int(targetRes[1] - (undersize[0] + undersize[1]) + (oversize[0] + oversize[1]))]), interpolation=cv.INTER_CUBIC)
        #cv.imwrite(cacheName + str(count + 1) + "T.png", image)
        image = cv.copyMakeBorder(image, undersize[0], undersize[1], undersize[2], undersize[3], cv.BORDER_REPLICATE)
        image = trimImage(image, oversize[0], image.shape[0] - oversize[1], oversize[2], image.shape[1] - oversize[3])

        cv.imwrite(cacheName + str(count + 1) + ".png", image)
    pass


if __name__ == "__main__":
    main()