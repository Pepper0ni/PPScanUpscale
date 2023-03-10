import math
import cv2 as cv
import numpy as np
import os
import argparse
from decimal import Decimal
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.util import img_as_ubyte
import skimage.draw
import maskcorners
import operator
from contextlib import suppress
import taskbarFuncs
from skimage.filters import unsharp_mask
import re
import sys
import utilities
import random

ANGLE_TOLERANCE = 0.0314  # maximum variance from a straight line the line detector will accept in radians
DEFAULT_BORDER_TOLERANCE = 0.0327272727273  # sets how far into an image to look for a border, multiplied by y coord
DEFAULT_H_THRESHOLD = 0.1  # sets how strict the horizontal line detector is, multiplied by X coord
DEFAULT_V_THRESHOLD = 0.1  # sets how strict the vertical line detector is, multiplied by y coord
MIN_LINE_EDGE = 3  # sets how many pixels from the edge to exclude from line detection
MIN_ELEC_AVG_RANGE = 5  # sets the distance from the edge to exclude from apative border detection
MAX_ELEC_AVG_RANGE = 30  # sets the distance away from the edge to include in apative border detection
ELEC_HUE_VAR = 1  # sets the variance in hue for the adaptive border detection to accept
ELEC_SAT_VAR = 3  # sets the variance in saturation for the adaptive border detection to accept
ELEC_VAL_VAR = 1  # sets the variance in value for the adaptive border detection to accept
CORNER_FIX_STREGNTH = 5  # sets how large an area, and how strong a blur, to use for the corner fixing
timesRun = 0
BORDER_LIMIT = 26
MIN_SAT = 80

def matchLowSatValue(img, border, side):
    if side == 0:
        trimImg = utilities.trimImage(np.copy(img), 0, 20, 0, img.shape[1])
        trimBor = utilities.trimImage(np.copy(border), 0, 20, 0, border.shape[1])
    elif side == 1:
        trimImg = utilities.trimImage(np.copy(img), img.shape[0] - 20, img.shape[0], 0, img.shape[1])
        trimBor = utilities.trimImage(np.copy(border), border.shape[0] - 20, border.shape[0], 0, border.shape[1])
    elif side == 2:
        trimImg = utilities.trimImage(np.copy(img), 0, img.shape[0], 0, 20)
        trimBor = utilities.trimImage(np.copy(border), 0, border.shape[0], 0, 20)
    elif side == 3:
        trimImg = utilities.trimImage(np.copy(img), 0, img.shape[0], img.shape[1] - 20, img.shape[1])
        trimBor = utilities.trimImage(np.copy(border), 0, border.shape[0], border.shape[1] - 20, border.shape[1])
    #
    # cv.imshow("img", trimImg)
    # cv.imshow("border", trimBor)
    # cv.waitKey()

    bH, bS, bV = cv.split(cv.cvtColor(trimBor, cv.COLOR_BGR2HSV))
    bEx = np.extract(np.less(bS, MIN_SAT), bV)
    borVal = np.percentile(bEx, 95)

    iH, iS, iV = cv.split(cv.cvtColor(trimImg, cv.COLOR_BGR2HSV))
    iEx = np.extract(np.less(iS, MIN_SAT), iV)
    imgVal = np.percentile(iEx, 95)

    bH, bS, bV = cv.split(cv.cvtColor(border, cv.COLOR_BGR2HSV))

    reduce = int((borVal-imgVal)-5)

    # cv.imshow("old", border)

    if reduce > 0:
        satMask = np.less(bS, MIN_SAT)
        np.subtract(bV, reduce, bV, where=satMask)
        np.putmask(bV, np.logical_and(np.greater(bV, 255 - reduce), satMask), 0) #counter undeflow
        border = cv.cvtColor(cv.merge([bH, bS, bV]), cv.COLOR_HSV2BGR)

    return border


def finishDonerRow(img, border, imgOffset, row, start, end, isUp, valThresh, buffer, limit):
    imgRow = row + imgOffset
    startPoint = False
    endPoint = False
    if any(border[row, start] > valThresh) and all(img[imgRow, start] <= valThresh) <= valThresh:  # if start is bright
        startPoint = start
        while startPoint - 1 >= buffer and any(border[row, startPoint - 1] > valThresh) and all(img[imgRow, startPoint - 1] <= valThresh):
            startPoint -= 1  # go back 1 at a time until a pixel is dim and mark the pixel before that as the start
    elif start != border.shape[1] - 2:
        startPoint = start + 1  # else check when bright pixels start, and set start to it
        while startPoint <= end and all(border[row, startPoint] <= valThresh) and all(img[imgRow, startPoint] <= valThresh):
            startPoint += 1

    if not startPoint or end < startPoint:  # if start is past end, no more bright pixels, so stop looking
        return False

    if any(border[row, end] > valThresh) and all(img[imgRow, end] <= valThresh):  # find end point of the zone.
        endPoint = end  # if it's bright, keep going until it's dim
        while endPoint + 1 <= img.shape[1] - buffer and any(border[row, endPoint + 1] > valThresh) and all(img[imgRow, endPoint + 1] <= valThresh):
            endPoint += 1
    elif end != 0:
        endPoint = end - 1  # else go back and look for the start
        while endPoint >= startPoint and all(border[row, endPoint] <= valThresh) and all(img[imgRow, endPoint] <= valThresh):
            endPoint -= 1

    if not endPoint or endPoint < startPoint: #if the end is before the start, stop looking
        return

    for count in range(startPoint, endPoint):
        img[imgRow, count] = border[row, count] #actually transfer over the pixels

    if isUp:  # go to the next row
        if row >= 0:
            finishDonerRow(img, border, imgOffset, row - 1, startPoint, endPoint, isUp, valThresh, buffer, limit)
    else:
        if row < limit:
            finishDonerRow(img, border, imgOffset, row + 1, startPoint, endPoint, isUp, valThresh, buffer, limit)

def finishDonerCol(img, border, imgOffset, col, start, end, isLeft, valThresh, buffer, limit):
    imgCol = col + imgOffset
    startPoint = False
    endPoint = False
    if any(border[start, col] > valThresh) and all(img[start, imgCol] <= valThresh) <= valThresh:  # if start is bright...
        startPoint = start
        while startPoint - 1 >= buffer and any(border[startPoint - 1, col] > valThresh) and all(img[startPoint - 1, imgCol] <= valThresh):
            startPoint -= 1  # go back 1 at a time until a pixel is dim, and mark the last bright pixel as the start
    elif start != border.shape[0] - 2:
        startPoint = start + 1  # else check when bright pixels start, and set start to it
        while startPoint <= end and all(border[startPoint, col] <= valThresh) and all(img[startPoint, imgCol] <= valThresh):
            startPoint += 1

    if not startPoint or end < startPoint:  # if start is past end, no more bright pixels, stop looking
        return False

    if any(border[end, col] > valThresh) and all(img[end, imgCol] <= valThresh):  # find end point of the zone.
        endPoint = end  # if it's bright, keep going until it's dim
        while endPoint + 1 <= img.shape[0] - buffer and any(border[endPoint + 1, col] > valThresh) and all(img[endPoint + 1, imgCol] <= valThresh):
            endPoint += 1
    elif end != 0:
        endPoint = end - 1  # else go back and look for the first bright pixel
        while endPoint >= startPoint and all(border[endPoint, col] <= valThresh) and all(img[endPoint, imgCol] <= valThresh):
            endPoint -= 1

    if not endPoint or endPoint < startPoint:
        return

    for count in range(startPoint, endPoint):
        img[count, imgCol] = border[count, col]

    if isLeft:  # go to next column
        if col >= 0:
            finishDonerCol(img, border, imgOffset, col - 1, startPoint, endPoint, isLeft, valThresh, buffer, limit)
    else:
        if col < limit:
            finishDonerCol(img, border, imgOffset, col + 1, startPoint, endPoint, isLeft, valThresh, buffer, limit)


def mergeBorderH(img, borImg, valThresh, size, offset, top):
    oldImg = img.copy()
    iH, iS, iV = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    if top:
        trimImg = utilities.trimImage(np.copy(img), offset, offset + size, 0, img.shape[1])
    else:
        trimImg = utilities.trimImage(np.copy(img), img.shape[0] - size, img.shape[0], 0, img.shape[1])

    nH, nS, nV = cv.split(cv.cvtColor(trimImg, cv.COLOR_BGR2HSV))

    LBGRd = cv.cvtColor(cv.cvtColor(trimImg, cv.COLOR_BGR2LAB), cv.COLOR_LAB2LBGR)
    LBGRb = cv.cvtColor(cv.cvtColor(borImg, cv.COLOR_BGR2LAB), cv.COLOR_LAB2LBGR)
    borImg = cv.normalize(PCAColorTransfer(LBGRb, LBGRd), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    borImg = (utilities.clipToOne(
        cv.cvtColor(cv.cvtColor(borImg, cv.COLOR_LBGR2LAB), cv.COLOR_LAB2BGR) / 255) * 255).astype(np.uint8)

    if top:
        borImg = matchLowSatValue(trimImg, borImg, 0)
        trimBor = utilities.trimImage(borImg, 0, size, 0, borImg.shape[1])
    else:
        borImg = matchLowSatValue(trimImg, borImg, 1)
        trimBor = utilities.trimImage(borImg, borImg.shape[0] - size, borImg.shape[0], 0, borImg.shape[1])

    if valThresh:
        bH, bS, bV = cv.split(cv.cvtColor(trimBor, cv.COLOR_BGR2HSV))
        np.putmask(trimImg, logialOrValSat(trimImg.shape, nV, bV, nS, valThresh, 50), trimBor)

    if top:
        pasteImage(img, trimImg, 0, 0)
    else:
        pasteImage(img, trimImg, 0, img.shape[0] - size)

    if valThresh:
        col = 0
        bH, bS, bV = cv.split(cv.cvtColor(borImg, cv.COLOR_BGR2HSV))
        if top:
            tarRowb = size + 1
            tarRowi = size + 1
            nextRowi = tarRowi + 1
            nextRowb = tarRowb + 1
        else:
            tarRowb = borImg.shape[0] - size
            tarRowi = img.shape[0] - size - 1
            nextRowi = tarRowi - 1
            nextRowb = tarRowb - 1

        while col < img.shape[1]:  # finish speckles of doner border
            if bV[tarRowb, col] > valThresh:  # if last on border is bright
                img[tarRowi, col] = borImg[tarRowb, col]
                start = col
                while col + 1 < img.shape[1] and bV[tarRowb, col + 1] > valThresh:
                    col += 1
                    #print("writing " + str(img[tarRowi, col]) + " from" + str([tarRowb, col]) + "to" + str([tarRowi, col]))
                    img[tarRowi, col] = borImg[tarRowb, col]
                #print(start)
                #print(col)
                finishDonerRow(img, borImg, offset, nextRowb, start, col, not top, valThresh, size, BORDER_LIMIT)
                # repeat for next row up
            col += 1

        # cv.imshow("top", img)
        # cv.waitKey()

        col = 0
        climbList = []
        while col < img.shape[1]:  #finish circles of base border
            loCol = max(0, col - 1)
            hiCol = min(img.shape[1] - 1, col + 1)
            if iV[tarRowi, col] > valThresh and iV[nextRowi, loCol] > valThresh and iV[nextRowi, hiCol] > valThresh:
                result = averagePixels(oldImg[tarRowi, col], img[nextRowi, loCol], img[nextRowi, hiCol])
                img[tarRowi, col] = result
                #print("writing " + str(result) + " to " + str([tarRowi, col]))
                climbList.append(col)
            col += 1

        X = 0
        newClimbList = []
        curRow = tarRowi
        while True:
            lastRow = curRow
            #print(climbList)
            #print(lastRow)
            if top:
                curRow -= 1
            else:
                curRow += 1
            if not climbList or curRow < 0 or curRow >= img.shape[0]:
                break
            for col in climbList:
                hiCol = min(img.shape[1] - 1, col + X)
                loCol = max(0, col - X)
                # print([curRow, col])
                # print([lastRow, loCol])
                # print([lastRow, hiCol])
                if iV[curRow, col] > valThresh and any(img[lastRow, loCol] > valThresh) and any(img[lastRow, hiCol] > valThresh):
                    result = averagePixels(oldImg[curRow, col], img[lastRow, loCol], img[lastRow, hiCol])
                    img[curRow, col] = result
                    #print("writing " + str(result) + " to " + str([curRow, col]))
                    newClimbList.append(col)
            climbList = newClimbList
            newClimbList = []
            X += 1

    # cv.imshow("top", img)
    # cv.waitKey()
    return img

def mergeBorderV(img, borImg, valThresh, size, offset, left):
    print(offset)
    oldImg = img.copy()
    iH, iS, iV = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    if left:
        trimImg = utilities.trimImage(np.copy(img), 0, img.shape[0], offset, offset + size)
    else:
        trimImg = utilities.trimImage(np.copy(img), 0, img.shape[0], img.shape[1] - size, img.shape[1])

    # cv.imshow("top", trimImg)
    # cv.waitKey()
    nH, nS, nV = cv.split(cv.cvtColor(trimImg, cv.COLOR_BGR2HSV))

    LBGRd = cv.cvtColor(cv.cvtColor(trimImg, cv.COLOR_BGR2LAB), cv.COLOR_LAB2LBGR)
    LBGRb = cv.cvtColor(cv.cvtColor(borImg, cv.COLOR_BGR2LAB), cv.COLOR_LAB2LBGR)
    borImg = cv.normalize(PCAColorTransfer(LBGRb, LBGRd), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    borImg = (utilities.clipToOne(
        cv.cvtColor(cv.cvtColor(borImg, cv.COLOR_LBGR2LAB), cv.COLOR_LAB2BGR) / 255) * 255).astype(np.uint8)

    if left:
        borImg = matchLowSatValue(trimImg, borImg, 2)
        trimBor = utilities.trimImage(borImg, 0, borImg.shape[0], 0, size)
    else:
        borImg = matchLowSatValue(trimImg, borImg, 3)
        trimBor = utilities.trimImage(borImg, 0, borImg.shape[0], borImg.shape[1] - size, borImg.shape[1])

    if valThresh:
        bH, bS, bV = cv.split(cv.cvtColor(trimBor, cv.COLOR_BGR2HSV))
        np.putmask(trimImg, logialOrValSat(trimImg.shape, nV, bV, nS, valThresh, MIN_SAT), trimBor)

    if left:
        pasteImage(img, trimImg, 0, 0)
    else:
        pasteImage(img, trimImg, img.shape[1] - size, 0)

    # cv.imshow("top", trimImg)
    # cv.waitKey()

    if valThresh:
        row = 0
        bH, bS, bV = cv.split(cv.cvtColor(borImg, cv.COLOR_BGR2HSV))
        if left:
            tarColb = size + 1
            tarColi = size + 1
            nextColi = tarColi + 1
            nextColb = tarColb + 1
        else:
            tarColb = borImg.shape[1] - size
            tarColi = img.shape[1] - size - 1
            nextColi = tarColi - 1
            nextColb = tarColb - 1

        # print(tarColb)
        # print(tarColi)

        while row < img.shape[0]:  # top finish speckles on doner border
            if bV[row, tarColb] > valThresh:  # if last on border is bright
                img[row, tarColi] = borImg[row, tarColb]
                start = row
                while row + 1 < img.shape[0] and bV[row + 1, tarColb] > valThresh:
                    row += 1
                    #print("writing " + str(img[row, tarColi]) + " from" + str([row, tarColb]) + "to" + str([row, tarColi]))
                    img[row, tarColi] = borImg[row, tarColb]
                # print(start)
                # print(row)
                # print(nextColb)
                finishDonerCol(img, borImg, offset, nextColb, start, row, not left,
                               valThresh, size, BORDER_LIMIT)  # repeat for next row up
            row += 1

        # cv.imshow("top", img)
        # cv.waitKey()

        row = 0
        climbList = []
        while row < img.shape[0]:  # top finish circles of base border
            loRow = max(0, row - 1)
            hiRow = min(img.shape[0] - 1, row + 1)
            if iV[row, tarColi] > valThresh and iV[loRow, nextColi] > valThresh and iV[hiRow, nextColi] > valThresh:
                result = averagePixels(oldImg[row, tarColi], img[loRow, nextColi], img[hiRow, nextColi])
                img[row, tarColi] = result
                #print("writing " + str(result) + " to " + str([row, tarColi]))
                climbList.append(row)
            row += 1

        X = 0
        newClimbList = []
        curCol = tarColi
        while True:
            lastCol = curCol
            #print(lastCol)
            if left:
                curCol -= 1
            else:
                curCol += 1
            if not climbList or curCol < 0 or curCol >= img.shape[1]:
                break
            for row in climbList:
                hiRow = min(img.shape[1] - 1, row + X)
                loRow = max(0, row - X)
                # print([curCol, row])
                # print([lastCol, loRow])
                # print([lastCol, hiCol])
                if iV[row, curCol] > valThresh and any(img[loRow, lastCol] > valThresh) and any(img[hiRow, lastCol] > valThresh):
                    result = averagePixels(oldImg[row, curCol], img[loRow, lastCol], img[hiRow, lastCol])
                    img[row, curCol] = result
                    #print("writing " + str(result) + " to " + str([row, curCol]))
                    newClimbList.append(row)
            #print(newClimbList)
            climbList = newClimbList
            newClimbList = []
            X += 1

    # cv.imshow("top", img)
    # cv.waitKey()
    return img


def averagePixels(pix1, pix2, pix3):
    return np.array([(int(pix1[0]) + int(pix2[0]) + int(pix3[0])) / 3, (int(pix1[1]) + int(pix2[1]) + int(pix3[1])) / 3,
                     (int(pix1[2]) + int(pix2[2]) + int(pix3[2])) / 3])


def logialOrValSat(targetShape, nV, tV, nS, valThresh, satThresh):
    if valThresh:
        return np.broadcast_to(np.logical_or(
            np.logical_or(np.greater_equal(nV, valThresh),
                          np.greater_equal(tV, valThresh)),
            np.greater_equal(nS, satThresh),
        )[:, :, np.newaxis]
                               , targetShape)
    else:
        return np.full(targetShape, True)


def getRandomImage(path, folder):
    fullPath = os.path.join(path, folder)
    filePath = os.path.join(fullPath,
                            random.choice(
                                [x for x in os.listdir(fullPath) if os.path.isfile(os.path.join(fullPath, x))]))
    return cv.imread(cv.samples.findFile(filePath))



def pasteBorders(img, path, valThresh=175, size=15):
    newImg = np.copy(img)

    tBor = getRandomImage(path, "top")
    newImg = mergeBorderH(newImg, tBor, valThresh, size, 0, True)

    bBor = getRandomImage(path, "bottom")
    newImg = mergeBorderH(newImg, bBor, valThresh, size, newImg.shape[0] - bBor.shape[0], False)

    lBor = getRandomImage(path, "left")
    newImg = mergeBorderV(newImg, lBor, valThresh, size, 0, True)

    rBor = getRandomImage(path, "right")
    newImg = mergeBorderV(newImg, rBor, valThresh, size, newImg.shape[1] - rBor.shape[1], False)


    return newImg


def checkImageHue(hsvClean, lowRange, hiRange):
    elecCheck = cv.inRange(
        utilities.trimImage(hsvClean, int(hsvClean.shape[0] * 0.6), hsvClean.shape[0], 0, hsvClean.shape[1]), lowRange,
        hiRange)  # check the colour of the image to see if it's electric and needs tighter filters
    elecCheck = cv.medianBlur(elecCheck, 7)
    if np.sum(elecCheck) > hsvClean.shape[0] * hsvClean.shape[
        1] * 64:  # check only against the text of the image, to avoid the art throwing it off
        return True
    return False


def weightedAverageMatrix(img1, img2, mask, max, valThresh=False):
    invMask = np.abs(np.subtract(mask, max))
    multi1 = np.multiply(mask, img1)
    multi2 = np.multiply(invMask, img2)
    return np.array(np.divide(np.add(multi1, multi2), max), dtype=np.uint8)


def getChroma(img):
    high = np.max(img, (2))
    low = np.min(img, (2))
    return np.subtract(high, low)


def setSaturation(img, sats):
    img = img.astype(float)
    sats = sats.astype(float)
    order = np.argpartition(img, 1, 2)
    result = np.zeros_like(img)
    zeros = np.zeros_like(sats)
    sorted = np.take_along_axis(img, order, 2)
    mids = np.divide((sorted[..., 1] - sorted[..., 0]) * sats, sorted[..., 2] - sorted[..., 0],
                     where=sorted[..., 2] != sorted[..., 0])
    comb = cv.merge([zeros, mids, sats])
    np.put_along_axis(result, order, comb, 2)
    return result
    # return cv.normalize(result, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def addLightness(img, light):
    # print("beep")
    # print(light[474, 393])
    order = np.argpartition(img, 1, 2)
    Limg = np.add(img, light)
    # Limg[Limg > 1] = 1 #TODO fix properly
    Limg[Limg < 0] = 0
    min, mid, max = cv.split(np.take_along_axis(Limg, order, 2))
    # print(Limg[474, 393])

    Y = utilities.getLuminosity(Limg)
    # print(Y[474, 393])
    # print(max[474, 393])
    nY = Y[:, :, np.newaxis]
    nMin = min[:, :, np.newaxis]
    nMax = max[:, :, np.newaxis]

    # print(np.min(Limg))
    # print(np.min(Y))
    # print(np.max(Limg))

    condMask = np.less(min, 0)[:, :, np.newaxis]
    np.multiply((Limg - nY) * nY, np.divide(1, nY - nMin, where=condMask), Limg, where=condMask)
    np.add(Limg, nY, Limg, where=condMask)

    # print(np.min(Limg))
    # print(np.max(Limg))

    # print((max[474, 393] - 1) / sys.float_info.epsilon)
    condMask = np.logical_and(np.round(max, 1) > 1, max - Y > sys.float_info.epsilon)[:, :, np.newaxis]
    np.multiply((Limg - nY) * (1 - nY), np.divide(1, nMax - nY, where=condMask), Limg, where=condMask)
    np.add(Limg, nY, Limg, where=condMask)
    # print(Limg[474, 393])
    Limg[Limg > 1] = 1
    # print(np.min(Limg))
    # print(np.max(Limg))
    return Limg


def revertBadHue(og, mod):
    # ogH, ogS, ogV = cv.split(cv.cvtColor(og, cv.COLOR_BGR2HSV))
    # modH, modS, modV = cv.split(cv.cvtColor(mod, cv.COLOR_BGR2HSV))
    # fixed = np.where(np.logical_or(np.less(np.subtract(ogH, modH), 15), np.greater(np.subtract(ogH, modH), 240))[:, :, np.newaxis], mod, og)
    # fixedH, fixedS, fixedV = cv.split(cv.cvtColor(fixed, cv.COLOR_BGR2HSV))
    # #return fixed
    # return cv.cvtColor(cv.merge([fixedH, fixedS, fixedV]), cv.COLOR_HSV2BGR)

    ogH, ogS, ogV = cv.split(cv.cvtColor(og, cv.COLOR_BGR2HSV))
    modH, modS, modV = cv.split(cv.cvtColor(mod, cv.COLOR_BGR2HSV))
    fixed = np.where(
        np.logical_or(np.less(np.subtract(ogH, modH), 5), np.greater(np.subtract(ogH, modH), 250))[:, :, np.newaxis],
        mod, og)
    fixedH, fixedS, fixedV = cv.split(cv.cvtColor(fixed, cv.COLOR_BGR2HSV))
    # return fixed
    return cv.cvtColor(cv.merge([fixedH, fixedS, modV]), cv.COLOR_HSV2BGR)


def boostContrast(img, doner):
    H, S, V = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    sV = utilities.trimImage(V, 525, V.shape[0], 0, V.shape[1])
    VMean = np.percentile(sV, 30)  # min(sV.mean(0).mean(0)*0.9, 255)
    # print(VMean)
    dH, dS, dV = cv.split(cv.cvtColor(doner, cv.COLOR_BGR2HSV))
    sdV = utilities.trimImage(dV, 525, doner.shape[0], 0, doner.shape[1])
    # print("source percentile " + str(np.percentile(sV, 0.5)))
    # print("doner percentile " + str(np.percentile(sdV, 0.1)))
    toLower = np.percentile(sV, 0.5) - np.percentile(sdV, 0.1)
    # print(toLower)
    V = V.astype(float)
    # print(np.less(V, VMean))
    # print(utilities.customLog(VMean, VMean-toLower))
    if toLower > 0:
        # print(V[750, 289])
        np.subtract(V, toLower, V, where=np.less(V, VMean))
        # print(V[750, 289])
        np.multiply(V, VMean / (VMean - toLower), V, where=np.less(V, VMean))
        # np.power(V, utilities.customLog(VMean, VMean-toLower), V,  where=np.less(V, VMean))
        # print(V[750, 289])
    V[V < 0] = 0
    V[V > 255] = 255
    # print(cv.cvtColor(cv.merge([H, S, V.astype(np.uint8)]), cv.COLOR_HSV2BGR)[750, 289])
    return cv.cvtColor(cv.merge([H, S, V.astype(np.uint8)]), cv.COLOR_HSV2BGR)


def varianceColorTransfer(img, doner, debug):
    doner = cv.resize(doner, (img.shape[1], img.shape[0]))
    donerRe = np.reshape(doner, (doner.shape[1] * doner.shape[0], doner.shape[2])).astype(float)
    imgRe = np.reshape(img, (img.shape[1] * img.shape[0], img.shape[2])).astype(float)
    donerVar = np.var(donerRe, 0)
    imgVar = np.var(imgRe, 0)
    # powers = np.log(donerVar) / np.log(imgVar)
    # if debug:
    #     print("powers: " + str(powers))
    # scaledImg = np.power(img, powers[np.newaxis: np.newaxis:])
    factors = donerVar / imgVar
    if debug:
        print("factors: " + str(factors))
    scaledImg = np.multiply(img, factors[np.newaxis: np.newaxis:])
    simgRe = np.reshape(scaledImg, (scaledImg.shape[1] * scaledImg.shape[0], scaledImg.shape[2]))
    meanDiff = np.mean(donerRe - simgRe, 0)
    scaledImg = scaledImg + meanDiff[np.newaxis: np.newaxis:]
    scaledImg[scaledImg > 255] = 255
    scaledImg[scaledImg < 0] = 0
    return cv.normalize(scaledImg, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def PCAColorTransfer(img, doner):
    iMean = img.mean(0).mean(0)
    dMean = doner.mean(0).mean(0)
    _, dStd = cv.meanStdDev(doner)
    imgUnrav = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2])) - iMean
    donUnrav = np.reshape(doner, (doner.shape[0] * doner.shape[1], img.shape[2])) - dMean
    imgCov = np.cov(imgUnrav.T)
    donCov = np.cov(donUnrav.T)
    imgEval, imgEvect = np.linalg.eigh(imgCov)
    donEval, donEvect = np.linalg.eigh(donCov)
    QiUnk = (imgEvect * np.sqrt(np.abs(imgEval))).dot(imgEvect.T) + 1e-5 * np.eye(imgUnrav.shape[1])
    QdUnk = (donEvect * np.sqrt(np.abs(donEval))).dot(donEvect.T) + 1e-5 * np.eye(donUnrav.shape[1])
    result = QdUnk.dot(np.linalg.inv(QiUnk)).dot(imgUnrav.T)
    result = result.reshape(*img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    result += dMean
    result[result > 255] = 255
    result[result < 0] = 0
    return result


def smoothBorders(src, top, bottom, left, right, blur, cb, valThresh):
    if cb:
        borderImgs = [[], [], [], []]
        masks = [[], [], [], []]
        cornerMasks = [[], [], [], []]
        cornerImgs = [[], [], [], []]
        finalCorners = [[], [], [], []]
        top = top - cb[0]
        bottom = bottom - cb[1]
        left = left - cb[2]
        right = right - cb[3]
        caps = [0, 0, 0, 0]

        borderImgs[0] = utilities.trimImage(src, 0, top, 0, src.shape[1])  # get just the borders
        borderImgs[1] = utilities.trimImage(src, src.shape[0] - bottom, src.shape[0], 0, src.shape[1])
        borderImgs[2] = utilities.trimImage(src, 0, src.shape[0], 0, left)
        borderImgs[3] = utilities.trimImage(src, 0, src.shape[0], src.shape[1] - right, src.shape[1])

        maskRow = np.tile(np.array([[[1, 1, 1]], [[2, 2, 2]], [[3, 3, 3]]], dtype=np.int16),
                          (1, borderImgs[0].shape[1], 1))  # get the smooth transition masks
        invMaskRow = np.tile(np.array([[[3, 3, 3]], [[2, 2, 2]], [[1, 1, 1]]], dtype=np.int16),
                             (1, borderImgs[0].shape[1], 1))
        maskCol = np.tile(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int16).transpose(),
                          (borderImgs[2].shape[0], 1, 1))
        invMaskCol = np.tile(np.array([[3, 3, 3], [2, 2, 2], [1, 1, 1]], dtype=np.int16).transpose(),
                             (borderImgs[2].shape[0], 1, 1))

        # finish the 4 masks by making solid blocks and attacking the smoothness to them
        masks[0] = np.full(shape=(borderImgs[0].shape[0] - 3, borderImgs[0].shape[1], 3), fill_value=0, dtype=np.int16)
        masks[0] = np.row_stack((masks[0], maskRow))
        masks[0] = pasteImage(masks[0], np.zeros((3, left, 3), np.int16), 0, top - 3)
        masks[0] = pasteImage(masks[0], np.zeros((3, right, 3), np.int16), masks[0].shape[1] - right, top - 3)

        masks[1] = np.full(shape=(borderImgs[1].shape[0] - 3, borderImgs[1].shape[1], 3), fill_value=0, dtype=np.int16)
        masks[1] = np.row_stack((invMaskRow, masks[1]))
        masks[1] = pasteImage(masks[1], np.zeros((3, left, 3), np.int16), 0, 0)
        masks[1] = pasteImage(masks[1], np.zeros((3, right, 3), np.int16), masks[1].shape[1] - right, 0)

        masks[2] = np.full(shape=(borderImgs[2].shape[0], borderImgs[2].shape[1] - 3, 3), fill_value=0, dtype=np.int16)
        masks[2] = np.column_stack((masks[2], maskCol))
        masks[2] = pasteImage(masks[2], np.zeros((top, 3, 3), np.int16), left - 3, 0)
        masks[2] = pasteImage(masks[2], np.zeros((bottom, 3, 3), np.int16), left - 3, masks[2].shape[0] - bottom)

        masks[3] = np.full(shape=(borderImgs[3].shape[0], borderImgs[3].shape[1] - 3, 3), fill_value=0, dtype=np.int16)
        masks[3] = np.column_stack((invMaskCol, masks[3]))
        masks[3] = pasteImage(masks[3], np.zeros((top, 3, 3), np.int16), 0, 0)
        masks[3] = pasteImage(masks[3], np.zeros((bottom, 3, 3), np.int16), 0, masks[3].shape[0] - bottom)

        # sharpen the image
        trimmed = sharpen(utilities.trimImage(src, top, src.shape[0] - bottom, left, src.shape[1] - right), 3, 0.28, 1)

        if valThresh:
            count = 0
            for mask in masks:
                sH, sS, sV = cv.split(cv.cvtColor(borderImgs[count], cv.COLOR_BGR2HSV))
                np.putmask(mask, logialOrValSat(mask.shape, sV, sV, sS, valThresh, 125), 4)
                count += 1
        count = 0
        for border in borderImgs:
            blurred = cv.blur(np.array(border, dtype=np.int16), (blur[0], blur[1]))
            borderImgs[count] = weightedAverageMatrix(border, blurred, masks[count],
                                                      4)  # make an image of the images merged
            count += 1

        cornerImgs[0] = [utilities.trimImage(borderImgs[0], 0, borderImgs[0].shape[0], 0, left),
                         utilities.trimImage(borderImgs[2], 0, top, 0,
                                             borderImgs[2].shape[1])]  # get images of the corners

        cornerImgs[1] = [utilities.trimImage(borderImgs[0], 0, borderImgs[0].shape[0], borderImgs[0].shape[1] - right,
                                             borderImgs[0].shape[1]),
                         utilities.trimImage(borderImgs[3], 0, top, 0, borderImgs[3].shape[1])]

        cornerImgs[2] = [utilities.trimImage(borderImgs[1], 0, borderImgs[1].shape[0], 0, left),
                         utilities.trimImage(borderImgs[2], borderImgs[2].shape[0] - bottom, borderImgs[2].shape[0], 0,
                                             borderImgs[2].shape[1])]

        cornerImgs[3] = [utilities.trimImage(borderImgs[1], 0, borderImgs[1].shape[0], borderImgs[1].shape[1] - right,
                                             borderImgs[1].shape[1]),
                         utilities.trimImage(borderImgs[3], borderImgs[3].shape[0] - bottom, borderImgs[3].shape[0], 0,
                                             borderImgs[3].shape[1])]

        lowest = min(top, left)
        caps[0] = lowest * 2
        cornerMasks[0] = np.fromfunction(lambda i, j: np.minimum(caps[0], np.maximum(0, np.add(lowest, np.subtract(
            (np.subtract(top - 1, i)), np.subtract(left - 1, j))))), (top, left), dtype=np.int16)[:, :,
                         np.newaxis]  # make masks to smoothly merge corners

        lowest = min(top, right)
        caps[1] = lowest * 2
        cornerMasks[1] = np.fromfunction(lambda i, j: np.minimum(caps[1], np.maximum(0, np.add(lowest, np.subtract(
            (np.subtract(top - 1, i)), j)))), (top, right), dtype=np.int16)[:, :, np.newaxis]

        lowest = min(bottom, left)
        caps[2] = lowest * 2
        cornerMasks[2] = np.fromfunction(lambda i, j: np.minimum(caps[2], np.maximum(0, np.add(lowest, np.subtract(
            i, np.subtract(left - 1, j))))), (bottom, left), dtype=np.int16)[:, :, np.newaxis]

        lowest = min(bottom, right)
        caps[3] = lowest * 2
        cornerMasks[3] = np.fromfunction(lambda i, j: np.minimum(caps[3], np.maximum(0, np.add(lowest, np.subtract(
            i, j)))), (bottom, right), dtype=np.int16)[:, :, np.newaxis]

        count = 0
        for corner in cornerImgs:  # merge corners
            finalCorners[count] = weightedAverageMatrix(corner[0], corner[1], cornerMasks[count], caps[count])
            count += 1

        borderImgs[0] = pasteImage(pasteImage(borderImgs[0], finalCorners[0], 0, 0), finalCorners[1],
                                   borderImgs[0].shape[1] - right, 0)
        borderImgs[1] = pasteImage(pasteImage(borderImgs[1], finalCorners[2], 0, 0), finalCorners[3],
                                   borderImgs[1].shape[1] - right, 0)
        borderImgs[2] = utilities.trimImage(borderImgs[2], top, borderImgs[2].shape[0] - bottom, 0,
                                            borderImgs[2].shape[1])
        borderImgs[3] = utilities.trimImage(borderImgs[3], top, borderImgs[3].shape[0] - bottom, 0,
                                            borderImgs[3].shape[1])

        return cv.vconcat([borderImgs[0], cv.hconcat([borderImgs[2], trimmed, borderImgs[3]]),
                           borderImgs[1]])  # reattach borders to main image
    else:
        return sharpen(src, 3, 0.28, 1)


def sharpen(src, ksize, amount, iter):
    hsvSrc = cv.cvtColor(src, cv.COLOR_BGR2HSV_FULL)
    h, s, v = cv.split(hsvSrc)
    for _ in range(iter):
        v = img_as_ubyte(unsharp_mask(v, ksize, amount))
    return cv.cvtColor(cv.merge([h, s, v]), cv.COLOR_HSV2BGR_FULL)


def getMidpoint(line):  # get the midpoint of a line
    return [(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2]


def intersect(line1, line2):  # find interdection between 2 points
    if not (line1 and line2) or not (line1[0] and line1[1] and line2[0] and line2[1]):
        # if line missing, propogate None
        return None
    denom = (line2[1][1] - line2[0][1]) * (line1[1][0] - line1[0][0]) - (line2[1][0] - line2[0][0]) * (
            line1[1][1] - line1[0][1])
    if denom == 0:
        # print("Lines parallel")
        return None
    ua = ((line2[1][0] - line2[0][0]) * (line1[0][1] - line2[0][1]) - (line2[1][1] - line2[0][1]) * (
            line1[0][0] - line2[0][0])) / denom
    if ua < 0 or ua > 1:
        # print("Lines out of range 1")
        return None
    ub = ((line1[1][0] - line1[0][0]) * (line1[0][1] - line2[0][1]) - (line1[1][1] - line1[0][1]) * (
            line1[0][0] - line2[0][0])) / denom
    if ub < 0 or ub > 1:
        # print("Lines out of range 2")
        return None
    return [line1[0][0] + ua * (line1[1][0] - line1[0][0]), line1[0][1] + ua * (line1[1][1] - line1[0][1])]


def sumLinePixels(img, pt1, pt2, testImg=None):  # sum the number of lit pixels on a single line of a monocolour image
    rating = 0
    pointsList = list(zip(*skimage.draw.line(round(pt1[0]), round(pt1[1]), round(pt2[0]), round(pt2[1]))))
    for x, y in pointsList:
        with suppress(IndexError):
            rating += img[y - 1, x - 1]
            rating += img[y, x - 1]
            rating += img[y + 1, x - 1]
            rating += img[y - 1, x]
            rating += img[y, x]
            rating += img[y + 1, x]
            rating += img[y - 1, x + 1]
            rating += img[y, x + 1]
            rating += img[y, x + 1]
            # if testimg is not None:
            #     testimg[y, x] = img[y - 1, x - 1] / 2
            #     testimg[y, x] = img[y, x - 1] / 2
            #     testimg[y, x] = img[y + 1, x - 1] / 2
            #     testimg[y, x] = img[y - 1, x] / 2
            #     testimg[y, x] = img[y, x]
            #     testimg[y, x] = img[y + 1, x] / 2
            #     testimg[y, x] = img[y - 1, x + 1] / 2
            #     testimg[y, x] = img[y, x + 1] / 2
            #     testimg[y, x] = img[y, x + 1] / 2
    return rating, testImg  # / len(pointsList), testimg


def sumLinesPixels(img, line1, line2, debug,
                   show):  # sum the lit pixels on each combonation of 2 intersecting lines and return the highest
    innerImg = None
    outerImg = None
    # if show:
    # innerImg = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2BGR)#np.zeros(shape=[img.shape[0],img.shape[1],3], dtype=np.uint8)
    # outerImg = np.copy(innerImg)
    mid = intersect(line1, line2)
    if not mid: return None
    inner1, innerImg = sumLinePixels(img, line1[0], mid, innerImg)
    inner2, innerImg = sumLinePixels(img, line2[1], mid, innerImg)
    outer1, outerImg = sumLinePixels(img, line2[0], mid, outerImg)
    outer2, outerImg = sumLinePixels(img, line1[1], mid, outerImg)
    innerScore = inner1 + inner2
    outerScore = outer1 + outer2
    if debug:
        linesName = str(round(line1[0][0], 1)) + " " + str(round(line1[0][1], 1)) + " " + str(
            round(line1[1][0], 1)) + " " + str(round(line1[1][1], 1)) + " " \
                    + str(round(line2[0][0], 1)) + " " + str(round(line2[0][1], 1)) + " " + str(
            round(line2[1][0], 1)) + " " + str(round(line2[1][1], 1))
        print(linesName)
        print(mid)
        print("innerscore: " + str(innerScore))
        print("outerscore: " + str(outerScore))
    # if show:
    # innerImg = drawLineWithMid(img, line1[0], line2[1], mid)
    # outerImg = drawLineWithMid(img, line1[1], line2[0], mid)
    # cv.imshow("inner" + linesName, innerImg)
    # cv.imshow("outer" + linesName, outerImg)
    # cv.waitKey()
    if innerScore > outerScore:
        return [True, innerScore]
    return [False, outerScore]


def pasteImage(base, sprite, posX, posY):  # paste 1 image into another in the specified position
    base[round(posY):round(sprite.shape[0] + posY), round(posX):round(sprite.shape[1] + posX), :] = sprite
    return base


def reflectExtendBorder(src, top, bottom, left, right, options):
    while True:
        src = cv.copyMakeBorder(src, min(round(top), options[0]), min(round(bottom), options[1]),
                                min(round(left), options[2]), min(round(right), options[3]), cv.BORDER_REFLECT)
        top = max(top - options[0], 0)
        bottom = max(bottom - options[1], 0)
        left = max(left - options[2], 0)
        right = max(right - options[3], 0)
        options = [options[0] * 2, options[1] * 2, options[2] * 2, options[3] * 2]
        if top + bottom + left + right == 0:
            break
    return src


def addBlurredExtendBorder(src, top, bottom, left, right,
                           blur):  # add an extend border with a blur effect to smooth out varience
    # blurred = cv.GaussianBlur(src, (blur[0], blur[1]), 2)
    # blurred = cv.blur(src, (blur[0], blur[1]))
    blurred = cv.copyMakeBorder(src, blur[1], blur[1], blur[0], blur[0], cv.BORDER_WRAP)
    blurred = utilities.trimImage(cv.blur(blurred, (blur[0], blur[1])), blur[1], blurred.shape[0] - blur[1], blur[0],
                                  blurred.shape[1] - blur[0])
    blurred = cv.copyMakeBorder(blurred, round(top), round(bottom), round(left), round(right), cv.BORDER_REPLICATE)
    blurred = pasteImage(blurred, src, left, top)
    return blurred


def fixBadCorners(src):  # replace the corners of the image with median blurred versions.
    blurred = cv.copyMakeBorder(src, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH,
                                cv.BORDER_WRAP)
    blurred = cv.medianBlur(blurred, CORNER_FIX_STREGNTH * 2 + 1)
    src = pasteImage(src,
                     utilities.trimImage(blurred, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH * 3, CORNER_FIX_STREGNTH,
                                         CORNER_FIX_STREGNTH * 3), 0, 0)
    src = pasteImage(src, utilities.trimImage(blurred, blurred.shape[0] - CORNER_FIX_STREGNTH * 3,
                                              blurred.shape[0] - CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH,
                                              CORNER_FIX_STREGNTH * 3), 0, src.shape[0] - CORNER_FIX_STREGNTH * 2)
    src = pasteImage(src, utilities.trimImage(blurred, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH * 3,
                                              blurred.shape[1] - CORNER_FIX_STREGNTH * 3,
                                              blurred.shape[1] - CORNER_FIX_STREGNTH),
                     src.shape[1] - CORNER_FIX_STREGNTH * 2, 0)
    src = pasteImage(src, utilities.trimImage(blurred, blurred.shape[0] - CORNER_FIX_STREGNTH * 3,
                                              blurred.shape[0] - CORNER_FIX_STREGNTH,
                                              blurred.shape[1] - CORNER_FIX_STREGNTH * 3,
                                              blurred.shape[1] - CORNER_FIX_STREGNTH),
                     src.shape[1] - CORNER_FIX_STREGNTH * 2, src.shape[0] - CORNER_FIX_STREGNTH * 2)
    return src


def trimNegLine(pt1, pt2):  # trim a line so it no longer goes below 0
    disX = pt2[0] - pt1[0]
    disY = pt2[1] - pt1[1]
    # print(disX)
    # print(disY)
    Xratio = 0
    Yratio = 0
    if pt1[0] < 0:
        Xratio = abs(pt1[0]) / disX
    if pt1[1] < 0:
        Yratio = abs(pt1[1]) / disY
    # print(Xratio)
    # print(Yratio)
    ratio = max(Xratio, Yratio)
    return [round((disX * ratio) + pt1[0], 5), round((disY * ratio) + pt1[1], 5)]


def trimLongLine(pt1, pt2, maxX, maxY):  # trim a line to specified maximums
    disX = pt2[0] - pt1[0]
    disY = pt2[1] - pt1[1]
    # print(disX)
    # print(disY)
    overshootX = max(pt1[0] - maxX, 0)
    overshootY = max(pt1[1] - maxY, 0)
    # print(overshootX)
    # print(overshootY)
    Xratio = 0
    Yratio = 0
    if overshootX > 0:
        Xratio = overshootX / disX
    if overshootY > 0:
        Yratio = overshootY / disY
    # print(Xratio)
    # print(Yratio)
    ratio = max(abs(Xratio), abs(Yratio))
    return [round((ratio * disX) + pt1[0], 5), round((ratio * disY) + pt1[1], 5)]


def trimLine(pt1, pt2, maxX, maxY):  # trim a line on both ends to limit it to the bounds of an image
    # print(maxX)
    # print(maxY)
    if pt1[0] < 0 or pt1[1] < 0:
        pt1 = trimNegLine(pt1, pt2)
    if pt2[0] < 0 or pt2[1] < 0:
        pt2 = trimNegLine(pt2, pt1)
    # print(pt1)
    # print(pt2)
    if pt1[0] > maxX or pt1[1] > maxY:
        pt1 = trimLongLine(pt1, pt2, maxX, maxY)
    if pt2[0] > maxX or pt2[1] > maxY:
        pt2 = trimLongLine(pt2, pt1, maxX, maxY)
    return pt1, pt2


def detectLines(img, threshold, side, debug,
                show):  # find the 8-point lines in an edge detected image and find the best one
    # side: 0 = top, 1 = bottom, 2 = left, 3 = right
    if side <= 1:  # set the angle limitation and correct axis for the side
        baseAngle = np.pi * 0.5
        axis = 1
        offAxis = 0
    else:
        baseAngle = np.pi * 1
        axis = 0
        offAxis = 1
    if side % 2 == 0:  # set whether we are looking for the highest or lowest lines, based on side
        op = operator.lt
    else:
        op = operator.gt
    if debug:
        print("side: " + str(side))

    lines = cv.HoughLines(img, 0.25, np.pi / 2880, round(threshold), None, 0, 0,
                          baseAngle - ANGLE_TOLERANCE,
                          baseAngle + ANGLE_TOLERANCE)  # get lines
    if show:
        allImg = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2BGR)
        proImg = np.copy(allImg)
        drawProImg = False
    maxCorner = None
    minCorner = None
    processedLines = []
    prunedLines = []
    highScore = None
    corners = None
    mid = None
    count = 0

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (Decimal(x0 + 10000 * (-b)), Decimal(y0 + 10000 * (a)))
            pt2 = (Decimal(x0 - 10000 * (-b)), Decimal(y0 - 10000 * (a)))
            pt1, pt2 = trimLine(pt1, pt2, img.shape[1],
                                img.shape[0])  # get line in cartesean coords and limit it to the bounds of the image
            if pt1[offAxis] > pt2[offAxis]:  # sort the points based on the axis not being tested
                buffer = pt1
                pt1 = pt2
                pt2 = buffer
            if show:
                int1 = [int(pt1[0]), int(pt1[1])]
                int2 = [int(pt2[0]), int(pt2[1])]
                cv.line(allImg, int1, int2, (0, 255, 0), 1, cv.LINE_AA)
            # if debug:
            #     print("low point: " + str(pt1))
            #     print("high point: " + str(pt2))
            if (img.shape[axis] == pt1[offAxis] or pt1[offAxis] == 0) and \
                    (img.shape[axis] == pt2[offAxis] or pt2[
                        offAxis] == 0):  # check if both points touch the far ends of the image, reject if they don't
                processedLines.append([pt1, pt2])
                if minCorner is None or op(pt1[axis], processedLines[minCorner][0][
                    axis]):  # keep track of the line furthest towards the card edge on each side
                    minCorner = count
                if maxCorner is None or op(pt2[axis], processedLines[maxCorner][1][axis]):
                    maxCorner = count
                count += 1
            elif debug:
                print("line failed: " + str([pt1, pt2]))
        if debug:
            print("minCorner: " + str(minCorner))
            print("maxCorner: " + str(maxCorner))
        if len(processedLines) > 0:
            if maxCorner == minCorner:  # if the furthest out line on each side is the same, use that line.
                corners = processedLines[maxCorner]
                mid = getMidpoint(corners)
            else:  # otherwise test each line combonation to find the one that matches the true line the most
                minRange = (processedLines[minCorner][0][axis],
                            processedLines[maxCorner][0][axis])  # filter out lines not within the 2 furthest out lines
                maxRange = (processedLines[maxCorner][1][axis], processedLines[minCorner][1][axis])
                if debug:
                    print("low-side range: " + str(minRange))
                    print("high-side range: " + str(maxRange))
                    print(processedLines)
                for line in processedLines:  # filter out lines not within the 2 furthest out lines
                    if max(minRange) >= line[0][axis] >= min(minRange) or \
                            max(maxRange) >= line[1][axis] >= min(maxRange):
                        if show:
                            int1 = [int(line[0][0]), int(line[0][1])]
                            int2 = [int(line[1][0]), int(line[1][1])]
                            cv.line(proImg, int1, int2, (0, 255, 0), 1, cv.LINE_AA)
                            drawProImg = True
                        prunedLines.append(line)
                processedLines.sort(key=lambda prunedLines: prunedLines[0][axis])
                for i in range(0, len(prunedLines)):
                    if 1 != 0:
                        for j in range(0, len(prunedLines)):
                            if i > j:
                                score = sumLinesPixels(img, prunedLines[i], prunedLines[j], debug,
                                                       show)  # get the amount of each line that matches
                                if score:
                                    if not highScore or score[1] > highScore[1]:
                                        highScore = score + [i, j]  # save the best line and if it's inwards or outwards
                    score = sumLinePixels(img, prunedLines[i][0], prunedLines[i][1])  # check single lines too
                    if debug:
                        print("score for solo line: " + str(i) + " " + str(score[0]))
                    if score:
                        if not highScore or score[0] > highScore[1]:
                            highScore = [True, score[0], i, i]
                if highScore[2] == highScore[3]:  # if a single line is best, use it
                    if debug:
                        print("Chose line " + str(highScore[2]))
                    corners = prunedLines[highScore[2]]
                    mid = getMidpoint(corners)
                else:
                    highI = prunedLines[highScore[
                        2]]  # otherwise use a combonation of 2 lines to find the best, storing the intersection as the mid point
                    highJ = prunedLines[highScore[3]]
                    if highScore[0]:
                        if debug:
                            print("Chose Inner for lines " + str(highScore[2]) + " " + str(highScore[3]))
                        corners = [highI[0], highJ[1]]
                    else:
                        if debug:
                            print("Chose Outer for lines " + str(highScore[2]) + " " + str(highScore[3]))
                        corners = [highJ[0], highI[1]]
                    mid = intersect(highI, highJ)
                    if debug:
                        print("mid: " + str(mid))
    if corners and mid:
        if show:
            cv.imshow("all lines for side " + str(side), allImg)
            if drawProImg:
                # pass
                cv.imshow("possible lines for side " + str(side), proImg)
        return corners, mid
    return None, None


def correctMid(mid, midAxis, minCorner, maxCorner,
               ownCorners):  # check if the midpoint is outside the border, change to a single line if it is
    if mid and minCorner and maxCorner and ownCorners and (
            mid[midAxis] <= minCorner + 1 or mid[midAxis] >= maxCorner - 1):
        print("mid out of bounds, attempting to correct")
        return getMidpoint(ownCorners)
    return mid


def getLines(clean, edge, debug, show):  # get the 9 points of the border
    clean = cv.medianBlur(clean, 3)  # use a median blur to fill in small gaps
    edges = cv.Canny(clean, 1700, 3120, True, 5)  # 25, 1200, True, 5)
    global timesRun
    timesRun += 1
    if show:
        cv.imshow("clean " + str(timesRun), clean)
        cv.imshow("edges " + str(timesRun), edges)

    threshold = DEFAULT_H_THRESHOLD * edges.shape[1]  # set line threshold based on image resolution
    upCorners, upMid = detectLines(utilities.trimImage(np.copy(edges), MIN_LINE_EDGE, edge[0], 0, edges.shape[1]),
                                   threshold, 0, debug, show)
    lowCorners, lowMid = detectLines(
        utilities.trimImage(np.copy(edges), edges.shape[0] - edge[1], edges.shape[0] - MIN_LINE_EDGE, 0,
                            edges.shape[1]), threshold, 1, debug, show)

    threshold = DEFAULT_V_THRESHOLD * edges.shape[0]
    leftCorners, leftMid = detectLines(utilities.trimImage(np.copy(edges), 0, edges.shape[0], MIN_LINE_EDGE, edge[2]),
                                       threshold, 2,
                                       debug, show)
    rightCorners, rightMid = detectLines(
        utilities.trimImage(np.copy(edges), 0, edges.shape[0], edges.shape[1] - edge[3],
                            edges.shape[1] - MIN_LINE_EDGE), threshold, 3, debug,
        show)

    with suppress(TypeError):
        upCorners[0][1] += MIN_LINE_EDGE
        upCorners[1][1] += MIN_LINE_EDGE
    with suppress(TypeError):
        upMid[1] += MIN_LINE_EDGE
    with suppress(TypeError):
        leftCorners[0][0] += MIN_LINE_EDGE
        leftCorners[1][0] += MIN_LINE_EDGE
    with suppress(TypeError):
        leftMid[0] += MIN_LINE_EDGE
    with suppress(TypeError):
        lowCorners[0][1] += edges.shape[0] - edge[1]
        lowCorners[1][1] += edges.shape[0] - edge[1]
    with suppress(TypeError):
        lowMid[1] += edges.shape[0] - edge[1]
    with suppress(TypeError):
        rightCorners[0][0] += edges.shape[1] - edge[3]
        rightCorners[1][0] += edges.shape[1] - edge[3]
    with suppress(TypeError):
        rightMid[0] += edges.shape[1] - edge[3]

    return upCorners, upMid, lowCorners, lowMid, leftCorners, leftMid, rightCorners, rightMid


def getPointsFromLines(clean, edge, debug, show, manual, src):
    global timesRun
    upLeft, upRight, lowLeft, lowRight = None, None, None, None
    first = True
    upCorners, upMid, lowCorners, lowMid, leftCorners, leftMid, rightCorners, rightMid = getLines(clean, edge, debug,
                                                                                                  show)
    if not (upCorners and lowCorners and leftCorners and rightCorners):
        print("ERROR: Could not find 4 edges")
        first = False
        if show:
            lines = np.copy(src)
            with suppress(TypeError):
                lines = drawLineWithMid(lines, upCorners[0], upCorners[1], upMid)
            with suppress(TypeError):
                lines = drawLineWithMid(lines, lowCorners[0], lowCorners[1], lowMid)
            with suppress(TypeError):
                lines = drawLineWithMid(lines, leftCorners[0], leftCorners[1], leftMid)
            with suppress(TypeError):
                lines = drawLineWithMid(lines, rightCorners[0], rightCorners[1], rightMid)
            cv.imshow("Detected Lines " + str(timesRun), lines)
    # sanity checks in case of a midpoint in the boundry
    with suppress(TypeError):
        upMid = correctMid(upMid, 0, leftCorners[0][0], rightCorners[0][0], upCorners)
    with suppress(TypeError):
        lowMid = correctMid(lowMid, 0, leftCorners[1][0], rightCorners[1][0], lowCorners)
    with suppress(TypeError):
        leftMid = correctMid(leftMid, 1, upCorners[0][1], lowCorners[0][1], leftCorners)
    with suppress(TypeError):
        rightMid = correctMid(rightMid, 1, upCorners[1][1], lowCorners[1][1], rightCorners)

    if debug:
        print("upCorners: " + str(upCorners))
        print("lowCorners: " + str(lowCorners))
        print("leftCorners: " + str(leftCorners))
        print("rightCorners: " + str(rightCorners))
        print("upMid: " + str(upMid))
        print("lowMid: " + str(lowMid))
        print("leftMid: " + str(leftMid))
        print("rightMid: " + str(rightMid))
    # get corners by intersecting lines
    with suppress(TypeError):
        upRight = intersect((upCorners[1], upMid), (rightCorners[0], rightMid))
    with suppress(TypeError):
        upLeft = intersect((upCorners[0], upMid), (leftCorners[0], leftMid))
    with suppress(TypeError):
        lowLeft = intersect((lowCorners[0], lowMid), (leftCorners[1], leftMid))
    with suppress(TypeError):
        lowRight = intersect((lowCorners[1], lowMid), (rightCorners[1], rightMid))
    if debug:
        print("UpperLeft: " + str(upLeft))
        print("UpperRight: " + str(upRight))
        print("LowerLeft: " + str(lowLeft))
        print("LowerRight: " + str(lowRight))
    if first and not (upLeft and upRight and lowLeft and lowRight):
        print("ERROR: Lines do not intersect")
        if show:
            box = drawBox(src, upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight)
            cv.imshow("Detected Lines " + str(timesRun), box)

    if manual:
        upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight = taskbarFuncs.CustomBordersUI(src, upLeft,
                                                                                                            upMid,
                                                                                                            upRight,
                                                                                                            leftMid,
                                                                                                            rightMid,
                                                                                                            lowLeft,
                                                                                                            lowMid,
                                                                                                            lowRight)
    return upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight


def filterImage(hsvClean, adaptive, filter, exCorrect, debug, show):
    if adaptive:  # an apative method that tried to take a chunk of the border and make a filter based on it.
        borderBase = cv.vconcat(
            [utilities.trimImage(hsvClean, MIN_ELEC_AVG_RANGE, MAX_ELEC_AVG_RANGE, MIN_ELEC_AVG_RANGE,
                                 hsvClean.shape[1] - MIN_ELEC_AVG_RANGE),
             utilities.trimImage(hsvClean, hsvClean.shape[0] - MAX_ELEC_AVG_RANGE,
                                 hsvClean.shape[0] - MIN_ELEC_AVG_RANGE, MIN_ELEC_AVG_RANGE,
                                 hsvClean.shape[1] - MIN_ELEC_AVG_RANGE)])

        borderBaseV = cv.hconcat(
            [utilities.trimImage(hsvClean, MIN_ELEC_AVG_RANGE, hsvClean.shape[0] - MIN_ELEC_AVG_RANGE,
                                 MIN_ELEC_AVG_RANGE, MAX_ELEC_AVG_RANGE),
             utilities.trimImage(hsvClean, MIN_ELEC_AVG_RANGE, hsvClean.shape[0] - MIN_ELEC_AVG_RANGE,
                                 hsvClean.shape[1] - MAX_ELEC_AVG_RANGE,
                                 hsvClean.shape[1] - MIN_ELEC_AVG_RANGE)])
        borderBaseV = cv.transpose(borderBaseV)  # make an image from the assumed borders of the image

        borderBase = cv.hconcat([borderBase, borderBaseV])

        highPerc = np.percentile(borderBase, 99,
                                 (0, 1))  # get the range of the image's HSV values, ignoring 1% outliers)
        lowPerc = np.percentile(borderBase, 1, (0, 1))
        taskbarFuncs.lowHue = int(lowPerc[0] - ELEC_HUE_VAR)
        taskbarFuncs.hiHue = int((highPerc[0] + ELEC_HUE_VAR))
        taskbarFuncs.lowSat = int(lowPerc[1] - ELEC_SAT_VAR)
        taskbarFuncs.hiSat = int(highPerc[1] + ELEC_SAT_VAR)
        taskbarFuncs.lowVal = int(lowPerc[2] - ELEC_VAL_VAR)
        taskbarFuncs.hiVal = int(highPerc[2] + ELEC_VAL_VAR)
        clean = cv.inRange(hsvClean, (taskbarFuncs.lowHue, taskbarFuncs.lowSat, taskbarFuncs.lowVal),
                           (taskbarFuncs.hiHue, taskbarFuncs.hiSat,
                            taskbarFuncs.hiVal))  # apply the range filter to the values
        if show:
            cv.imshow("adaptive clean", clean)
        return clean
    else:  # a simpler method that tries to match a large range of yellow
        if exCorrect and checkImageHue(hsvClean, (0, 20, 170), (24, 170, 237)):
            if debug:
                print("EX CORRECT RED")
            taskbarFuncs.lowHue = 27
            taskbarFuncs.hiHue = filter[1]
            taskbarFuncs.lowSat = 0
            taskbarFuncs.hiSat = filter[3]
            taskbarFuncs.lowVal = filter[4]
            taskbarFuncs.hiVal = filter[5]
        elif exCorrect and checkImageHue(hsvClean, (10, 0, 24), (123, 36, 180)):
            if debug:
                print("EX CORRECT METAL")  # force fallback
            taskbarFuncs.lowHue = 0
            taskbarFuncs.hiHue = 180
            taskbarFuncs.lowSat = 0
            taskbarFuncs.hiSat = 255
            taskbarFuncs.lowVal = 0
            taskbarFuncs.hiVal = 255
        elif exCorrect and checkImageHue(hsvClean, (0, 0, 124), (180, 36, 255)):
            if debug:
                print("EX CORRECT WHITE")
            taskbarFuncs.lowHue = 49
            taskbarFuncs.hiHue = filter[1]
            taskbarFuncs.lowSat = 0
            taskbarFuncs.hiSat = filter[3]
            taskbarFuncs.lowVal = filter[4]
            taskbarFuncs.hiVal = 156
        else:
            taskbarFuncs.lowHue = filter[0]
            taskbarFuncs.hiHue = filter[1]
            taskbarFuncs.lowSat = filter[2]
            taskbarFuncs.hiSat = filter[3]
            taskbarFuncs.lowVal = filter[4]
            taskbarFuncs.hiVal = filter[5]

        clean = cv.inRange(hsvClean, (taskbarFuncs.lowHue, taskbarFuncs.lowSat, taskbarFuncs.lowVal),
                           (taskbarFuncs.hiHue, taskbarFuncs.hiSat, taskbarFuncs.hiVal))
        if show:
            cv.imshow("basic clean", clean)
        return clean


def calculateOuterAndInnerPoint(pnt, middle,
                                extraSpace):  # from point and midspace, get a matching outer point for mesh transform
    return [
        [pnt[0] + ((pnt[0] - middle[0]) * (extraSpace[0] * 2)),
         pnt[1] + ((pnt[1] - middle[1]) * (extraSpace[1] * 2))], pnt]


def drawLineWithMid(img, pt1, pt2, mid):  # draw a 3 point line
    if pt1 and mid:
        cv.line(img, [round(pt1[0]), round(pt1[1])],
                [round(mid[0]), round(mid[1])], (0, 255, 0), 1, cv.LINE_AA)
    if pt2 and mid:
        cv.line(img, [round(pt2[0]), round(pt2[1])],
                [round(mid[0]), round(mid[1])], (0, 255, 0), 1, cv.LINE_AA)
    return img


def drawBox(img, upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight):  # draw an 8 point box
    drawLineWithMid(img, upLeft, upRight, upMid)
    drawLineWithMid(img, lowLeft, lowRight, lowMid)
    drawLineWithMid(img, upLeft, lowLeft, leftMid)
    drawLineWithMid(img, upRight, lowRight, rightMid)
    return img


def findMinMax(points, xmax, ymax):
    xCol = None
    yCol = None
    for col in range(len(points[0])):
        if xmax:
            if yCol is None or points[1][col] > points[1][yCol]:
                yCol = col
            elif points[1][col] == points[1][yCol]:
                if ymax and points[0][col] > points[0][yCol]:
                    yCol = col
                elif (not ymax) and points[0][col] < points[0][yCol]:
                    yCol = col
        else:
            if yCol is None or points[1][col] < points[1][yCol]:
                yCol = col
            elif points[1][col] == points[1][yCol]:
                if ymax and points[0][col] > points[0][yCol]:
                    yCol = col
                elif (not ymax) and points[0][col] < points[0][yCol]:
                    yCol = col
        if ymax:
            if xCol is None or points[0][col] > points[0][xCol]:
                xCol = col
            elif points[0][col] == points[0][xCol]:
                if xmax and points[1][col] > points[1][xCol]:
                    xCol = col
                elif (not xmax) and points[1][col] < points[1][xCol]:
                    xCol = col
        else:
            if xCol is None or points[0][col] < points[0][xCol]:
                xCol = col
            elif points[0][col] == points[0][xCol]:
                if xmax and points[1][col] > points[1][xCol]:
                    xCol = col
                elif (not xmax) and points[1][col] < points[1][xCol]:
                    xCol = col
    if xCol is not None and yCol is not None:
        return (points[0][yCol], points[1][xCol])
    return None


def handleCorner(basePos, corners, xmax, ymax):
    rounded = [round(basePos[1]), round(basePos[0])]
    locCorner = np.vstack(
        np.nonzero(utilities.trimImage(corners, rounded[0] - 2, rounded[0] + 3, rounded[1] - 2, rounded[1] + 3)))
    minMaxCor = findMinMax(locCorner, xmax, ymax)
    if minMaxCor is None:
        return [0, 0]
    else:
        xside = 1
        if xmax:
            xside = 2
        yside = 1
        if ymax:
            yside = 2
        finalCor = (minMaxCor[0] + rounded[0] - yside, minMaxCor[1] + rounded[1] - xside)
        finalCor = [(finalCor[1] + basePos[0]) / 2, (finalCor[0] + basePos[1]) / 2]
        return [Decimal((finalCor[0] - basePos[0]) / 2), Decimal((finalCor[1] - basePos[1]) / 2)]


def harrisCorrection(img, upLeft, upRight, lowLeft, lowRight, harSet):
    simg = sharpen(img, 5, 0.28, 1)
    gimg = np.float32(cv.cvtColor(simg, cv.COLOR_BGR2GRAY))
    _, simg, _2 = cv.split(np.float32(cv.cvtColor(simg, cv.COLOR_BGR2HSV)))
    gCorners = cv.cornerHarris(gimg, int(harSet[0]), int(harSet[1]), float(harSet[2]))
    sCorners = cv.cornerHarris(simg, int(harSet[3]), int(harSet[4]), float(harSet[5]))
    corners = np.logical_or(np.greater(sCorners, 0.1 * sCorners.max()), np.greater(gCorners, 0.074 * gCorners.max()))

    corTL = handleCorner(upLeft, corners, False, False)
    corTR = handleCorner(upRight, corners, True, False)
    corBL = handleCorner(lowLeft, corners, False, True)
    corBR = handleCorner(lowRight, corners, True, True)

    return corTL, corTR, corBL, corBR


def processImage(baseImg, cleanImg, border, trim, edge, res, mask, manual, filter, cusFilter, elec, exCorrect, reflect,
                 replace, fixCuts, harSet, blur, cb, doner, debug=False, show=False):
    src = cv.imread(cv.samples.findFile(baseImg))
    if src is None:
        print('Base Image at ' + baseImg + ' Not Found, skipping')
        return
    if not exCorrect:
        fixBadCorners(src)
    if not edge:
        edgeSize = round(DEFAULT_BORDER_TOLERANCE * src.shape[0])  # set default value for expected border size
        edge = [edgeSize, edgeSize, edgeSize, edgeSize]
    if cleanImg:
        clean = cv.imread(cv.samples.findFile(cleanImg))
        if clean is None:
            print('Clean Image at ' + cleanImg + ' Not Found, attempting with base image')
            clean = src
    else:
        clean = src

    hsvClean = cv.cvtColor(clean, cv.COLOR_BGR2HSV)  # make a HSV version of clean for filtering

    # H, S, V = cv.split(hsvClean)
    # S = S.astype(float) * 4
    # S[S > 255] = 255
    # clean = cv.cvtColor(cv.merge([H, S.astype(np.uint8), V]), cv.COLOR_HSV2BGR)
    # upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight = getPointsFromLines(clean, edge,
    #                                                                                           debug, show,
    #                                                                                           manual, src)
    # if not (upLeft and upMid and upRight and leftMid and rightMid and lowLeft and lowMid and lowRight):
    #     print("ERROR: 4 lines not found in image " + baseImg)
    #     if show:
    #         cv.waitKey()
    #     return None

    electric = False
    if elec:
        electric = checkImageHue(hsvClean, (20, 70, 0), (30, 255, 255))
    if debug:
        print("image size: " + str(src.shape))
        print("edges " + str(edge))
        print("Electric: " + str(electric))
    filtered = filterImage(hsvClean, electric, filter, exCorrect, debug, show)
    if cusFilter:
        filtered = taskbarFuncs.HSVFilterUI(
            clean)  # set custom filter is enabled, does a normal filter first to get a sane default
    upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight = getPointsFromLines(filtered, edge, debug,
                                                                                              show, manual, src)
    if not (upLeft and upMid and upRight and leftMid and rightMid and lowLeft and lowMid and lowRight):
        print("Trying again with other filter...")
        filtered = filterImage(hsvClean, not electric, filter, exCorrect, debug, show)
        upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight = getPointsFromLines(filtered, edge, debug,
                                                                                                  show, manual, src)
        if not (upLeft and upMid and upRight and leftMid and rightMid and lowLeft and lowMid and lowRight):
            print("Trying again without a filter...")
            H, S, V = cv.split(hsvClean)
            S = S.astype(float) * 4
            S[S > 255] = 255
            clean = cv.cvtColor(cv.merge([H, S.astype(np.uint8), V]), cv.COLOR_HSV2BGR)
            upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight = getPointsFromLines(clean, edge,
                                                                                                      debug, show,
                                                                                                      manual, src)
            if not (upLeft and upMid and upRight and leftMid and rightMid and lowLeft and lowMid and lowRight):
                print("ERROR: 4 lines not found in image " + baseImg)
                if show:
                    cv.waitKey()
                return None

    if not border:
        border = [0, 0, 0, 0]
    # set the amount of space outside the card frame to keep
    extraSpace = [max(border[0], border[1]) + Decimal(0.05), max(border[2], border[3]) + Decimal(0.05)]
    if debug:
        print("extraSpace: " + str(extraSpace))
    # the offset from the frames current position compared to before the extra was added
    offsetX = Decimal(round(src.shape[0] * extraSpace[0]))
    offsetY = Decimal(round(src.shape[1] * extraSpace[1]))

    if show and not manual:
        edges = drawBox(np.copy(src), upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight)
        cv.imshow("4 main lines pre cor", edges)

    if harSet:
        ULChange, URChange, LLChange, LRChange = harrisCorrection(src, upLeft, upRight, lowLeft, lowRight, harSet)
    else:
        ULChange, URChange, LLChange, LRChange = ([0, 0], [0, 0], [0, 0], [0, 0])

    upLeft = upLeft + ULChange
    upRight = upRight + URChange
    lowLeft = lowLeft + LLChange
    lowRight = lowRight + LRChange

    upMidMulti = Decimal((upMid[0] - upLeft[0]) / (upRight[0] - upLeft[0]))
    lowMidMulti = Decimal((lowMid[0] - lowLeft[0]) / (lowRight[0] - lowLeft[0]))
    leftMidMulti = Decimal((leftMid[1] - upLeft[1]) / (lowLeft[1] - upLeft[1]))
    rightMidMulti = Decimal((rightMid[1] - upRight[1]) / (lowRight[1] - upRight[1]))

    upMid = [upMid[0], upMid[1] + (URChange[1] * upMidMulti) + (ULChange[1] * (1 - upMidMulti))]
    lowMid = [lowMid[0], lowMid[1] + (LRChange[1] * lowMidMulti) + (LLChange[1] * (1 - lowMidMulti))]
    leftMid = [leftMid[0] + (LLChange[0] * leftMidMulti) + (ULChange[0] * (1 - leftMidMulti)), leftMid[1]]
    rightMid = [rightMid[0] + (LLChange[0] * rightMidMulti) + (ULChange[0] * (1 - rightMidMulti)), rightMid[1]]

    if show and not manual:
        edges = drawBox(np.copy(src), upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight)
        cv.imshow("4 main lines post cor", edges)

    offUpLeft = [upLeft[0] + offsetX, upLeft[1] + offsetY]
    offUpRight = [upRight[0] + offsetX, upRight[1] + offsetY]
    offLowLeft = [lowLeft[0] + offsetX, lowLeft[1] + offsetY]
    offLowRight = [lowRight[0] + offsetX, lowRight[1] + offsetY]
    cardWidth = max(upRight[0] - upLeft[0], lowRight[0] - lowLeft[0])
    cardHeight = max(lowRight[1] - upRight[1], lowLeft[1] - upLeft[1])

    offUpMid = [upMid[0] + offsetX, upMid[1] + offsetY]
    offLowMid = [lowMid[0] + offsetX, lowMid[1] + offsetY]
    offLeftMid = [leftMid[0] + offsetX, leftMid[1] + offsetY]
    offRightMid = [rightMid[0] + offsetX, rightMid[1] + offsetY]
    midPoint = [(offLeftMid[0] + offRightMid[0]) / 2, (offUpMid[1] + offLowMid[1]) / 2]

    if debug:
        print("UpperLeft: " + str(upLeft))
        print("UpperRight: " + str(upRight))
        print("LowerLeft: " + str(lowLeft))
        print("LowerRight: " + str(lowRight))
        print("middlePoint: " + str(midPoint))
        print("cardWidth: " + str(cardWidth))
        print("cardHeight: " + str(cardHeight))
        print("border: " + str(border))
        print("res: " + str(res))

    if res:  # set the target card's size
        border = [res[1] * Decimal(border[0]), res[1] * Decimal(border[1]), res[0] * Decimal(border[2]),
                  res[0] * Decimal(border[3])]
        targetWidth = round(res[0] - (border[2] + border[3]))
        targetHeight = round(res[1] - (border[0] + border[1]))
    else:
        targetWidth = cardWidth
        targetHeight = cardHeight
        border = [Decimal((cardHeight * border[0])), Decimal((cardHeight * border[1])),
                  Decimal((cardWidth * border[2])),
                  Decimal((cardWidth * border[3]))]
    # set the target position
    targetOffsetX = Decimal(targetWidth * extraSpace[0])
    targetOffsetY = Decimal(targetHeight * extraSpace[1])
    targetCard = [targetOffsetY, targetOffsetY + targetHeight, targetOffsetX, targetOffsetX + targetWidth]
    targetMid = (
    Decimal(targetWidth * (Decimal(0.5) + extraSpace[0])), Decimal(targetHeight * (Decimal(0.5) + extraSpace[1])))

    targetUpMid = [targetWidth * upMidMulti + targetOffsetX, targetCard[0]]
    targetLowMid = [targetWidth * lowMidMulti + targetOffsetX, targetCard[1]]
    targetLeftMid = [targetCard[2], targetHeight * leftMidMulti + targetOffsetY]
    targetRightMid = [targetCard[3], targetHeight * rightMidMulti + targetOffsetY]

    if debug:
        print("border: " + str(border))
        print("extraSpace: " + str(extraSpace))
        print("targetwidth: " + str(targetWidth))
        print("targethieght: " + str(targetHeight))
        print("targetmid: " + str(targetMid))
        print("targetcard: " + str(targetCard))
        print("targetUpMid: " + str(targetUpMid))
        print("targetLowMid: " + str(targetLowMid))
        print("targetLeftMid: " + str(targetLeftMid))
        print("targetRightMid: " + str(targetRightMid))
        print("targetOffsetX: " + str(targetOffsetX))
        print("targetOffsetY: " + str(targetOffsetY))
    # create arrays in scipy compatible format
    # np.set_printoptions(suppress=True)
    srcP = np.array(
        calculateOuterAndInnerPoint(offUpLeft, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(offLeftMid, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(offLowLeft, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(offLowMid, midPoint, extraSpace) +
        [midPoint] +
        calculateOuterAndInnerPoint(offUpMid, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(offUpRight, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(offRightMid, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(offLowRight, midPoint, extraSpace), dtype="float64")

    # srcP = np.round(srcP)

    dstP = np.array(
        calculateOuterAndInnerPoint([targetCard[2], targetCard[0]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint(targetLeftMid, targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetCard[2], targetCard[1]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint(targetLowMid, targetMid, extraSpace) +
        [targetMid] +
        calculateOuterAndInnerPoint(targetUpMid, targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetCard[3], targetCard[0]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint(targetRightMid, targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetCard[3], targetCard[1]], targetMid, extraSpace), dtype="float64")

    dstP = np.round(dstP)

    if debug:
        print(srcP)
        print(dstP)
        print("offsetX: " + str(offsetX))
        print("offsetY: " + str(offsetY))

    # expand the border to fill the extra space
    if reflect:
        bordersize = [round(min(upLeft[1], upMid[1], upRight[1]) - 2),
                      round(src.shape[0] - (max(lowLeft[1], lowMid[1], lowRight[1]) + 3)),
                      round(min(upLeft[0], leftMid[0], lowLeft[0]) - 2),
                      round(src.shape[1] - (max(upRight[0], rightMid[0], lowRight[0]) + 3))]
        if show:
            cv.imshow("preflect", src)
            print(bordersize)
        bordered = reflectExtendBorder(src, round(offsetY), round(offsetY), round(offsetX), round(offsetX), bordersize)
    else:
        bordered = addBlurredExtendBorder(src, round(offsetY), round(offsetY), round(offsetX), round(offsetX), blur)

    # cv.imshow("prewarp", bordered)
    tform = PiecewiseAffineTransform()
    tform.estimate(dstP, srcP)
    # mesh transform the image into shape
    warped = img_as_ubyte(warp(bordered, tform, output_shape=(
        round(targetHeight + targetOffsetY * 2), round(targetWidth + targetOffsetX * 2))))
    # cv.imshow("precut", warped)

    if trim:
        warped = utilities.trimImage(math.floor(targetCard[0]), math.ceil(targetCard[1]), math.floor(targetCard[2]),
                                     math.ceil(targetCard[3]))
    # calculate how much of the new image to trim off/add on to create the correct size border
    adjustNeeded = [round((targetCard[0] - border[0]) * Decimal(-1)),
                    round((targetCard[1] + border[1]) - warped.shape[0]),
                    round((targetCard[2] - border[2]) * Decimal(-1)),
                    round((targetCard[3] + border[3]) - warped.shape[1])]
    if warped.shape[0] + adjustNeeded[0] + adjustNeeded[1] < res[1]:
        if (targetCard[0] - border[0]) * Decimal(-1) - adjustNeeded[0] < (targetCard[1] + border[1]) - warped.shape[0] - \
                adjustNeeded[1]:
            adjustNeeded[0] += 1
        else:
            adjustNeeded[1] += 1
    elif warped.shape[0] + adjustNeeded[0] + adjustNeeded[1] > res[1]:
        if (targetCard[0] - border[0]) * Decimal(-1) - adjustNeeded[0] > (targetCard[1] + border[1]) - warped.shape[0] - \
                adjustNeeded[1]:
            adjustNeeded[0] -= 1
        else:
            adjustNeeded[1] -= 1

    if warped.shape[1] + adjustNeeded[2] + adjustNeeded[3] < res[0]:
        if (targetCard[2] - border[2]) * Decimal(-1) - adjustNeeded[2] < (targetCard[3] + border[3]) - warped.shape[1] - \
                adjustNeeded[3]:
            adjustNeeded[2] += 1
        else:
            adjustNeeded[3] += 1
    elif warped.shape[1] + adjustNeeded[2] + adjustNeeded[3] > res[0]:
        if (targetCard[2] - border[2]) * Decimal(-1) - adjustNeeded[2] > (targetCard[3] + border[3]) - warped.shape[1] - \
                adjustNeeded[3]:
            adjustNeeded[2] -= 1
        else:
            adjustNeeded[3] -= 1

    if debug:
        print("adjustNeeded: " + str(adjustNeeded))
        print("preadjust shape:" + str(warped.shape))
        print("preround adjust: " + str([(targetCard[0] - border[0]) * Decimal(-1),
                                         (targetCard[1] + border[1]) - warped.shape[0],
                                         (targetCard[2] - border[2]) * Decimal(-1),
                                         (targetCard[3] + border[3]) - warped.shape[1]]))
    if any(side < 0 for side in adjustNeeded):
        warped = warped[max(0, adjustNeeded[0] * -1):len(warped) + min(0, adjustNeeded[1]),
                 max(0, adjustNeeded[2] * -1):len(warped[0]) + min(0, adjustNeeded[3])]
    if any(side > 0 for side in adjustNeeded):
        if reflect:
            warped = reflectExtendBorder(warped, max(0, adjustNeeded[0]), max(0, adjustNeeded[1]),
                                         max(0, adjustNeeded[2]), max(0, adjustNeeded[3]),
                                         [border[0] - max(0, adjustNeeded[0] - 2),
                                          border[1] - max(0, adjustNeeded[1] - 2),
                                          border[2] - max(0, adjustNeeded[2] - 2),
                                          border[3] - max(0, adjustNeeded[3] - 2)])  # border - adjust needed
        else:
            warped = addBlurredExtendBorder(warped, max(0, adjustNeeded[0]), max(0, adjustNeeded[1]),
                                            max(0, adjustNeeded[2]),
                                            max(0, adjustNeeded[3]), blur)

    if replace:
        warped = pasteBorders(warped, replace, fixCuts)

    warped = smoothBorders(warped, round(border[0]), round(border[1]), round(border[2]), round(border[3]), blur, cb,
                           fixCuts)

    if doner is not None:
        LBGRd = cv.cvtColor(cv.cvtColor(doner, cv.COLOR_BGR2LAB), cv.COLOR_LAB2LBGR)
        LBGRw = cv.cvtColor(cv.cvtColor(warped, cv.COLOR_BGR2LAB), cv.COLOR_LAB2LBGR)
        PCA = cv.normalize(PCAColorTransfer(LBGRw, LBGRd), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        PCA = cv.cvtColor(cv.cvtColor(PCA, cv.COLOR_LBGR2LAB), cv.COLOR_LAB2BGR) / 255
        fDoner = doner / 255
        aPCA = PCAColorTransfer(warped / 255, fDoner)

        # oldLuma = getLuma(PCA)[:, :, np.newaxis]
        # sTransfer = setSaturation(PCA, getChroma(fDoner))
        # sTransfer = addLightness(sTransfer, oldLuma - getLuma(sTransfer)[:, :, np.newaxis])
        cTransfer = addLightness(fDoner,
                                 utilities.getLuminosity(aPCA)[:, :, np.newaxis] - utilities.getLuminosity(fDoner)[:, :,
                                                                                   np.newaxis])
        aTransfer = addLightness(fDoner,
                                 utilities.getLuminosity(PCA)[:, :, np.newaxis] - utilities.getLuminosity(fDoner)[:, :,
                                                                                  np.newaxis])
        # sTransfer = boostContrast((utilities.clipToOne(sTransfer) * 255).astype(np.uint8), doner)
        cTransfer = (utilities.clipToOne(cTransfer) * 255).astype(np.uint8)
        aTransfer = (utilities.clipToOne(aTransfer) * 255).astype(np.uint8)
        PCA = (utilities.clipToOne(PCA) * 255).astype(np.uint8)
        aPCA = (utilities.clipToOne(aPCA) * 255).astype(np.uint8)
        aPCA = boostContrast(aPCA, doner)
        PCA = boostContrast(PCA, doner)
        cBoosted = boostContrast(cTransfer, doner)
        aTransfer = boostContrast(aTransfer, doner)
        images = [PCA,
                  aPCA,
                  # sTransfer,
                  cBoosted,
                  aTransfer,
                  cTransfer,
                  warped
                  ]
    else:
        images = [warped]

    # apply an alpha mask if provided, to make the corners
    if mask:
        for count in range(len(images)):
            images[count - 1] = maskcorners.processMask(images[count - 1], mask)
    if show:
        for image in images:
            cv.imshow("outputed " + str(count), image)
        cv.waitKey()
        return None
    return images


def processMultiArg(arg, numNeeded, decimal):
    arg = arg.split(",")
    argList = []
    for num in arg:
        if decimal:
            argList.append(Decimal(num))
        else:
            argList.append(int(num))
    if len(argList) != numNeeded:
        raise ValueError("var must have exactly" + str(numNeeded) + "numbers")
    return argList


def processArgs(inputText):
    input = os.path.join(os.getcwd(), "input")
    clean = None
    output = os.path.join(os.getcwd(), "output")
    border = None
    trim = False
    edge = None
    show = False
    debug = False
    res = [734, 1024]
    mask = None
    manual = False
    filter = [20, 30, 120, 255, 190, 255]
    cusFilter = False
    elec = True
    blur = [25, 25]
    cb = None
    trans = None
    transFil = [21, 30, 87, 191, 173, 255]  # 28
    harSet = None
    dHarSet = None
    reflect = False
    exCorrect = False
    replace = None
    fixCuts = False

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input" + inputText)
    parser.add_argument("-o", "--Output", help="Set Output" + inputText)
    parser.add_argument("-c", "--Clean", help="Set a secondary Clean " + inputText + "Image used for edge detection\n"
                                                                                     "Default: None")
    parser.add_argument("-d", "--Debug", help="Enable debug prints default False" + inputText)
    parser.add_argument("-b", "--BorderSize",
                        help="Set the size of the 4 borders to trim to \n"
                             "Accepts 4 numbers seperated by commas, as so: 't,b,l,r'\n"
                             "May result in transparent edges. defaults to 0,0,0,0")
    parser.add_argument("-e", "--EdgeFiltering",
                        help="customises the filtering of lines too far away from the edge. \n"
                             "Accepts 4 numbers seperated by commas, as so: 't,b,l,r'. \n"
                             "default is Y res dependent, 27 at 800")
    parser.add_argument("-t", "--Trim", help="If enabled, trim all the original border off of the image. default False")
    parser.add_argument("-s", "--Show", help="show images instead of saving them. default False")
    parser.add_argument("-r", "--Resolution", help="The resolution of the output, as so: 'x,y'.\n"
                                                   "Default: existing card size")
    parser.add_argument("-m", "--Mask", help="Mask the card using the provided mask.\n"
                                             "good for rounded corners.")
    parser.add_argument("-ma", "--Manual", help="Detect the edges manually. default: False")
    parser.add_argument("-f", "--Filter",
                        help="set the HSV filter to use, in the horder of HiHue, LoHue, HiSat, LoSat, HiVal, LowVal. default: False")
    parser.add_argument("-cf", "--CustomFilter",
                        help="Bring up the filter menu to customise the filter used. default: 20,30,120,255,190,255")
    parser.add_argument("-ed", "--ElectricDetection",
                        help="Automatically switch to adaptive detection on yellow cards. default: True")
    parser.add_argument("-bb", "--BorderBlur", help="how much to blur the border, as so x,y. default: 25,25")
    parser.add_argument("-cb", "--CleanBorder",
                        help="avoid sharpening the border, cleaning it instead. enable by setting tolerance like so t,b,l,r.")
    parser.add_argument("-ct", "--ColorTransfer",
                        help="Transfer Colors from a render with with file or a similarly named one from this folder. default None")
    parser.add_argument("-tf", "--TransferFilter",
                        help="set the filter for correcting the borders of the image used for transfering. default 21,30,87,191,173,255")
    parser.add_argument("-ha", "--Harris",
                        help="Use Harris corner detection to improve accuracy. arranged as such: greyBlock,greyKsize,greyK,satBlock,satKsize,satK")
    parser.add_argument("-dh", "--DonerHarris",
                        help="Use Harris corner detection to improve accuracy of the doner. arranged as such: greyBlock,greyKsize,greyK,satBlock,satKsize,satK")
    parser.add_argument("-rb", "--ReflectBorder",
                        help="Reflect the border when extending instead of expanding it. default: False")
    parser.add_argument("-ex", "--EXcorrect",
                        help="correct the hue detection based on the hue, used for exs. default: False")
    parser.add_argument("-br", "--BorderReplace",
                        help="path to borders to replace the image's borders with. default: None")
    parser.add_argument("-fc", "--FixCuts",
                        help="fix high value circles of cut borders. argument is the value to use. default: False")

    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Clean:
        clean = args.Clean
    if args.Output:
        output = args.Output
    if args.Debug:
        debug = args.Debug
    if args.Show:
        show = args.Show
    if args.BorderSize:
        border = processMultiArg(args.BorderSize, 4, True)
    if args.Trim:
        trim = bool(args.Trim)
    if args.EdgeFiltering:
        edge = processMultiArg(args.EdgeFiltering, 4, False)
    if args.Resolution:
        res = processMultiArg(args.Resolution, 2, False)
    if args.Mask:
        mask = args.Mask
    if args.Manual:
        manual = args.Manual
    if args.Filter:
        filter = processMultiArg(args.Filter, 6, False)
    if args.CustomFilter:
        cusFilter = args.CustomFilter
    if args.ElectricDetection:
        elec = args.ElectricDetection
    if args.BorderBlur:
        blur = processMultiArg(args.BorderBlur, 2, False)
    if args.CleanBorder:
        cb = processMultiArg(args.CleanBorder, 4, False)
    if args.ColorTransfer:
        trans = args.ColorTransfer
    if args.TransferFilter:
        transFil = processMultiArg(args.TransferFilter, 6, False)
    if args.Harris:
        harSet = processMultiArg(args.Harris, 6, True)
    if args.DonerHarris:
        dHarSet = processMultiArg(args.DonerHarris, 6, True)
    if args.ReflectBorder:
        reflect = args.ReflectBorder
    if args.EXcorrect:
        exCorrect = args.EXcorrect
    if args.BorderReplace:
        replace = args.BorderReplace
    if args.FixCuts:
        fixCuts = int(args.FixCuts)
    return input, clean, output, border, trim, edge, res, mask, manual, filter, cusFilter, elec, exCorrect, reflect, replace, fixCuts, harSet, dHarSet, blur, cb, trans, transFil, debug, show


def getCacheFilename(transPath, inputPath):
    iBase = os.path.basename(inputPath)
    numStr = re.search("^0*(\d+)", iBase)
    tPath = os.path.dirname(transPath)
    if numStr:
        return os.path.join(tPath, numStr[1] + ".png")
    else:
        return os.path.join(tPath, iBase)


def resolveImage(input, clean, output, border, trim, edge, res, mask, manual, filter, cusFilter, elec, exCorrect,
                 reflect, replace, fixCuts, harSet, dHarSet, blur, cb, trans, transFil, debug, show):
    doner = None
    if trans:
        if os.path.isfile(trans):
            print("processing " + trans)
            doner = processImage(trans, None, border, False, [45, 45, 45, 45], res, None, False, transFil, cusFilter,
                                 False, False, reflect, None, False, dHarSet, [3, 3], None, None, debug, False)
            if doner:
                doner = doner[0]
            else:
                print("doner file " + trans + "failed to find 4 lines, skipping transfer")
        else:
            print("doner file " + trans + "not found, skipping transfer.")
    print("processing " + input)
    images = processImage(input, clean, border, trim, edge, res, mask, manual, filter, cusFilter, elec, exCorrect,
                          reflect, replace, fixCuts, harSet, blur, cb, doner, debug, show)
    if images is not None:
        count = 1
        for image in images:
            if len(images) == 1:
                cv.imwrite(output, image)
            else:
                cv.imwrite(re.sub("\.png$", str(count) + ".png", output), image)
                count += 1


def processFolder(input, clean, output, border, trim, edge, res, mask, manual, filter, cusFilter, elec, exCorrect,
                  reflect, replace, fixCuts, harSet, dHarSet, blur, cb, trans, transFil, debug, show):
    with suppress(FileExistsError):
        os.mkdir(output)
    with os.scandir(input) as entries:
        for entry in entries:
            cleanPath = None
            inputPath = os.path.join(input, entry.name)
            outputPath = os.path.join(output, entry.name)
            if trans:
                transPath = os.path.join(trans, entry.name)
            else:
                transPath = None
            if clean:
                cleanPath = os.path.join(clean, entry.name)
            if os.path.isfile(inputPath) and entry.name != "Place Images Here":
                if transPath:
                    transPath = getCacheFilename(transPath, inputPath)
                resolveImage(inputPath, cleanPath, outputPath, border, trim, edge, res, mask, manual, filter, cusFilter,
                             elec, exCorrect, reflect, replace, fixCuts, harSet, dHarSet, blur, cb, transPath, transFil,
                             debug, show)
            elif os.path.isdir(inputPath):
                processFolder(inputPath, cleanPath, outputPath, border, trim, edge, res, mask, manual, filter,
                              cusFilter, elec, exCorrect, reflect, replace, fixCuts, harSet, dHarSet, blur, cb,
                              transPath, transFil, debug, show)


def main():
    input, clean, output, border, trim, edge, res, mask, manual, filter, cusFilter, elec, exCorrect, reflect, replace, fixCuts, harSet, dHarSet, blur, cb, trans, transFil, debug, show = processArgs(
        "folder")
    if os.path.isfile(input):
        if trans:
            transPath = getCacheFilename(trans, input)
        else:
            transPath = None
        resolveImage(input, clean, output, border, trim, edge, res, mask, manual, filter, cusFilter, elec, exCorrect,
                     reflect, replace, fixCuts, harSet, dHarSet, blur, cb, transPath, transFil, debug, show)
    elif os.path.isdir(input):
        processFolder(input, clean, output, border, trim, edge, res, mask, manual, filter, cusFilter, elec, exCorrect,
                      reflect, replace, fixCuts, harSet, dHarSet, blur, cb, trans, transFil, debug, show)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()
