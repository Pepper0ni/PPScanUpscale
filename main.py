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
import sys

ANGLE_TOLERANCE = 0.01  # maximum variance from a straight line the line detector will accept. TODO find scale
DEFAULT_BORDER_TOLERANCE = 0.0327272727273  # sets how far into an image to look for a border, multiplied by y coord
DEFAULT_H_THRESHOLD = 0.1 # sets how strict the horizontal line detector is, multiplied by X coord
DEFAULT_V_THRESHOLD = 0.1 # sets how strict the verticle line detector is, multiplied by y coord
MIN_LINE_EDGE = 3 #sets how many pixels from the edge to exclude from line detection
MIN_ELEC_AVG_RANGE = 5 #sets the distance from the edge to exclude from apative border detection
MAX_ELEC_AVG_RANGE = 30 #sets the distance away from the edge to include in apative border detection
ELEC_HUE_VAR = 1 #sets the variance in hue for the adaptive border detection to accept
ELEC_SAT_VAR = 3 #sets the variance in saturation for the adaptive border detection to accept
ELEC_VAL_VAR = 1 #sets the variance in value for the adaptive border detection to accept
CORNER_FIX_STREGNTH = 5 #sets how large an area, and how strong a blur, to use for the corner fixing
lineTog = True
TRACKBAR_WINDOW_NAME = "Manual point set"


def getMidpoint(line): #get the midpoint of a line
    return [(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2]


def intersect(line1, line2): #find interdection between 2 points
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


def sumLinePixels(img, pt1, pt2, testImg=None): #sum the number of lit pixels on a single line of a monocolour image
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
    return rating, testImg #/ len(pointsList), testimg


def sumLinesPixels(img, line1, line2, debug, show): #sum the lit pixels on each combonation of 2 intersecting lines and return the highest
    innerImg = None
    outerImg = None
    if show:
        innerImg = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2BGR)#np.zeros(shape=[img.shape[0],img.shape[1],3], dtype=np.uint8)
        outerImg = np.copy(innerImg)
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
        innerImg = drawLineWithMid(img, line1[0], line2[1], mid)
        outerImg = drawLineWithMid(img, line1[1], line2[0], mid)
        #cv.imshow("inner" + linesName, innerImg)
        #cv.imshow("outer" + linesName, outerImg)
        #cv.waitKey()
    if innerScore > outerScore:
        return [True, innerScore]
    return [False, outerScore]

def pasteImage(base, sprite, posX, posY): #paste 1 image into another in the specified position
    base[round(posY):round(sprite.shape[0] + posY), round(posX):round(sprite.shape[1] + posX), :] = sprite
    return base

def addBlurredExtendBorder(src, top, bottom, left, right): #add an extend border with a blur effect to smooth out varience
    blurred = cv.blur(src, (25, 25), 0)
    blurred = cv.copyMakeBorder(blurred, round(top), round(bottom), round(left), round(right), cv.BORDER_REPLICATE)
    blurred = pasteImage(blurred, src, left, top)
    return blurred

def fixBadCorners(src): #replace the corners of the image with median blurred versions.
    blurred = cv.copyMakeBorder(src, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH, cv.BORDER_WRAP)
    blurred = cv.medianBlur(blurred, CORNER_FIX_STREGNTH*2+1)
    src = pasteImage(src, trimImage(blurred, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH*3, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH*3), 0, 0)
    src = pasteImage(src, trimImage(blurred, blurred.shape[0]-CORNER_FIX_STREGNTH*3, blurred.shape[0]-CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH*3), 0, src.shape[0]-CORNER_FIX_STREGNTH*2)
    src = pasteImage(src, trimImage(blurred, CORNER_FIX_STREGNTH, CORNER_FIX_STREGNTH*3, blurred.shape[1]-CORNER_FIX_STREGNTH*3, blurred.shape[1]-CORNER_FIX_STREGNTH), src.shape[1]-CORNER_FIX_STREGNTH*2, 0)
    src = pasteImage(src, trimImage(blurred, blurred.shape[0]-CORNER_FIX_STREGNTH*3, blurred.shape[0]-CORNER_FIX_STREGNTH, blurred.shape[1]-CORNER_FIX_STREGNTH*3, blurred.shape[1]-CORNER_FIX_STREGNTH), src.shape[1]-CORNER_FIX_STREGNTH*2, src.shape[0]-CORNER_FIX_STREGNTH*2)
    return src

def trimNegLine(pt1, pt2): #trim a line so it no longer goes below 0
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
    return [round((disX * ratio) + pt1[0], 5), round((disY * ratio) + pt1[1],5)]


def trimLongLine(pt1, pt2, maxX, maxY): #trim a line to specified maximums
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


def trimLine(pt1, pt2, maxX, maxY): #trim a line on both ends to limit it to the bounds of an image
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

def detectLines(img, threshold, side, debug, show): #find the 8-point lines in an edge detected image and find the best one
    # side: 0 = top, 1 = bottom, 2 = left, 3 = right
    if side <= 1: #set the angle limitation and correct axis for the side
        baseAngle = 0.5
        axis = 1
        offAxis = 0
    else:
        baseAngle = 1
        axis = 0
        offAxis = 1
    if side % 2 == 0: #set whether we are looking for the highest or lowest lines, based on side
        op = operator.lt
    else:
        op = operator.gt
    if debug:
        print("side: " + str(side))

    lines = cv.HoughLines(img, 0.25, np.pi / 2880, round(threshold), None, 0, 0,
                          np.pi * (baseAngle - ANGLE_TOLERANCE),
                          np.pi * (baseAngle + ANGLE_TOLERANCE))
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
            pt1, pt2 = trimLine(pt1, pt2, img.shape[1], img.shape[0])
            if pt1[offAxis] > pt2[offAxis]:
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
                    (img.shape[axis] == pt2[offAxis] or pt2[offAxis] == 0):
                processedLines.append([pt1, pt2])
                if minCorner is None or op(pt1[axis], processedLines[minCorner][0][axis]):
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
            if maxCorner == minCorner:
                corners = processedLines[maxCorner]
                mid = getMidpoint(corners)
            else:
                minRange = (processedLines[minCorner][0][axis], processedLines[maxCorner][0][axis])
                maxRange = (processedLines[maxCorner][1][axis], processedLines[minCorner][1][axis])
                if debug:
                    print("low-side range: " + str(minRange))
                    print("high-side range: " + str(maxRange))
                    print(processedLines)
                for line in processedLines:
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
                                score = sumLinesPixels(img, prunedLines[i], prunedLines[j], show, debug)
                                if score:
                                    if not highScore or score[1] > highScore[1]:
                                        highScore = score + [i, j]
                    score = sumLinePixels(img, prunedLines[i][0], prunedLines[i][1])
                    if debug:
                        print("score for solo line: " + str(i) + " " + str(score[0]))
                    if score:
                        if not highScore or score[0] > highScore[1]:
                            highScore = [True, score[0], i, i]
                if highScore[2] == highScore[3]:
                    if debug:
                        print("Chose line " + str(highScore[2]))
                    corners = prunedLines[highScore[2]]
                    mid = getMidpoint(corners)
                else:
                    highI = prunedLines[highScore[2]]
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
                cv.imshow("possible lines for side " + str(side), proImg)
        return corners, mid
    # threshold -= 20
    # print("looping: " + str(threshold))
    # if threshold <= 0:
    return None, None

def correctMid(mid, midAxis, minCorner, maxCorner, ownCorners):
    if mid[midAxis] <= minCorner or mid[midAxis] >= maxCorner:
        print("mid out of bounds, attempting to correct")
        return getMidpoint(ownCorners)
    return mid

def getPoints(edges, edge, debug, show):
    threshold = DEFAULT_H_THRESHOLD * edges.shape[1]
    upCorners, upMid = detectLines(trimImage(np.copy(edges), MIN_LINE_EDGE, edge[0], 0, edges.shape[1]), threshold, 0, debug, show)
    lowCorners, lowMid = detectLines(trimImage(np.copy(edges), edges.shape[0] - edge[1], edges.shape[0] - MIN_LINE_EDGE, 0, edges.shape[1]), threshold, 1, debug,
                                     show)

    threshold = DEFAULT_V_THRESHOLD * edges.shape[0]
    leftCorners, leftMid = detectLines(trimImage(np.copy(edges), 0, edges.shape[0] - MIN_LINE_EDGE, MIN_LINE_EDGE, edge[2]), threshold, 2,
                                       debug, show)
    rightCorners, rightMid = detectLines(
        trimImage(np.copy(edges), 0, edges.shape[0], edges.shape[1] - edge[3], edges.shape[1] - MIN_LINE_EDGE), threshold, 3, debug,
        show)

    if upCorners and lowCorners and leftCorners and rightCorners:
        upCorners[0][1] += MIN_LINE_EDGE
        upCorners[1][1] += MIN_LINE_EDGE
        upMid[1] += MIN_LINE_EDGE
        leftCorners[0][0] += MIN_LINE_EDGE
        leftCorners[1][0] += MIN_LINE_EDGE
        leftMid[0] += MIN_LINE_EDGE
        lowCorners[0][1] += edges.shape[0] - edge[1]
        lowCorners[1][1] += edges.shape[0] - edge[1]
        lowMid[1] += edges.shape[0] - edge[1]
        rightCorners[0][0] += edges.shape[1] - edge[3]
        rightCorners[1][0] += edges.shape[1] - edge[3]
        rightMid[0] += edges.shape[1] - edge[3]

    return upCorners, upMid, lowCorners, lowMid, leftCorners, leftMid, rightCorners, rightMid

def getAdaptiveClean(hsvSrc):
    borderBase = cv.vconcat([trimImage(hsvSrc, MIN_ELEC_AVG_RANGE, MAX_ELEC_AVG_RANGE, MIN_ELEC_AVG_RANGE, hsvSrc.shape[1] - MIN_ELEC_AVG_RANGE),
                             trimImage(hsvSrc, hsvSrc.shape[0] - MAX_ELEC_AVG_RANGE, hsvSrc.shape[0] - MIN_ELEC_AVG_RANGE, MIN_ELEC_AVG_RANGE,hsvSrc.shape[1] - MIN_ELEC_AVG_RANGE)])

    borderBaseV = cv.hconcat([trimImage(hsvSrc, MIN_ELEC_AVG_RANGE, hsvSrc.shape[0] - MIN_ELEC_AVG_RANGE, MIN_ELEC_AVG_RANGE, MAX_ELEC_AVG_RANGE),
                              trimImage(hsvSrc, MIN_ELEC_AVG_RANGE, hsvSrc.shape[0] - MIN_ELEC_AVG_RANGE, hsvSrc.shape[1] - MAX_ELEC_AVG_RANGE, hsvSrc.shape[1] - MIN_ELEC_AVG_RANGE)])
    borderBaseV = cv.transpose(borderBaseV)

    borderBase = cv.hconcat([borderBase, borderBaseV])

    highPerc = np.percentile(borderBase, 99, (0, 1))
    lowPerc = np.percentile(borderBase, 1, (0, 1))
    global lowHue
    global hiHue
    global lowSat
    global hiSat
    global lowVal
    global hiVal
    lowHue = int(lowPerc[0] - ELEC_HUE_VAR)
    hiHue = int((highPerc[0] + ELEC_HUE_VAR))
    lowSat = int(lowPerc[1] - ELEC_SAT_VAR)
    hiSat = int(highPerc[1] + ELEC_SAT_VAR)
    lowVal = int(lowPerc[2] - ELEC_VAL_VAR)
    hiVal = int(highPerc[2] + ELEC_VAL_VAR)
    return cv.inRange(hsvSrc, (lowHue, lowSat, lowVal), (hiHue, hiSat, hiVal))

def trimImage(img, fromTop, newBot, fromLeft, newRight):
    return img[fromTop:newBot, fromLeft:newRight]


def calculateOuterAndInnerPoint(pnt, middle, extraSpace):
    return [
        [pnt[0] + ((pnt[0] - middle[0]) * (extraSpace[0] * 2)), pnt[1] + ((pnt[1] - middle[1]) * (extraSpace[1] * 2))],
        pnt]

def drawLineWithMid(img, pt1, pt2, mid):
    cv.line(img, [round(pt1[0]), round(pt1[1])],
            [round(mid[0]), round(mid[1])], (0, 255, 0), 1, cv.LINE_AA)
    cv.line(img, [round(pt2[0]), round(pt2[1])],
            [round(mid[0]), round(mid[1])], (0, 255, 0), 1, cv.LINE_AA)
    return img

def drawBox(img, upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight):
    drawLineWithMid(img, upLeft, upRight, upMid)
    drawLineWithMid(img, lowLeft, lowRight, lowMid)
    drawLineWithMid(img, upLeft, lowLeft, leftMid)
    drawLineWithMid(img, upRight, lowRight, rightMid)
    return img

def TLXcallbackTrackbar(val):
    global posDict
    posDict["TLX"] = val

def TLYcallbackTrackbar(val):
    global posDict
    posDict["TLY"] = val
    
def TMXcallbackTrackbar(val):
    global posDict
    posDict["TMX"] = val
    
def TMYcallbackTrackbar(val):
    global posDict
    posDict["TMY"] = val
    
def TRXcallbackTrackbar(val):
    global posDict
    posDict["TRX"] = val
    
def TRYcallbackTrackbar(val):
    global posDict
    posDict["TRY"] = val
    
def MLXcallbackTrackbar(val):
    global posDict
    posDict["MLX"] = val

def MLYcallbackTrackbar(val):
    global posDict
    posDict["MLY"] = val
    
def MRXcallbackTrackbar(val):
    global posDict
    posDict["MRX"] = val
    
def MRYcallbackTrackbar(val):
    global posDict
    posDict["MRY"] = val
    
def BLXcallbackTrackbar(val):
    global posDict
    posDict["BLX"] = val
    
def BLYcallbackTrackbar(val):
    global posDict
    posDict["BLY"] = val

def BMXcallbackTrackbar(val):
    global posDict
    posDict["BMX"] = val
    
def BMYcallbackTrackbar(val):
    global posDict
    posDict["BMY"] = val
    
def BRXcallbackTrackbar(val):
    global posDict
    posDict["BRX"] = val
    
def BRYcallbackTrackbar(val):
    global posDict
    posDict["BRY"] = val

def lowHueTrackbar(val):
    global lowHue
    global hiHue
    lowHue = val
    lowHue = min(hiHue - 1, lowHue)
    cv.setTrackbarPos("LowHue", TRACKBAR_WINDOW_NAME, lowHue)


def hiHueTrackbar(val):
    global lowHue
    global hiHue
    hiHue = val
    hiHue = max(hiHue, lowHue + 1)
    cv.setTrackbarPos("hiHue", TRACKBAR_WINDOW_NAME, hiHue)


def lowSatTrackbar(val):
    global lowSat
    global hiSat
    lowSat = val
    lowSat = min(hiSat - 1, lowSat)
    cv.setTrackbarPos("LowSat", TRACKBAR_WINDOW_NAME, lowSat)


def hiSatTrackbar(val):
    global lowSat
    global hiSat
    hiSat = val
    hiSat = max(hiSat, lowSat + 1)
    cv.setTrackbarPos("hiSat", TRACKBAR_WINDOW_NAME, hiSat)


def lowValTrackbar(val):
    global lowVal
    global hiVal
    lowVal = val
    lowVal = min(hiVal - 1, lowVal)
    cv.setTrackbarPos("lowVal", TRACKBAR_WINDOW_NAME, lowVal)


def hiValTrackbar(val):
    global lowVal
    global hiVal
    hiVal = val
    hiVal = max(hiVal, lowVal + 1)
    cv.setTrackbarPos("hiVal", TRACKBAR_WINDOW_NAME, hiVal)

def toggleLines(a,b):
    global lineTog
    if lineTog:
        lineTog = False
    else:
        lineTog = True

declared = False
lowHue = 0
hiHue = 180
lowSat = 0
hiSat = 255
lowVal = 0
hiVal = 255

def processImage(baseImg, cleanImg, border, trim, edge, res, mask, manual, filter, debug=False, show=False):
    global declared
    src = cv.imread(cv.samples.findFile(baseImg))
    if src is None:
        print('Base Image at ' + baseImg + ' Not Found, skipping')
        return
    fixBadCorners(src)
    if manual:
        cv.namedWindow(TRACKBAR_WINDOW_NAME)
        global posDict
        if not declared:
            posDict = {"TLX": 0, "TLY": 0, "TMX": round(src.shape[1]/2), "TMY": 0, "TRX": src.shape[1], "TRY": 0, "MLX": 0,
                       "MLY": round(src.shape[0]/2), "MRX": src.shape[1], "MRY": round(src.shape[0]/2), "BLX": 0,
                       "BLY": src.shape[0], "BMX": round(src.shape[1]/2), "BMY": src.shape[0], "BRX": src.shape[1], "BRY": src.shape[0]}
            for point, value in posDict.items():
                cv.createTrackbar(point, TRACKBAR_WINDOW_NAME, value, max(src.shape[1], src.shape[0]), globals()[point+"callbackTrackbar"])
            cv.createButton("Toggle Lines", toggleLines)
            cv.namedWindow("manual preview", cv.WINDOW_NORMAL)
            declared = True
        blank = np.zeros(shape=[3, 600], dtype=np.uint8)
        while True:
            if lineTog:
                boxSrc = drawBox(np.copy(src), [posDict["TLX"], posDict["TLY"]], [posDict["TMX"], posDict["TMY"]], [posDict["TRX"], posDict["TRY"]],
                                 [posDict["MLX"], posDict["MLY"]], [posDict["MRX"], posDict["MRY"]],
                                 [posDict["BLX"], posDict["BLY"]], [posDict["BMX"], posDict["BMY"]], [posDict["BRX"], posDict["BRY"]])
            else:
                boxSrc = src
            cv.imshow(TRACKBAR_WINDOW_NAME, blank)
            cv.imshow("manual preview", boxSrc)
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break
            upLeft = [Decimal(posDict["TLX"]), Decimal(posDict["TLY"])]
            upMid = [Decimal(posDict["TMX"]), Decimal(posDict["TMY"])]
            upRight = [Decimal(posDict["TRX"]), Decimal(posDict["TRY"])]
            leftMid = [Decimal(posDict["MLX"]), Decimal(posDict["MLY"])]
            rightMid = [Decimal(posDict["MRX"]), Decimal(posDict["MRY"])]
            lowLeft = [Decimal(posDict["BLX"]), Decimal(posDict["BLY"])]
            lowMid = [Decimal(posDict["BMX"]), Decimal(posDict["BMY"])]
            lowRight = [Decimal(posDict["BRX"]), Decimal(posDict["BRY"])]
    else:
        if cleanImg:
            clean = cv.imread(cv.samples.findFile(cleanImg))
        else:
            clean = src

        if clean is None:
            print('Clean Image at ' + cleanImg + ' Not Found, attempting with base image')
            clean = src
        if debug:
            print("image size: " + str(src.shape))
        global lowHue
        global hiHue
        global lowSat
        global hiSat
        global lowVal
        global hiVal

        hsvClean = cv.cvtColor(clean, cv.COLOR_BGR2HSV)
        elecCheck = cv.inRange(hsvClean, (20, 70, 0), (30, 255, 255))
        electric = False
        if np.sum(elecCheck) > src.shape[0] * src.shape[1] * 128:
            electric = True
        if debug:
            print("Electric: " + str(electric))
        if electric:
            clean = getAdaptiveClean(hsvClean)
        else:
            lowHue = 20
            hiHue = 30
            lowSat = 120
            hiSat = 255
            lowVal = 190
            hiVal = 255
            clean = cv.inRange(hsvClean, (lowHue, lowSat, lowVal), (hiHue, hiSat, hiVal))

        if filter:
            cv.namedWindow(TRACKBAR_WINDOW_NAME)
            if not declared:
                cv.createTrackbar("hiHue", TRACKBAR_WINDOW_NAME, hiHue, 180, hiHueTrackbar)
                cv.createTrackbar("LowHue", TRACKBAR_WINDOW_NAME, lowHue, 255, lowHueTrackbar)
                cv.createTrackbar("HiSat", TRACKBAR_WINDOW_NAME, hiSat, 255, hiSatTrackbar)
                cv.createTrackbar("LoSat", TRACKBAR_WINDOW_NAME, lowSat, 255, lowSatTrackbar)
                cv.createTrackbar("HiVal", TRACKBAR_WINDOW_NAME, hiVal, 255, hiValTrackbar)
                cv.createTrackbar("LoVal", TRACKBAR_WINDOW_NAME, lowVal, 255, lowValTrackbar)
                cv.createButton("Toggle Lines", toggleLines)
                cv.namedWindow("manual preview", cv.WINDOW_NORMAL)
                declared = True
            blank = np.zeros(shape=[3, 600], dtype=np.uint8)

            while True:
                clean = cv.inRange(hsvClean, (lowHue, lowSat, lowVal), (hiHue, hiSat, hiVal))
                preview = cv.medianBlur(clean, 3)
                cv.imshow(TRACKBAR_WINDOW_NAME, blank)
                cv.imshow("manual preview", preview)
                key = cv.waitKey(30)
                if key == ord('q') or key == 27:
                    break
                    
        clean = cv.medianBlur(clean, 3)

        edges = cv.Canny(clean, 25, 1200, True, 5)
        if show:
            cv.imshow("clean", clean)
            cv.imshow("edges", edges)
        if not edge:
            edgeSize = round(DEFAULT_BORDER_TOLERANCE * src.shape[0])
            edge = [edgeSize, edgeSize, edgeSize, edgeSize]
        if debug:
            print("edges " + str(edge))

        upCorners, upMid, lowCorners, lowMid, leftCorners, leftMid, rightCorners, rightMid = getPoints(edges, edge, debug, show)
        if not (upCorners and lowCorners and leftCorners and rightCorners):
            if not electric:
                clean = getAdaptiveClean(hsvClean)
                clean = cv.medianBlur(clean, 3)
                edges = cv.Canny(clean, 25, 1200, True, 5)
                upCorners, upMid, lowCorners, lowMid, leftCorners, leftMid, rightCorners, rightMid = getPoints(
                edges, edge, debug, show)
                if not (upCorners and lowCorners and leftCorners and rightCorners):
                    print("could not find 4 edges")
                    if show:
                        cv.imshow("adaptive clean", clean)
                        cv.imshow("adaptive edges", edges)
                        cv.waitKey()
                    return
            else:
                clean = cv.inRange(hsvClean, (20, 180, 200), (30, 255, 255))
                clean = cv.medianBlur(clean, 3)
                edges = cv.Canny(clean, 25, 1200, True, 5)
                upCorners, upMid, lowCorners, lowMid, leftCorners, leftMid, rightCorners, rightMid = getPoints(
                edges, edge, debug, show)
                if not (upCorners and lowCorners and leftCorners and rightCorners):
                    print("could not find 4 edges")
                    if show:
                        cv.imshow("basic clean", clean)
                        cv.imshow("basic edges", edges)
                        cv.waitKey()
                    return

        if lowCorners and rightCorners and upCorners and leftCorners:
            #sanity checks in case of a midpoint in the boundry
            upMid = correctMid(upMid, 0, leftCorners[0][0], rightCorners[0][0], upCorners)
            lowMid = correctMid(lowMid, 0, leftCorners[1][0], rightCorners[1][0], lowCorners)
            leftMid = correctMid(leftMid, 1, upCorners[0][1], lowCorners[0][1], leftCorners)
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

            upLeft = intersect((upCorners[0], upMid), (leftCorners[0], leftMid))
            upRight = intersect((upCorners[1], upMid), (rightCorners[0], rightMid))
            lowLeft = intersect((lowCorners[0], lowMid), (leftCorners[1], leftMid))
            lowRight = intersect((lowCorners[1], lowMid), (rightCorners[1], rightMid))

            if not (upLeft and upRight and lowLeft and lowRight):
                print("ERROR: Lines do not intersect")
                print("UpperLeft: " + str(upLeft))
                print("UpperRight: " + str(upRight))
                print("LowerLeft: " + str(lowLeft))
                print("LowerRight: " + str(lowRight))
                if show:
                    edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                    edges = drawLineWithMid(edges, upCorners[0], upCorners[1], upMid)
                    edges = drawLineWithMid(edges, lowCorners[0], lowCorners[1], lowMid)
                    edges = drawLineWithMid(edges, leftCorners[0], leftCorners[1], leftMid)
                    edges = drawLineWithMid(edges, rightCorners[0], upCorners[1], upMid)
                    cv.imshow("Detected Lines", edges)
                    cv.waitKey()
                return None
        else:
            print("ERROR: 4 lines not found in image " + baseImg)
            return None

    if not border:
        border = [0, 0, 0, 0]
    extraSpace = [max(border[0], border[1]) + Decimal(0.05), max(border[2], border[3]) + Decimal(0.05)]
    if debug:
        print("extraSpace: " + str(extraSpace))
    offsetX = Decimal(round(src.shape[0] * extraSpace[0]))
    offsetY = Decimal(round(src.shape[1] * extraSpace[1]))

    if show and not manual:
        edges = drawBox(cv.cvtColor(edges, cv.COLOR_GRAY2BGR), upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight)
        cv.imshow("4 main lines", edges)

    upLeft = [upLeft[0] + offsetX, upLeft[1] + offsetY]
    upRight = [upRight[0] + offsetX, upRight[1] + offsetY]
    lowLeft = [lowLeft[0] + offsetX, lowLeft[1] + offsetY]
    lowRight = [lowRight[0] + offsetX, lowRight[1] + offsetY]
    cardWidth = max(upRight[0] - upLeft[0], lowRight[0] - lowLeft[0])
    cardHeight = max(lowRight[1] - upRight[1], lowLeft[1] - upLeft[1])

    upMid = [(upLeft[0] + upRight[0]) / 2, upMid[1] + offsetY]
    lowMid = [(lowLeft[0] + lowRight[0]) / 2, lowMid[1] + offsetY]
    leftMid = [leftMid[0] + offsetX, (lowLeft[1] + upLeft[1]) / 2]
    rightMid = [rightMid[0] + offsetX, (lowRight[1] + upRight[1]) / 2]
    midPoint = [(leftMid[0] + rightMid[0]) / 2, (upMid[1] + lowMid[1]) / 2]



    if debug:
        print("UpperLeft: " + str(upLeft))
        print("UpperRight: " + str(upRight))
        print("LowerLeft: " + str(lowLeft))
        print("LowerRight: " + str(lowRight))
        print("middlePoint: " + str(midPoint))
        print("cardWidth: " + str(cardWidth))
        print("cardHeight: " + str(cardHeight))
        print("border: " + str(border))

    if res:
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

    targetOffsetX = Decimal(round(((targetWidth * extraSpace[0])) * 2) / 2)
    targetOffsetY = Decimal(round(((targetHeight * extraSpace[1])) * 2) / 2)
    targetCard = [targetOffsetY, targetOffsetY + targetHeight, targetOffsetX, targetOffsetX + targetWidth]
    targetMid = (
    Decimal(targetWidth * (Decimal(0.5) + extraSpace[0])), Decimal(targetHeight * (Decimal(0.5) + extraSpace[1])))
    if debug:
        print("border: " + str(border))
        print("extraSpace: " + str(extraSpace))
        print("targetwidth: " + str(targetWidth))
        print("targethieght: " + str(targetHeight))
        print("targetmid: " + str(targetMid))
        print("targetcard: " + str(targetCard))
    srcP = np.array(
        calculateOuterAndInnerPoint(upLeft, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(leftMid, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(lowLeft, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(lowMid, midPoint, extraSpace) +
        [midPoint] +
        calculateOuterAndInnerPoint(upMid, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(upRight, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(rightMid, midPoint, extraSpace) +
        calculateOuterAndInnerPoint(lowRight, midPoint, extraSpace), dtype="float64")

    dstP = np.array(
        calculateOuterAndInnerPoint([targetCard[2], targetCard[0]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetCard[2], targetMid[1]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetCard[2], targetCard[1]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetMid[0], targetCard[1]], targetMid, extraSpace) +
        [targetMid] +
        calculateOuterAndInnerPoint([targetMid[0], targetCard[0]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetCard[3], targetCard[0]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetCard[3], targetMid[1]], targetMid, extraSpace) +
        calculateOuterAndInnerPoint([targetCard[3], targetCard[1]], targetMid, extraSpace), dtype="float64")

    if debug:
        print(srcP)
        print(dstP)
        print(offsetX)
        print(offsetY)

    src = cv.cvtColor(src, cv.COLOR_BGR2BGRA)
    bordered = addBlurredExtendBorder(src, round(offsetY), round(offsetY), round(offsetX), round(offsetX))

    tform = PiecewiseAffineTransform()
    tform.estimate(dstP, srcP)
    warped = img_as_ubyte(warp(bordered, tform, output_shape=(
    round(targetHeight + targetOffsetY * 2), round(targetWidth + targetOffsetX * 2))))

    if trim:
        warped = trimImage(targetCard[0], targetCard[1], targetCard[2], targetCard[3])

    adjustNeeded = [round(round(targetCard[0] - border[0]) * Decimal(-1)),
                    round(round(targetCard[1] + border[1]) - warped.shape[0]),
                    round(round(targetCard[2] - border[2]) * Decimal(-1)),
                    round(round(targetCard[3] + border[3]) - warped.shape[1])]
    if debug:
        print("adjustNeeded: " + str(adjustNeeded))
    if any(side < 0 for side in adjustNeeded):
        warped = warped[max(0, adjustNeeded[0] * -1):len(warped) + min(0, adjustNeeded[1]),
                 max(0, adjustNeeded[2] * -1):len(warped[0]) + min(0, adjustNeeded[3])]
    if any(side > 0 for side in adjustNeeded):
        warped = addBlurredExtendBorder(warped, max(0, border[0]), max(0, border[1]), max(0, border[2]),
                                        max(0, border[3]))

    if mask:
        warped = maskcorners.processMask(warped, mask)
    if show:
        cv.imshow("outputed", warped)
        cv.waitKey()
        return None
    return warped


def processMultiArg(arg, numNeeded, decimal):
    arg = arg.split(",")
    argList = []
    for num in arg:
        if decimal:
            argList.append(Decimal(num))
        else:
            argList.append(int(num))
    if len(argList) != numNeeded:
        raise ValueError("EdgeFiltering must have exactly 4 numbers")
    return argList


def processArgs(inputText):
    input = None
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
    filter = False

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
    parser.add_argument("-f", "--Filter", help="Bring up the filter menu to customise the filter used. default: False")

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
        filter = args.Filter
    return input, clean, output, border, trim, edge, res, mask, manual, filter, debug, show


def resolveImage(input, clean, output, border, trim, edge, res, mask, manual, filter, debug, show):
    print("processing " + input)
    image = processImage(input, clean, border, trim, edge, res, mask, manual, filter, debug, show)
    if image is not None:
        cv.imwrite(output, image)


def processFolder(input, clean, output, border, trim, edge, res, mask, manual, filter, debug, show):
    try:
        os.mkdir(output)
    except FileExistsError:
        pass
    with os.scandir(input) as entries:
        for entry in entries:
            cleanPath = None
            inputPath = os.path.join(input, entry.name)
            outputPath = os.path.join(output, entry.name)
            if clean:
                cleanPath = os.path.join(clean, entry.name)
            if os.path.isfile(inputPath) and entry.name != "Place Images Here":
                resolveImage(inputPath, cleanPath, outputPath, border, trim, edge, res, mask, manual, filter, debug, show)
            elif os.path.isdir(inputPath):
                processFolder(inputPath, cleanPath, outputPath, border, trim, edge, res, mask, manual, filter, debug, show)


def main():
    input, clean, output, border, trim, edge, res, mask, manual, filter, debug, show = processArgs("folder")
    if not input:
        input = os.path.join(os.getcwd(), "input")
    if os.path.isfile(input):
        resolveImage(input, clean, output, border, trim, edge, res, mask, manual, filter, debug, show)
    elif os.path.isdir(input):
        processFolder(input, clean, output, border, trim, edge, res, mask, manual, filter, debug, show)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()
