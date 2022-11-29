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

ANGLE_TOLERANCE = 0.0314  # maximum variance from a straight line the line detector will accept in radians
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
timesRun = 0


def sharpen(src, ksize, amount, iter):
    hsvSrc = cv.cvtColor(src, cv.COLOR_BGR2HSV_FULL)
    h, s, v = cv.split(hsvSrc)
    for _ in range(iter):
        v = img_as_ubyte(unsharp_mask(v, ksize, amount))
    return  cv.cvtColor(cv.merge([h, s, v]), cv.COLOR_HSV2BGR_FULL)

def getMidpoint(line): #get the midpoint of a line
    return [(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2]

def intersect(line1, line2): #find interdection between 2 points
    if not (line1 and line2) or not (line1[0] and line1[1] and line2[0] and line2[1]):
        #if line missing, propogate None
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
    #if show:
        #innerImg = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2BGR)#np.zeros(shape=[img.shape[0],img.shape[1],3], dtype=np.uint8)
        #outerImg = np.copy(innerImg)
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
    #if show:
        #innerImg = drawLineWithMid(img, line1[0], line2[1], mid)
        #outerImg = drawLineWithMid(img, line1[1], line2[0], mid)
        #cv.imshow("inner" + linesName, innerImg)
        #cv.imshow("outer" + linesName, outerImg)
        #cv.waitKey()
    if innerScore > outerScore:
        return [True, innerScore]
    return [False, outerScore]

def pasteImage(base, sprite, posX, posY): #paste 1 image into another in the specified position
    base[round(posY):round(sprite.shape[0] + posY), round(posX):round(sprite.shape[1] + posX), :] = sprite
    return base

def addBlurredExtendBorder(src, top, bottom, left, right, blur): #add an extend border with a blur effect to smooth out varience
    blurred = cv.blur(src, (blur[0], blur[1]), 0)
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
        baseAngle = np.pi * 0.5
        axis = 1
        offAxis = 0
    else:
        baseAngle = np.pi * 1
        axis = 0
        offAxis = 1
    if side % 2 == 0: #set whether we are looking for the highest or lowest lines, based on side
        op = operator.lt
    else:
        op = operator.gt
    if debug:
        print("side: " + str(side))

    lines = cv.HoughLines(img, 0.25, np.pi / 2880, round(threshold), None, 0, 0,
                         baseAngle - ANGLE_TOLERANCE,
                         baseAngle + ANGLE_TOLERANCE) #get lines
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
            pt1, pt2 = trimLine(pt1, pt2, img.shape[1], img.shape[0]) #get line in cartesean coords and limit it to the bounds of the image
            if pt1[offAxis] > pt2[offAxis]: #sort the points based on the axis not being tested
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
                    (img.shape[axis] == pt2[offAxis] or pt2[offAxis] == 0): #check if both points touch the far ends of the image, reject if they don't
                processedLines.append([pt1, pt2])
                if minCorner is None or op(pt1[axis], processedLines[minCorner][0][axis]): #keep track of the line furthest towards the card edge on each side
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
            if maxCorner == minCorner: #if the furthest out line on each side is the same, use that line.
                corners = processedLines[maxCorner]
                mid = getMidpoint(corners)
            else: #otherwise test each line combonation to find the one that matches the true line the most
                minRange = (processedLines[minCorner][0][axis], processedLines[maxCorner][0][axis]) #filter out lines not within the 2 furthest out lines
                maxRange = (processedLines[maxCorner][1][axis], processedLines[minCorner][1][axis])
                if debug:
                    print("low-side range: " + str(minRange))
                    print("high-side range: " + str(maxRange))
                    print(processedLines)
                for line in processedLines: #filter out lines not within the 2 furthest out lines
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
                                score = sumLinesPixels(img, prunedLines[i], prunedLines[j], show, debug) #get the amount of each line that matches
                                if score:
                                    if not highScore or score[1] > highScore[1]:
                                        highScore = score + [i, j] #save the best line and if it's inwards or outwards
                    score = sumLinePixels(img, prunedLines[i][0], prunedLines[i][1]) #check single lines too
                    if debug:
                        print("score for solo line: " + str(i) + " " + str(score[0]))
                    if score:
                        if not highScore or score[0] > highScore[1]:
                            highScore = [True, score[0], i, i]
                if highScore[2] == highScore[3]: #if a single line is best, use it
                    if debug:
                        print("Chose line " + str(highScore[2]))
                    corners = prunedLines[highScore[2]]
                    mid = getMidpoint(corners)
                else:
                    highI = prunedLines[highScore[2]] #otherwise use a combonation of 2 lines to find the best, storing the intersection as the mid point
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
    return None, None

def correctMid(mid, midAxis, minCorner, maxCorner, ownCorners): #check if the midpoint is outside the border, change to a single line if it is
    if mid and minCorner and maxCorner and ownCorners and (mid[midAxis] <= minCorner+1 or mid[midAxis] >= maxCorner-1):
        print("mid out of bounds, attempting to correct")
        return getMidpoint(ownCorners)
    return mid

def getLines(clean, edge, debug, show): #get the 9 points of the border
    clean = cv.medianBlur(clean, 3) #use a median blur to fill in small gaps
    edges = cv.Canny(clean, 25, 1200, True, 5)
    global timesRun
    timesRun += 1
    if show:
        cv.imshow("clean " + str(timesRun), clean)
        cv.imshow("edges " + str(timesRun), edges)

    threshold = DEFAULT_H_THRESHOLD * edges.shape[1] #set line threshold based on image resolution
    upCorners, upMid = detectLines(trimImage(np.copy(edges), MIN_LINE_EDGE, edge[0], 0, edges.shape[1]), threshold, 0, debug, show)
    lowCorners, lowMid = detectLines(trimImage(np.copy(edges), edges.shape[0] - edge[1], edges.shape[0] - MIN_LINE_EDGE, 0, edges.shape[1]), threshold, 1, debug, show)

    threshold = DEFAULT_V_THRESHOLD * edges.shape[0]
    leftCorners, leftMid = detectLines(trimImage(np.copy(edges), 0, edges.shape[0], MIN_LINE_EDGE, edge[2]), threshold, 2,
                                       debug, show)
    rightCorners, rightMid = detectLines(
        trimImage(np.copy(edges), 0, edges.shape[0], edges.shape[1] - edge[3], edges.shape[1] - MIN_LINE_EDGE), threshold, 3, debug,
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
    upCorners, upMid, lowCorners, lowMid, leftCorners, leftMid, rightCorners, rightMid = getLines(clean, edge, debug, show)
    if not (upCorners and lowCorners and leftCorners and rightCorners):
        print("ERROR: Could not find 4 edges")
        first = False
        if show:
            lines = drawLineWithMid(src, upCorners[0], upCorners[1], upMid)
            lines = drawLineWithMid(lines, lowCorners[0], lowCorners[1], lowMid)
            lines = drawLineWithMid(lines, leftCorners[0], leftCorners[1], leftMid)
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
        upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight = taskbarFuncs.CustomBordersUI(src, upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight)
    return upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight

def filterImage(hsvClean, adaptive, show):
    if adaptive: #an apative method that tried to take a chunk of the border and make a filter based on it.
        borderBase = cv.vconcat([trimImage(hsvClean, MIN_ELEC_AVG_RANGE, MAX_ELEC_AVG_RANGE, MIN_ELEC_AVG_RANGE,
                                           hsvClean.shape[1] - MIN_ELEC_AVG_RANGE),
                                 trimImage(hsvClean, hsvClean.shape[0] - MAX_ELEC_AVG_RANGE,
                                           hsvClean.shape[0] - MIN_ELEC_AVG_RANGE, MIN_ELEC_AVG_RANGE,
                                           hsvClean.shape[1] - MIN_ELEC_AVG_RANGE)])

        borderBaseV = cv.hconcat([trimImage(hsvClean, MIN_ELEC_AVG_RANGE, hsvClean.shape[0] - MIN_ELEC_AVG_RANGE,
                                            MIN_ELEC_AVG_RANGE, MAX_ELEC_AVG_RANGE),
                                  trimImage(hsvClean, MIN_ELEC_AVG_RANGE, hsvClean.shape[0] - MIN_ELEC_AVG_RANGE,
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
                           (taskbarFuncs.hiHue, taskbarFuncs.hiSat, taskbarFuncs.hiVal))  # apply the range filter to the values
        if show:
            cv.imshow("adaptive clean", clean)
        return clean
    else: #a simpler method that tries to match a large range of yellow
        taskbarFuncs.lowHue = 20
        taskbarFuncs.hiHue = 30
        taskbarFuncs.lowSat = 120
        taskbarFuncs.hiSat = 255
        taskbarFuncs.lowVal = 190
        taskbarFuncs.hiVal = 255
        clean = cv.inRange(hsvClean, (taskbarFuncs.lowHue, taskbarFuncs.lowSat, taskbarFuncs.lowVal),
                           (taskbarFuncs.hiHue, taskbarFuncs.hiSat, taskbarFuncs.hiVal))
        if show:
            cv.imshow("basic clean", clean)
        return clean


def trimImage(img, fromTop, newBot, fromLeft, newRight): #crop the image based on the supplied values
    return img[fromTop:newBot, fromLeft:newRight]


def calculateOuterAndInnerPoint(pnt, middle, extraSpace): #from point and midspace, get a matching outer point for mesh transform
    return [
        [pnt[0] + ((pnt[0] - middle[0]) * (extraSpace[0] * 2)), pnt[1] + ((pnt[1] - middle[1]) * (extraSpace[1] * 2))],
        pnt]


def drawLineWithMid(img, pt1, pt2, mid): #draw a 3 point line
    if pt1 and mid:
        cv.line(img, [round(pt1[0]), round(pt1[1])],
                [round(mid[0]), round(mid[1])], (0, 255, 0), 1, cv.LINE_AA)
    if pt2 and mid:
        cv.line(img, [round(pt2[0]), round(pt2[1])],
                [round(mid[0]), round(mid[1])], (0, 255, 0), 1, cv.LINE_AA)
    return img


def drawBox(img, upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight): #draw an 8 point box
    drawLineWithMid(img, upLeft, upRight, upMid)
    drawLineWithMid(img, lowLeft, lowRight, lowMid)
    drawLineWithMid(img, upLeft, lowLeft, leftMid)
    drawLineWithMid(img, upRight, lowRight, rightMid)
    return img


def processImage(baseImg, cleanImg, border, trim, edge, res, mask, manual, filter, blur, debug=False, show=False):
    src = cv.imread(cv.samples.findFile(baseImg))
    if src is None:
        print('Base Image at ' + baseImg + ' Not Found, skipping')
        return
    fixBadCorners(src)
    if not edge:
        edgeSize = round(DEFAULT_BORDER_TOLERANCE * src.shape[0]) #set default value for expected border size
        edge = [edgeSize, edgeSize, edgeSize, edgeSize]
    if cleanImg:
        clean = cv.imread(cv.samples.findFile(cleanImg))
        if clean is None:
            print('Clean Image at ' + cleanImg + ' Not Found, attempting with base image')
            clean = src
    else:
        clean = src

    hsvClean = cv.cvtColor(clean, cv.COLOR_BGR2HSV) #make a HSV version of clean for filtering
    elecCheck = cv.inRange(hsvClean, (20, 70, 0), (30, 255, 255)) #check the colour of the image to see if it's electric and needs tighter filters
    electric = False
    if np.sum(elecCheck) > src.shape[0] * src.shape[1] * 128: #check only against the text of the image, to avoid the art throwing it off
        electric = True
    if debug:
        print("image size: " + str(src.shape))
        print("edges " + str(edge))
        print("Electric: " + str(electric))
    clean = filterImage(hsvClean, electric, show)
    if filter:
        clean = taskbarFuncs.HSVFilterUI(hsvClean) #set custom filter is enabled, does a normal filter first to get a sane default
    upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight = getPointsFromLines(clean, edge, debug, show, manual, src)
    if not (upLeft and upMid and upRight and leftMid and rightMid and lowLeft and lowMid and lowRight):
        print("Trying again with other filter...")
        clean = filterImage(hsvClean, not electric, show)
        upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight = getPointsFromLines(clean, edge, debug, show, manual, src)
        if not (upLeft and upMid and upRight and leftMid and rightMid and lowLeft and lowMid and lowRight):
            print("ERROR: 4 lines not found in image " + baseImg)
            if show:
                cv.waitKey()
            return None

    if not border:
        border = [0, 0, 0, 0]
    #set the amount of space outside the card frame to keep
    extraSpace = [max(border[0], border[1]) + Decimal(0.05), max(border[2], border[3]) + Decimal(0.05)]
    if debug:
        print("extraSpace: " + str(extraSpace))
    #the offset from the frames current position compared to before the extra was added
    offsetX = Decimal(round(src.shape[0] * extraSpace[0]))
    offsetY = Decimal(round(src.shape[1] * extraSpace[1]))

    if show and not manual:
        edges = drawBox(src, upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight)
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

    if res: #set the target card's size
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
    #set the target position
    targetOffsetX = Decimal(round(((targetWidth * extraSpace[0])) * 2) / 2)
    targetOffsetY = Decimal(round(((targetHeight * extraSpace[1])) * 2) / 2)
    targetCard = [targetOffsetY, targetOffsetY + targetHeight, targetOffsetX, targetOffsetX + targetWidth]
    targetMid = (Decimal(targetWidth * (Decimal(0.5) + extraSpace[0])), Decimal(targetHeight * (Decimal(0.5) + extraSpace[1])))

    if debug:
        print("border: " + str(border))
        print("extraSpace: " + str(extraSpace))
        print("targetwidth: " + str(targetWidth))
        print("targethieght: " + str(targetHeight))
        print("targetmid: " + str(targetMid))
        print("targetcard: " + str(targetCard))
    #create arrays in scipy compatible format
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

    #src = cv.cvtColor(src, cv.COLOR_BGR2BGRA)
    #expand the border to fill the extra space
    bordered = addBlurredExtendBorder(src, round(offsetY), round(offsetY), round(offsetX), round(offsetX), blur)

    tform = PiecewiseAffineTransform()
    tform.estimate(dstP, srcP)
    # mesh transform the image into shape
    warped = img_as_ubyte(warp(bordered, tform, output_shape=(
    round(targetHeight + targetOffsetY * 2), round(targetWidth + targetOffsetX * 2))))

    if trim:
        warped = trimImage(targetCard[0], targetCard[1], targetCard[2], targetCard[3])
    #calculate how much of the new image to trim off/add on to create the correct size border
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
                                        max(0, border[3]), blur)
    #apply an alpha mask if provided, to make the corners
    warped = cv.cvtColor(sharpen(warped, 3, 0.28, 1), cv.COLOR_BGR2BGRA)
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
    blur = [25, 25]

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
    parser.add_argument("-bb", "--BorderBlur", help="how much to blur the border, as so x,y. default: 25,25")

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
    if args.BorderBlur:
        blur = processMultiArg(args.BorderBlur, 2, False)
    return input, clean, output, border, trim, edge, res, mask, manual, filter, blur, debug, show


def resolveImage(input, clean, output, border, trim, edge, res, mask, manual, filter, blur, debug, show):
    print("processing " + input)
    image = processImage(input, clean, border, trim, edge, res, mask, manual, filter, blur, debug, show)
    if image is not None:
        cv.imwrite(output, image)


def processFolder(input, clean, output, border, trim, edge, res, mask, manual, filter, blur, debug, show):
    with suppress(FileExistsError):
        os.mkdir(output)
    with os.scandir(input) as entries:
        for entry in entries:
            cleanPath = None
            inputPath = os.path.join(input, entry.name)
            outputPath = os.path.join(output, entry.name)
            if clean:
                cleanPath = os.path.join(clean, entry.name)
            if os.path.isfile(inputPath) and entry.name != "Place Images Here":
                resolveImage(inputPath, cleanPath, outputPath, border, trim, edge, res, mask, manual, filter, blur, debug, show)
            elif os.path.isdir(inputPath):
                processFolder(inputPath, cleanPath, outputPath, border, trim, edge, res, mask, manual, filter, blur, debug, show)


def main():
    input, clean, output, border, trim, edge, res, mask, manual, filter, blur, debug, show = processArgs("folder")
    if not input:
        input = os.path.join(os.getcwd(), "input")
    if os.path.isfile(input):
        resolveImage(input, clean, output, border, trim, edge, res, mask, manual, filter, blur, debug, show)
    elif os.path.isdir(input):
        processFolder(input, clean, output, border, trim, edge, res, mask, manual, filter, blur, debug, show)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()
