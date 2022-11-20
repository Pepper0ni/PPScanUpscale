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

ANGLE_TOLERANCE = 0.01  # in radians
DEFAULT_BORDER_TOLERANCE = 0.0327272727273  # multiplied by y coord
DEFAULT_H_THRESHOLD = 0.1
DEFAULT_V_THRESHOLD = 0.1


def angleToColour():  # rad):
    return (0, 255, 0)
    degrees = abs(rad) * 180 / np.pi
    angle = (degrees + 45) % 90 - 45
    if abs(angle) < (2.55 / 8):
        green = 255 - (abs(angle) * 800)
    else:
        green = 0
    if angle > 0:
        red = min(255, angle * 1600)
        blue = 0
    elif angle == 0:
        red = 0
        blue = 0
    else:
        blue = min(255, angle * -1600)
        red = 0
    return blue, green, red


def intersect(line1, line2):
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


def sumLinePixels(img, pt1, pt2, testimg=None):
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
    return rating / len(pointsList), testimg


def sumLinesPixels(img, line1, line2, debug, show):
    innerImg = None
    outerImg = None
    if show:
        innerImg = cv.cvtColor(np.copy(img), cv.COLOR_GRAY2BGR)#np.zeros(shape=[img.shape[0],img.shape[1],3], dtype=np.uint8)
        outerImg = np.copy(innerImg)
    midPnt = intersect(line1, line2)
    if not midPnt: return None
    inner1, innerImg = sumLinePixels(img, line1[0], midPnt, innerImg)
    inner2, innerImg = sumLinePixels(img, line2[1], midPnt, innerImg)
    outer1, outerImg = sumLinePixels(img, line2[0], midPnt, outerImg)
    outer2, outerImg = sumLinePixels(img, line1[1], midPnt, outerImg)
    innerScore = inner1 + inner2
    outerScore = outer1 + outer2
    if debug:
        linesName = str(round(line1[0][0], 1)) + " " + str(round(line1[0][1], 1)) + " " + str(
            round(line1[1][0], 1)) + " " + str(round(line1[1][1], 1)) + " " \
                    + str(round(line2[0][0], 1)) + " " + str(round(line2[0][1], 1)) + " " + str(
            round(line2[1][0], 1)) + " " + str(round(line2[1][1], 1))
        print(linesName)
        print(midPnt)
        print("innerscore: " + str(innerScore))
        print("outerscore: " + str(outerScore))
        cv.line(innerImg, [round(line1[0][0]), round(line1[0][1])],
                    [round(midPnt[0]), round(midPnt[1])], angleToColour(), 1, cv.LINE_AA)
        cv.line(innerImg, [round(line2[1][0]), round(line2[1][1])],
                    [round(midPnt[0]), round(midPnt[1])], angleToColour(), 1, cv.LINE_AA)
        cv.line(outerImg, [round(line1[1][0]), round(line1[1][1])],
                    [round(midPnt[0]), round(midPnt[1])], angleToColour(), 1, cv.LINE_AA)
        cv.line(outerImg, [round(line2[0][0]), round(line2[0][1])],
                    [round(midPnt[0]), round(midPnt[1])], angleToColour(), 1, cv.LINE_AA)
        #cv.imshow("inner" + linesName, innerImg)
        #cv.imshow("outer" + linesName, outerImg)
        #cv.waitKey()
    if innerScore > outerScore:
        return [True, innerScore]
    return [False, outerScore]


def addBlurredExtendBorder(src, top, bottom, left, right):
    # blurred = cv.blur(src,(5,5))
    blurred = cv.GaussianBlur(src, (31, 31), 0)
    blurred = cv.copyMakeBorder(blurred, round(top), round(bottom), round(left), round(right), cv.BORDER_REPLICATE)
    # cv.imshow("bordered", blurred)

    blurred[round(top):round(src.shape[0] + top), round(left):round(src.shape[1] + left), :] = src
    return blurred


def trimNegLine(pt1, pt2):
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
    return [(disX * ratio) + pt1[0], (disY * ratio) + pt1[1]]


def trimLongLine(pt1, pt2, maxX, maxY):
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
    return [(ratio * disX) + pt1[0], (ratio * disY) + pt1[1]]


def trimLine(pt1, pt2, maxX, maxY):
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


def detectLines(img, threshold, side, debug, show):
    # side: 0 = top, 1 = bottom, 2 = left, 3 = right
    if side <= 1:
        baseAngle = 0.5
        axis = 1
        offAxis = 0
    else:
        baseAngle = 1
        axis = 0
        offAxis = 1
    if side % 2 == 0:
        op = operator.lt
    else:
        op = operator.gt
    if debug:
        print("side: " + str(side))

    while True:
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
                    cv.line(allImg, int1, int2, angleToColour(), 1, cv.LINE_AA)
                # if debug:
                #     print("low point: " + str(pt1))
                #     print("high point: " + str(pt2))
                if (img.shape[offAxis] > pt1[axis] > 0 or img.shape[offAxis] < pt1[axis] < 0) and \
                        (img.shape[offAxis] > pt2[axis] > 0 or img.shape[offAxis] < pt2[axis] < 0):
                    processedLines.append([pt1, pt2])
                    if minCorner is None or op(pt1[axis], processedLines[minCorner][0][axis]):
                        minCorner = count
                    if maxCorner is None or op(pt2[axis], processedLines[maxCorner][1][axis]):
                        maxCorner = count
                    count += 1

            if maxCorner == minCorner:
                corners = processedLines[maxCorner]
                mid = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2]
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
                            cv.line(proImg, int1, int2, angleToColour(), 1, cv.LINE_AA)
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
                    if score:
                        if not highScore or score[0] > highScore[1]:
                            highScore = [True, score[0], i, i]

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
        if corners and mid:
            if show:
                cv.imshow("all lines for side " + str(side), allImg)
                if drawProImg:
                    cv.imshow("possible lines for side " + str(side), proImg)
            return corners, mid
        threshold -= 20
        print("looping: " + str(threshold))
        if threshold <= 0:
            return None, None


def trimImage(img, fromTop, newBot, fromLeft, newRight):
    #print((fromTop, newBot, fromLeft, newRight))
    return img[fromTop:newBot, fromLeft:newRight]


def calculateOuterAndInnerPoint(pnt, middle, extraSpace):
    return [
        [pnt[0] + ((pnt[0] - middle[0]) * (extraSpace[0] * 2)), pnt[1] + ((pnt[1] - middle[1]) * (extraSpace[1] * 2))],
        pnt]


def processImage(baseImg, cleanImg, border, trim, edge, res, mask, debug=False, show=False):
    src = cv.imread(cv.samples.findFile(baseImg))
    if cleanImg:
        clean = cv.imread(cv.samples.findFile(cleanImg))
    else:
        clean = src
    clean = cv.inRange(cv.cvtColor(clean, cv.COLOR_BGR2HSV), (21,180,226),(29,255,255))
    clean = cv.medianBlur(clean, 3)

    if src is None:
        print('Base Image at ' + baseImg + ' Not Found, skipping')
        return
    if clean is None:
        print('Clean Image at ' + cleanImg + ' Not Found, attempting with base image')
        clean = src
    if debug:
        print("image size: " + str(src.shape))
    edges = cv.Canny(clean, 25, 1200, True, 5)
    if show:
        cv.imshow("clean", clean)
        cv.imshow("edges", edges)
    if not edge:
        edgeSize = round(DEFAULT_BORDER_TOLERANCE * src.shape[0])
        edge = [edgeSize, edgeSize, edgeSize, edgeSize]
    if debug:
        print("edges " + str(edge))
    threshold = DEFAULT_H_THRESHOLD * src.shape[1]
    upperCorners, upperMid = detectLines(trimImage(np.copy(edges), 0, edge[0], 0, src.shape[1]), threshold, 0, debug,
                                         show)
    lowerCorners, lowerMid = detectLines(
        trimImage(np.copy(edges), src.shape[0] - edge[1], src.shape[0], 0, src.shape[1]), threshold, 1, debug,
        show)

    threshold = DEFAULT_V_THRESHOLD * src.shape[0]
    leftCorners, leftMid = detectLines(trimImage(np.copy(edges), 0, src.shape[0], 0, edge[2]), threshold, 2, debug,
                                       show)
    rightCorners, rightMid = detectLines(
        trimImage(np.copy(edges), 0, src.shape[0], src.shape[1] - edge[3], src.shape[1]), threshold, 3, debug,
        show)

    if not (upperCorners and lowerCorners and leftCorners and rightCorners):
        print("error finding edges")
        return

    lowerCorners[0][1] += src.shape[0] - edge[1]
    lowerCorners[1][1] += src.shape[0] - edge[1]
    lowerMid[1] += src.shape[0] - edge[1]
    rightCorners[0][0] += src.shape[1] - edge[3]
    rightCorners[1][0] += src.shape[1] - edge[3]
    rightMid[0] += src.shape[1] - edge[3]

    if not border:
        border = [0, 0, 0, 0]
    extraSpace = [max(border[0], border[1]) + Decimal(0.05), max(border[2], border[3]) + Decimal(0.05)]
    if debug:
        print("extraSpace: " + str(extraSpace))
    offsetX = Decimal(round(src.shape[0] * extraSpace[0]))
    offsetY = Decimal(round(src.shape[1] * extraSpace[1]))

    if lowerCorners and rightCorners and upperCorners and leftCorners:
        if debug:
            print("upperCorners: " + str(upperCorners))
            print("lowerCorners: " + str(lowerCorners))
            print("leftCorners: " + str(leftCorners))
            print("rightCorners: " + str(rightCorners))
            print("upperMid: " + str(upperMid))
            print("lowerMid: " + str(lowerMid))
            print("leftMid: " + str(leftMid))
            print("rightMid: " + str(rightMid))

        upperLeft = intersect((upperCorners[0], upperMid), (leftCorners[0], leftMid))
        upperRight = intersect((upperCorners[1], upperMid), (rightCorners[0], rightMid))
        lowerLeft = intersect((lowerCorners[0], lowerMid), (leftCorners[1], leftMid))
        lowerRight = intersect((lowerCorners[1], lowerMid), (rightCorners[1], rightMid))

        if not (upperLeft and upperRight and lowerLeft and lowerRight):
            print("ERROR: Lines do not intersect")
            print("UpperLeft: " + str(upperLeft))
            print("UpperRight: " + str(upperRight))
            print("LowerLeft: " + str(lowerLeft))
            print("LowerRight: " + str(lowerRight))
            if show:
                edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                cv.line(edges, [round(upperCorners[0][0]), round(upperCorners[0][1])],
                        [round(upperMid[0]), round(upperMid[1])], angleToColour(), 1, cv.LINE_AA)
                cv.line(edges, [round(upperCorners[1][0]), round(upperCorners[1][1])],
                        [round(upperMid[0]), round(upperMid[1])], angleToColour(), 1, cv.LINE_AA)

                cv.line(edges, [round(lowerCorners[0][0]), round(lowerCorners[0][1])],
                        [round(lowerMid[0]), round(lowerMid[1])], angleToColour(), 1, cv.LINE_AA)
                cv.line(edges, [round(lowerCorners[1][0]), round(lowerCorners[1][1])],
                        [round(lowerMid[0]), round(lowerMid[1])], angleToColour(), 1, cv.LINE_AA)

                cv.line(edges, [round(leftCorners[0][0]), round(leftCorners[0][1])],
                        [round(leftMid[0]), round(leftMid[1])], angleToColour(), 1, cv.LINE_AA)
                cv.line(edges, [round(leftCorners[1][0]), round(leftCorners[1][1])],
                        [round(leftMid[0]), round(leftMid[1])], angleToColour(), 1, cv.LINE_AA)

                cv.line(edges, [round(rightCorners[0][0]), round(rightCorners[0][1])],
                        [round(rightMid[0]), round(rightMid[1])], angleToColour(), 1, cv.LINE_AA)
                cv.line(edges, [round(rightCorners[1][0]), round(rightCorners[1][1])],
                        [round(rightMid[0]), round(rightMid[1])], angleToColour(), 1, cv.LINE_AA)
                cv.imshow("Detected Lines", edges)
                cv.waitKey()
            return None

        upperLeft = [upperLeft[0] + offsetX, upperLeft[1] + offsetY]
        upperRight = [upperRight[0] + offsetX, upperRight[1] + offsetY]
        lowerLeft = [lowerLeft[0] + offsetX, lowerLeft[1] + offsetY]
        lowerRight = [lowerRight[0] + offsetX, lowerRight[1] + offsetY]
        cardWidth = max(upperRight[0] - upperLeft[0], lowerRight[0] - lowerLeft[0])
        cardHeight = max(lowerRight[1] - upperRight[1], lowerLeft[1] - upperLeft[1])

        upperMid = [(upperLeft[0] + upperRight[0]) / 2, upperMid[1] + offsetY]
        lowerMid = [(lowerLeft[0] + lowerRight[0]) / 2, lowerMid[1] + offsetY]
        leftMid = [leftMid[0] + offsetX, (lowerLeft[1] + upperLeft[1]) / 2]
        rightMid = [rightMid[0] + offsetX, (lowerRight[1] + upperRight[1]) / 2]
        midPoint = [(leftMid[0] + rightMid[0]) / 2, (upperMid[1] + lowerMid[1]) / 2]

        if show:
            edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            cv.line(edges, [round(upperLeft[0] - offsetX), round(upperLeft[1] - offsetY)],
                    [round(upperMid[0] - offsetX), round(upperMid[1] - offsetY)], angleToColour(), 1, cv.LINE_AA)
            cv.line(edges, [round(upperRight[0] - offsetX), round(upperRight[1] - offsetY)],
                    [round(upperMid[0] - offsetX), round(upperMid[1] - offsetY)], angleToColour(), 1, cv.LINE_AA)
            cv.line(edges, [round(upperLeft[0] - offsetX), round(upperLeft[1] - offsetY)],
                    [round(leftMid[0] - offsetX), round(leftMid[1] - offsetY)], angleToColour(), 1, cv.LINE_AA)
            cv.line(edges, [round(lowerLeft[0] - offsetX), round(lowerLeft[1] - offsetY)],
                    [round(leftMid[0] - offsetX), round(leftMid[1] - offsetY)], angleToColour(), 1, cv.LINE_AA)

            cv.line(edges, [round(lowerLeft[0] - offsetX), round(lowerLeft[1] - offsetY)],
                    [round(lowerMid[0] - offsetX), round(lowerMid[1] - offsetY)], angleToColour(), 1, cv.LINE_AA)
            cv.line(edges, [round(lowerRight[0] - offsetX), round(lowerRight[1] - offsetY)],
                    [round(lowerMid[0] - offsetX), round(lowerMid[1] - offsetY)], angleToColour(), 1, cv.LINE_AA)
            cv.line(edges, [round(lowerRight[0] - offsetX), round(lowerRight[1] - offsetY)],
                    [round(rightMid[0] - offsetX), round(rightMid[1] - offsetY)], angleToColour(), 1, cv.LINE_AA)
            cv.line(edges, [round(upperRight[0] - offsetX), round(upperRight[1] - offsetY)],
                    [round(rightMid[0] - offsetX), round(rightMid[1] - offsetY)], angleToColour(), 1, cv.LINE_AA)
            cv.imshow("4 main lines", edges)

        if debug:
            print("UpperLeft: " + str(upperLeft))
            print("UpperRight: " + str(upperRight))
            print("LowerLeft: " + str(lowerLeft))
            print("LowerRight: " + str(lowerRight))
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
            calculateOuterAndInnerPoint(upperLeft, midPoint, extraSpace) +
            calculateOuterAndInnerPoint(leftMid, midPoint, extraSpace) +
            calculateOuterAndInnerPoint(lowerLeft, midPoint, extraSpace) +
            calculateOuterAndInnerPoint(lowerMid, midPoint, extraSpace) +
            [midPoint] +
            calculateOuterAndInnerPoint(upperMid, midPoint, extraSpace) +
            calculateOuterAndInnerPoint(upperRight, midPoint, extraSpace) +
            calculateOuterAndInnerPoint(rightMid, midPoint, extraSpace) +
            calculateOuterAndInnerPoint(lowerRight, midPoint, extraSpace), dtype="float64")

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
    else:
        print("ERROR: 4 lines not found in image " + baseImg)
        return None


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
    return input, clean, output, border, trim, edge, res, mask, debug, show


def resolveImage(input, clean, output, border, trim, edge, res, mask, debug, show):
    print("processing " + input)
    image = processImage(input, clean, border, trim, edge, res, mask, debug, show)
    if image is not None:
        cv.imwrite(output, image)


def processFolder(input, clean, output, border, trim, edge, res, mask, debug, show):
    try:
        os.mkdir(output)
    except FileExistsError:
        pass
    with os.scandir(input) as entries:
        for entry in entries:
            inputPath = os.path.join(input, entry.name)
            outputPath = os.path.join(output, entry.name)
            cleanPath = os.path.join(clean, entry.name)
            if os.path.isfile(inputPath) and entry.name != "Place Images Here":
                resolveImage(inputPath, cleanPath, outputPath, border, trim, edge, res, mask, debug, show)
            elif os.path.isdir(inputPath):
                processFolder(inputPath, cleanPath, outputPath, border, trim, edge, res, mask, debug, show)


def main():
    input, clean, output, border, trim, edge, res, mask, debug, show = processArgs("folder")
    if not input:
        input = os.path.join(os.getcwd(), "input")
    if not clean:
        clean = os.path.join(os.getcwd(), "temp")
    if os.path.isfile(input):
        resolveImage(input, clean, output, border, trim, edge, res, mask, debug, show)
    elif os.path.isdir(input):
        processFolder(input, clean, output, border, trim, edge, res, mask, debug, show)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()
