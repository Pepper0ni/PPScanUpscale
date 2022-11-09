import math
import cv2 as cv
import numpy as np
import os
import argparse
from decimal import Decimal
from wand.image import Image as wandImage
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.util import img_as_ubyte
from skimage import data
from skimage import io

ANGLE_TOLERANCE = 0.025  # in radians
DEFAULT_BORDER_TOLERANCE = 0.0327272727273  # multiplied by y coord

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
        print("Lines parallel")
        return None
    ua = ((line2[1][0] - line2[0][0]) * (line1[0][1] - line2[0][1]) - (line2[1][1] - line2[0][1]) * (
                line1[0][0] - line2[0][0])) / denom
    if ua < 0 or ua > 1:
        print("Lines out of range 1")
        return None
    ub = ((line1[1][0] - line1[0][0]) * (line1[0][1] - line2[0][1]) - (line1[1][1] - line1[0][1]) * (
                line1[0][0] - line2[0][0])) / denom
    if ub < 0 or ub > 1:
        print("Lines out of range 2")
        return None
    return [line1[0][0] + ua * (line1[1][0] - line1[0][0]), line1[0][1] + ua * (line1[1][1] - line1[0][1])]

def addBlurredExtendBorder(src,top,bottom,left,right):
    blurred = cv.blur(src,(5,5))
    blurred = cv.copyMakeBorder(blurred, top, bottom, left, right, cv.BORDER_REPLICATE)
    #cv.imshow("bordered", blurred)
    blurred = cv.GaussianBlur(blurred,(9,9), 0)
    blurred[top:src.shape[0] + top, left:src.shape[1] + left, :] = src
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


def processLines(img, lines, axis, edge, debug):
    drawImg = np.copy(img)
    maxLines = [None, None]
    minLines = [None, None]
    minMid = None
    maxMid = None
    imgSize = (img.shape[1], img.shape[0])
    if not edge:
        maxEdge = round(DEFAULT_BORDER_TOLERANCE * imgSize[1])
        minEdge = round(DEFAULT_BORDER_TOLERANCE * imgSize[1])
    elif axis == 1:
        minEdge = edge[0]
        maxEdge = edge[1]
    else:
        minEdge = edge[2]
        maxEdge = edge[3]

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
            if debug:
                int1 = [int(pt1[0]), int(pt1[1])]
                int2 = [int(pt2[0]), int(pt2[1])]
                cv.line(drawImg, int1, int2, angleToColour(), 1, cv.LINE_AA)
            pt1, pt2 = trimLine(pt1, pt2, imgSize[0], imgSize[1])
            if debug:
                print(pt1)
                print(pt2)
                print(axis)
                print(imgSize[axis] - maxEdge)
                print(minEdge)
            if pt1[axis] > imgSize[axis] - maxEdge and pt2[axis] > imgSize[axis] - maxEdge:
                if not maxLines[0] or maxLines[0][0][axis] < pt1[axis]:
                    maxLines = ((pt1, pt2), maxLines[1])
                if not maxLines[1] or maxLines[1][1][axis] < pt2[axis]:
                    maxLines = (maxLines[0], (pt1, pt2))

            if pt1[axis] < minEdge and pt2[axis] < minEdge:
                if not minLines[0] or minLines[0][0][axis] > pt1[axis]:
                    minLines = ((pt1, pt2), minLines[1])
                if not minLines[1] or minLines[1][1][axis] > pt2[axis]:
                    minLines = (minLines[0], (pt1, pt2))

    if maxLines[0] == None or maxLines[1] == None:
        maxLine = None
    else:
        maxLine = (maxLines[0][0], maxLines[1][1])
        if maxLines[0] == maxLines[1]:
            maxMid = ((maxLine[0][0]+maxLine[1][0])/2, (maxLine[0][1]+maxLine[1][1])/2)
        else:
            maxMid = [imgSize[0] / 2, imgSize[1] / 2]
            maxMid[axis] = intersect(maxLines[0], maxLines[1])[axis]

    if minLines[0] == None or minLines[1] == None:
        minLine = None
    else:
        minLine = (minLines[0][0], minLines[1][1])
        if minLines[0] == minLines[1]:
            minMid = ((minLine[0][0]+minLine[1][0])/2, (minLine[0][1]+minLine[1][1])/2)
        else:
            minMid = [imgSize[0] / 2, imgSize[1] / 2]
            minMid[axis] = intersect(minLines[0], minLines[1])[axis]

    return drawImg, maxLine, maxMid, minLine, minMid

def calculateOuterAndInnerPoint(pnt, middle, extraSpace):
    return ([pnt[0]+((pnt[0]-middle[0])*Decimal(extraSpace[0])), pnt[1]+((pnt[1]-middle[1])*Decimal(extraSpace[1]))], pnt)

def processImage(baseImg, cleanImg, border, trim, edge, res, debug=False, show=False):
    src = cv.imread(cv.samples.findFile(baseImg))
    clean = cv.imread(cv.samples.findFile(cleanImg))

    if src is None:
        print('Base Image at ' + baseImg + ' Not Found, skipping')
        return
    if clean is None:
        print('Clean Image at ' + cleanImg + ' Not Found, attempting with base image')
        clean = src #cv.imread(cv.samples.findFile(baseImg))

    edges = cv.Canny(clean, 25, 500, True, 5)

    cEdges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    threshold = 0.633333333333 * src.shape[1]
    if debug:
        print("starting H threshold: "+ str(threshold))
    while True:
        horiLines = cv.HoughLines(edges, 1, np.pi / 2880, round(threshold), None, 0, 0, np.pi * (0.5 - ANGLE_TOLERANCE),
                                  np.pi * (0.5 + ANGLE_TOLERANCE))
        cdstH, lowerLine, lowerMid, upperLine, upperMid = processLines(cEdges, horiLines, 1, edge, debug)
        if upperLine and lowerLine:
            break
        threshold -= 20
        if threshold <= 0:
            break
    if debug:
        print("H Threshold: " + str(round(threshold)))
    threshold = 0.475 * src.shape[1]
    if debug:
        print("starting V threshold: "+ str(threshold))
    while True:
        vertLines = cv.HoughLines(edges, 1, np.pi / 2880, round(threshold), None, 0, 0, np.pi * (1 - ANGLE_TOLERANCE),
                                  np.pi * (1 + ANGLE_TOLERANCE))
        cdstV, rightLine, rightMid, leftLine, leftMid = processLines(cEdges, vertLines, 0, edge, debug)
        if rightLine and leftLine:
            break
        threshold -= 20
        if threshold <= 0:
            break
    if debug:
        print("V Threshold: " + str(round(threshold)))
    extraSpace = [0.1, 0.1]
    offsetX = Decimal(round(src.shape[0]*extraSpace[0]))
    offsetY = Decimal(round(src.shape[1]*extraSpace[1]))

    if lowerLine and rightLine and upperLine and leftLine:
        if debug:
            print("upperLine: " + str(upperLine))
            print("lowerLine: " + str(lowerLine))
            print("leftLine: " + str(leftLine))
            print("rightLine: " + str(rightLine))
            cv.line(cEdges, [int(upperLine[0][0]), int(upperLine[0][1])],
                    [int(upperLine[1][0]), int(upperLine[1][1])], angleToColour(), 1, cv.LINE_AA)
            cv.line(cEdges, [int(lowerLine[0][0]), int(lowerLine[0][1])],
                    [int(lowerLine[1][0]), int(lowerLine[1][1])], angleToColour(), 1, cv.LINE_AA)
            cv.line(cEdges, [int(leftLine[0][0]), int(leftLine[0][1])],
                    [int(leftLine[1][0]), int(leftLine[1][1])], angleToColour(), 1, cv.LINE_AA)
            cv.line(cEdges, [int(rightLine[0][0]), int(rightLine[0][1])], [int(rightLine[1][0]), int(rightLine[1][1])],
                    angleToColour(), 1, cv.LINE_AA)

        upperLeft = intersect(upperLine, leftLine)
        upperRight = intersect(upperLine, rightLine)
        lowerLeft = intersect(lowerLine, leftLine)
        lowerRight = intersect(lowerLine, rightLine)

        upperLeft = [upperLeft[0]+offsetX, upperLeft[1]+offsetY]
        upperRight = [upperRight[0]+offsetX, upperRight[1]+offsetY]
        lowerLeft = [lowerLeft[0]+offsetX, lowerLeft[1]+offsetY]
        lowerRight = [lowerRight[0]+offsetX, lowerRight[1]+offsetY]
        cardWidth = max(upperRight[0] - upperLeft[0], lowerRight[0] - lowerLeft[0])
        cardHeight = max(lowerRight[1] - upperRight[1], lowerLeft[1] - upperLeft[1])

        if not (upperLeft and upperRight and lowerLeft and lowerRight):
            print("ERROR: Lines do not intersect")
            print("UpperLeft: " + str(upperLeft))
            print("UpperRight: " + str(upperRight))
            print("LowerLeft: " + str(lowerLeft))
            print("LowerRight: " + str(lowerRight))

            if show:
                return src, edges, cEdges, cdstV, cdstH
            else:
                return src


        midPoint = (Decimal(round(cardWidth / 2) + offsetX), Decimal(round(cardHeight / 2) + offsetY))
        upperMid = [midPoint[0], max(0, upperMid[1])+offsetY]
        lowerMid = [midPoint[0], min(src.shape[0], lowerMid[1])+offsetY]
        leftMid = [max(0, leftMid[0])+offsetX, midPoint[1]]
        rightMid = [min(src.shape[1], rightMid[0])+offsetX, midPoint[1]]

        if debug:
            print("UpperLeft: " + str(upperLeft))
            print("UpperRight: " + str(upperRight))
            print("LowerLeft: " + str(lowerLeft))
            print("LowerRight: " + str(lowerRight))
            print("upperMid: " + str(upperMid))
            print("lowerMid: " + str(lowerMid))
            print("leftMid: " + str(leftMid))
            print("rightMid: " + str(rightMid))
            print("middlePoint: " + str(midPoint))
            print("cardWidth: " + str(cardWidth))
            print("cardHeight: " + str(cardHeight))

        if res:
            targetWidth = round(res[0] / (1+border[2]+border[3]))
            targetHeight = round(res[1] / (1+border[0]+border[1]))
        else:
            targetWidth = cardWidth
            targetHeight = cardHeight

        targetOffsetX = round(targetHeight*extraSpace[0])
        targetOffsetY = round(targetWidth*extraSpace[1])
        targetCard = [targetOffsetX, targetOffsetX+targetHeight,targetOffsetY,targetOffsetY+targetWidth]
        targetMid = (Decimal(round(targetHeight*(0.5+extraSpace[0]))), Decimal(round(targetWidth*(0.5+extraSpace[1])))]
        srcP = np.array(
            calculateOuterAndInnerPoint(upperLeft, midPoint, extraSpace) +
             calculateOuterAndInnerPoint(leftMid, midPoint, extraSpace) +
             calculateOuterAndInnerPoint(lowerLeft, midPoint, extraSpace) +
             calculateOuterAndInnerPoint(lowerMid, midPoint, extraSpace) +
             midPoint +
             calculateOuterAndInnerPoint(upperMid, midPoint, extraSpace) +
             calculateOuterAndInnerPoint(upperRight, midPoint, extraSpace) +
             calculateOuterAndInnerPoint(rightMid, midPoint, extraSpace) +
             calculateOuterAndInnerPoint(lowerRight, midPoint, extraSpace), dtype="float64")

        dstP = np.array(
            calculateOuterAndInnerPoint([targetCard[2], targetCard[0]], targetMid, extraSpace) +
             calculateOuterAndInnerPoint([targetCard[2], targetMid[1]], targetMid, extraSpace) +
             calculateOuterAndInnerPoint([targetCard[2], targetCard[1]], targetMid, extraSpace) +
             calculateOuterAndInnerPoint([targetMid[0], targetCard[1]], targetMid, extraSpace) +
             targetMid +
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

        warped = img_as_ubyte(warp(bordered, tform, output_shape=(round(targetHeight+targetOffsetY*2), round(targetWidth+targetOffsetX*2))))

        # for row in warped:
        #     for col in row:
        #         print(col)
        if trim:
            warped = warped[targetCard[0]:targetCard[1], targetCard[2]:targetCard[3]]

        if not border:
            border = [0, 0, 0, 0]
        border = [round(cardHeight * Decimal(border[0])), round(cardHeight * Decimal(border[1])), round(cardWidth * Decimal(border[2])), round(cardWidth * Decimal(border[3]))]

        adjustNeeded = [(targetCard[0] - border[0])*-1, (targetCard[1] + border[1]-1) - warped.shape[0], (targetCard[2] - border[2])*-1, (targetCard[3] + border[3] -1) - warped.shape[1]]
        if debug:
            print(adjustNeeded)
        if any(side < 0 for side in adjustNeeded):
            if debug:
                print(max(0, adjustNeeded[0] * -1))
                print(min(-1, adjustNeeded[1]))
                print(max(0, adjustNeeded[2] * -1))
                print(min(-1, adjustNeeded[3]))
            warped = warped[max(0, adjustNeeded[0] * -1):min(-1, adjustNeeded[1]-1), max(0, adjustNeeded[2] * -1):min(-1, adjustNeeded[3]-1)]
        if any(side > 0 for side in adjustNeeded):
            warped = addBlurredExtendBorder(warped,max(0, border[0]),max(0, border[1]),max(0, border[2]),max(0, border[3]))
        if show:
            return warped, edges, cEdges, cdstV, cdstH
        else:
            return warped
    else:
        print("ERROR: 4 lines not found in image " + baseImg)
        if debug:
            return src, edges, cEdges, cdstV, cdstH


def processMultiArg(arg, numNeeded):
    arg = arg.split(",")
    argList = []
    for num in arg:
        argList.append(float(num))
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

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input" + inputText)
    parser.add_argument("-o", "--Output", help="Set Output" + inputText)
    parser.add_argument("-c", "--Clean", help="Set Clean Images" + inputText)
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
        border = processMultiArg(args.BorderSize, 4)
    if args.Trim:
        trim = bool(args.Trim)
    if args.EdgeFiltering:
        edge = processMultiArg(args.EdgeFiltering, 4)
    if args.Resolution:
        res = processMultiArg(args.Resolution, 2)
    return input, clean, output, border, trim, edge, res, debug, show


def resolveImage(input, clean, output, border, trim, edge, res, debug, show):
    print("processing " + input)
    if show:
        image, dst, cdst, cdstV, cdstH = processImage(input, clean, border, trim, edge, res, debug, show)
        cv.imshow("edges", dst)
        cv.imshow("Horizontal Lines - Standard Hough Line Transform", cdstH)
        cv.imshow("Vertical Lines - Standard Hough Line Transform", cdstV)
        cv.imshow("4 main lines - Probabilistic Line Transform", cdst)
        cv.imshow("outputed", image)
        cv.waitKey()
    else:
        image = processImage(input, clean, border, trim, edge, res, debug, show)
        if image is not None:
            cv.imwrite(output, image)


def main():
    input, clean, output, border, trim, edge, res, debug, show = processArgs("folder")
    if not input:
        input = os.path.join(os.getcwd(), "input")
    if not clean:
        clean = os.path.join(os.getcwd(), "temp")

    with os.scandir(input) as entries:
        for entry in entries:
            if entry.is_file() and entry.name != "Place Images Here":
                imgname, extension = os.path.splitext(os.path.basename(entry.name))
                cleanPath = os.path.join(clean, imgname + ".png")
                outputPath = os.path.join(output, imgname + ".png")
                inputPath = os.path.join(input, entry.name)
                resolveImage(inputPath, cleanPath, outputPath, border, trim, edge, res, debug, show)


if __name__ == "__main__":
    main()
