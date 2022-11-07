import math
import cv2 as cv
import numpy as np
import os
import argparse
from decimal import Decimal
from wand.image import Image as wandImage
from skimage.transform import PiecewiseAffineTransform, warp
import skimage.util
from skimage import data

ANGLE_TOLERANCE = 0.025  # in radians
DEFAULT_BORDER_TOLERANCE = 0.0327272727273  # multiplied by y coord

def warpPerspective(image, dstX, dstY, pts, debug):
    # upperMid = ((pts[0][0]+pts[1][0])/2,(pts[0][1]+pts[1][1])/2)
    # leftMid = ((pts[0][0] + pts[3][0]) / 2, (pts[0][1] + pts[3][1]) / 2)
    # lowerMid = ((pts[3][0] + pts[2][0]) / 2, (pts[3][1] + pts[2][1]) / 2)
    # rightMid = ((pts[2][0] + pts[1][0]) / 2, (pts[2][1] + pts[1][1]) / 2)
    output = np.array([
        [0, 0],
        [dstX - 1, 0],
        [dstX - 1, dstY - 1],
        [0, dstY - 1]], dtype="float32")
        # [0, 0],
        # [cv.norm((rightMid[0] - leftMid[0]),(rightMid[1] - leftMid[1])), 0],
        # [cv.norm((rightMid[0] - leftMid[0]),(rightMid[1] - leftMid[1])), cv.norm((upperMid[0] - lowerMid[0]),(upperMid[1] - lowerMid[1]))],
        # [0, cv.norm((upperMid[0] - lowerMid[0]),(upperMid[1] - lowerMid[1]))]], dtype="float32")
    if debug:
        print(pts)
        print(output)
    M = cv.getPerspectiveTransform(pts, output)
    return cv.warpPerspective(image, M, (round(dstX), round(dstY)))


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
    return (line1[0][0] + ua * (line1[1][0] - line1[0][0]), line1[0][1] + ua * (line1[1][1] - line1[0][1]))


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


def processImage(baseImg, cleanImg, border, trim, expand, edge, debug=False, show=False):
    src = cv.imread(cv.samples.findFile(baseImg))
    clean = cv.imread(cv.samples.findFile(cleanImg))

    if src is None:
        print('Base Image at ' + baseImg + ' Not Found, skipping')
        return
    if clean is None:
        print('Clean Image at ' + cleanImg + ' Not Found, attempting with base image')
        clean = cv.imread(cv.samples.findFile(baseImg))

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

    offsetX = Decimal(src.shape[1] / 2) #offsets for combined canvas
    offsetY = Decimal(src.shape[0] / 2)

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

        borderToLeave = [Decimal(border[0]), Decimal(border[1]), Decimal(border[2]), Decimal(border[3])]
        print((upperLine[0][1], upperLine[1][1], upperMid[1] , borderToLeave[0]))
        if upperLine[0][1] < borderToLeave[0] or upperLine[1][1] < borderToLeave[0] or upperMid[1] < borderToLeave[0]:
            borderToLeave[0] = math.floor(min(upperLine[0][1], upperLine[1][1], upperMid[1]))
        if lowerLine[0][1] + borderToLeave[1] >= src.shape[0] or lowerLine[1][1] + borderToLeave[1] >= src.shape[0] or \
                lowerMid[1] + borderToLeave[1] >= src.shape[0]:
            borderToLeave[1] = math.floor(src.shape[0] - max(lowerLine[0][1], lowerLine[1][1], lowerMid[1]))
        if leftLine[0][0] < borderToLeave[2] or leftLine[1][0] < borderToLeave[2] or leftMid[0] < borderToLeave[2]:
            borderToLeave[2] = math.floor(min(leftLine[0][0], leftLine[1][0], leftMid[0]))
        if rightLine[0][0] + borderToLeave[3] >= src.shape[1] or rightLine[1][0] + borderToLeave[3] >= src.shape[1] or \
                rightMid[0] + borderToLeave[3] >= src.shape[1]:
            borderToLeave[3] = math.floor(src.shape[1] - max(rightLine[0][0], rightLine[1][0], rightMid[0]))
        if debug:
            print("borderToLeave: " + str(borderToLeave))

        upperLeft = intersect(upperLine, leftLine)
        upperRight = intersect(upperLine, rightLine)
        lowerLeft = intersect(lowerLine, leftLine)
        lowerRight = intersect(lowerLine, rightLine)

        upperLeft = [upperLeft[0]-borderToLeave[2]+offsetX, upperLeft[1]-borderToLeave[0]+offsetY]
        upperRight = [upperRight[0]+borderToLeave[3]+offsetX, upperRight[1]-borderToLeave[0]+offsetY]
        lowerLeft = [lowerLeft[0]-borderToLeave[2]+offsetX, lowerLeft[1]+borderToLeave[1]+offsetY]
        lowerRight = [lowerRight[0]+borderToLeave[3]+offsetX, lowerRight[1]+borderToLeave[1]+offsetY]
        targetWidth = max(upperRight[0] - upperLeft[0], lowerRight[0] - lowerLeft[0])
        targetHeight = max(lowerRight[1] - upperRight[1], lowerLeft[1] - upperLeft[1])

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

        middlePoint = [round(targetWidth / 2 + borderToLeave[2]), round(targetHeight / 2 + borderToLeave[0])]
        offsetMidPoint = [middlePoint[0] + offsetX, middlePoint[1] + offsetY]
        upperMid = [offsetMidPoint[0], max(0, upperMid[1] - borderToLeave[0])+offsetY]
        lowerMid = [offsetMidPoint[0], min(src.shape[0], lowerMid[1] + borderToLeave[1])+offsetY]
        leftMid = [max(0, leftMid[0] - borderToLeave[2])+offsetX, offsetMidPoint[1]]
        rightMid = [min(src.shape[1], rightMid[0] + borderToLeave[3])+offsetX, offsetMidPoint[1]]


        if debug:
            print("UpperLeft: " + str(upperLeft))
            print("UpperRight: " + str(upperRight))
            print("LowerLeft: " + str(lowerLeft))
            print("LowerRight: " + str(lowerRight))
            print("upperMid: " + str(upperMid))
            print("lowerMid: " + str(lowerMid))
            print("leftMid: " + str(leftMid))
            print("rightMid: " + str(rightMid))
            print("middlePoint: " + str(middlePoint))
            print("targetWidth: " + str(targetWidth))
            print("targetHeight: " + str(targetHeight))

        if trim:

            srcP = np.array( #todo add outer mesh
                [upperLeft, leftMid, lowerLeft, lowerMid, [offsetX + middlePoint[0], offsetY + middlePoint[1]], upperMid, upperRight, rightMid, lowerRight], dtype="float64")
            dstP = np.array([[offsetX,offsetY], [offsetX, middlePoint[1] + offsetY], [offsetX, targetHeight + offsetY],
                             [offsetX + middlePoint[0], targetHeight + offsetY], [offsetX + middlePoint[0], offsetY + middlePoint[1]],
                             [offsetX + middlePoint[0], offsetY], [offsetX+targetWidth, offsetY],
                             [offsetX+targetWidth, middlePoint[1]+offsetY], [targetWidth+offsetX, targetHeight + offsetY]],dtype="float64")
            print(srcP)
            print(dstP)
            double = cv.copyMakeBorder(src, src.shape[0] * 2, src.shape[0] * 2, src.shape[1] * 2, src.shape[1] * 2,
                                       cv.BORDER_CONSTANT, (0, 0, 0, 0))
            print(src)
            double = skimage.util.img_as_ubyte(double)
            double = double[:, :, ::-1]
            tform = PiecewiseAffineTransform()
            tform.estimate(srcP, dstP)
            #cv.imshow("double", double)
            warped = warp(double, tform, output_shape=(src.shape[0]*2, src.shape[1]*2))
            cv.imshow("warped", warped)
            #cv.waitKey()


            # with wandImage(width=src.shape[1]*2, height=src.shape[0]*2) as canvas:
            #     canvas.alpha_channel = "transparent"
            #     canvas.virtual_pixel = "transparent"
            #     canvas.background_color = "transparent"
            #     cutX = middlePoint[0]
            #     cutY = middlePoint[1]
            #     MMPerspective = (float(offsetMidPoint[0]), float(offsetMidPoint[1]), float(offsetX + cutX), float(cutY +offsetY))
            #     LeMperspective = (float(leftMid[0]), float(leftMid[1]), float(offsetX), float(cutY + offsetY))
            #     RMperspective = rightMid + (offsetX+targetWidth, cutY+offsetY)
            #     LoMperspective = lowerMid + (offsetX + cutX, targetHeight + offsetY)
            #     UMperspective = (float(upperMid[0]), float(upperMid[1]), float(offsetX + cutX), float(offsetY))
            #     with wandImage.from_array(src) as img:
            #         img.alpha_channel = True
            #         img.virtual_pixel = "transparent"
            #         img.background_color = "transparent"
            #
            #         wandSrcUL = img[:cutX, :cutY]
            #         wandSrcUL.extent(src.shape[1] * 2, src.shape[0] * 2, round(-offsetX), round(-offsetY))
            #
            #         wandSrcLL = img[:cutX, cutY:]
            #         wandSrcLL.extent(src.shape[1] * 2, src.shape[0] * 2, round(-offsetX), round(-offsetY - cutY))
            #
            #         wandSrcLR = img[cutX:, cutY:]
            #         wandSrcLR.extent(src.shape[1] * 2, src.shape[0] * 2, round(-offsetX - cutX), round(-offsetY - cutY))
            #
            #         wandSrcUR = img[cutX:, :cutY]
            #         wandSrcUR.extent(src.shape[1] * 2, src.shape[0] * 2, round(-offsetX - cutX), round(-offsetY))
            #
            #         print("before perspectives")
            #         print((float(upperLeft[0]), float(upperLeft[1]), float(offsetX), float(offsetY)) + LeMperspective + MMPerspective + UMperspective)
            #         wandSrcUL.distort("perspective", (float(upperLeft[0]), float(upperLeft[1]), float(offsetX), float(offsetY)) + LeMperspective + MMPerspective + UMperspective)
            #         print("ul done")
            #         print(LeMperspective + lowerLeft + (offsetX, cutY + offsetY) + LoMperspective + MMPerspective)
            #         wandSrcLL.distort("perspective", LeMperspective + lowerLeft + (offsetX, targetHeight + offsetY) + LoMperspective + MMPerspective)
            #         print("ll done")
            #         print(MMPerspective + LoMperspective + lowerRight + (cutX+offsetX, cutY+offsetY) + RMperspective)
            #         wandSrcLR.distort("perspective", MMPerspective + LoMperspective + lowerRight + (targetWidth+offsetX, targetHeight + offsetY) + RMperspective)
            #         print("lr done")
            #         print(UMperspective + MMPerspective + RMperspective + upperRight + (cutX+offsetX, offsetY))
            #         wandSrcUR.distort("perspective", UMperspective + MMPerspective + RMperspective + upperRight + (offsetX+targetWidth, offsetY))
            #         print("after perspectives")
            #         canvas.composite(wandSrcUL, 0, 0, "plus")
            #         canvas.composite(wandSrcUR, 0, 0, "plus")
            #         canvas.composite(wandSrcLL, 0, 0, "plus")
            #         canvas.composite(wandSrcLR, 0, 0, "plus")
            #
            #         ULImage = np.array(wandSrcUL)
            #         URImage = np.array(wandSrcUR)
            #         LLImage = np.array(wandSrcLL)
            #         LRImage = np.array(wandSrcLR)
            #         warped = np.array(canvas) #TODO trim
            #         if debug:
            #             cv.imshow("UL", ULImage)
            #             cv.imshow("UR", URImage)
            #             cv.imshow("LL", LLImage)
            #             cv.imshow("LR", LRImage)
            #             cv.imshow("canvas", warped)
            #             # cv.waitKey()

            # ULImage = warpPerspective(src, targetWidth / 2 + borderToLeave[2], targetHeight / 2 + borderToLeave[0],
            #                           np.array([upperLeft, upperMid, middlePoint, leftMid], dtype="float32"), debug)
            # URImage = warpPerspective(src, targetWidth / 2 + borderToLeave[3], targetHeight / 2 + borderToLeave[0],
            #                           np.array([upperMid, upperRight, rightMid, middlePoint], dtype="float32"), debug)
            # LLImage = warpPerspective(src, targetWidth / 2 + borderToLeave[2], targetHeight / 2 + borderToLeave[1],
            #                           np.array([leftMid, middlePoint, lowerMid, lowerLeft], dtype="float32"), debug)
            # LRImage = warpPerspective(src, targetWidth / 2 + borderToLeave[3], targetHeight / 2 + borderToLeave[1],
            #                           np.array([middlePoint, rightMid, lowerRight, lowerMid], dtype="float32"), debug)

            #warped = cv.vconcat([cv.hconcat([ULImage, URImage]), cv.hconcat([LLImage, LRImage])])
        else:
            warped = np.copy(src)
        if expand:
            warped = cv.cvtColor(warped, cv.COLOR_BGR2BGRA)
            cardSize = (warped.shape[0] - (borderToLeave[0] + borderToLeave[1]),
                        warped.shape[1] - (borderToLeave[2] + borderToLeave[3]))
            expanded = cv.copyMakeBorder(warped, round(cardSize[0] * Decimal(expand[0]) - borderToLeave[0]),
                                         round(cardSize[0] * Decimal(expand[1]) - borderToLeave[1]),
                                         round(cardSize[1] * Decimal(expand[2]) - borderToLeave[2]),
                                         round(cardSize[1] * Decimal(expand[3]) - borderToLeave[3]), cv.BORDER_CONSTANT,
                                         (0, 0, 0, 0))
        else:
            expanded = np.copy(warped)

        if show:
            return expanded, edges, cEdges, cdstV, cdstH
        else:
            return expanded
    else:
        print("H Threshold:" + str(round(threshold)))
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
    deborder = os.path.join(os.getcwd(), "debordered")
    border = None
    trim = True
    expand = None
    edge = None
    show = False
    debug = False

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input" + inputText)
    parser.add_argument("-d", "--Deborder", help="Set Deborder" + inputText)
    parser.add_argument("-c", "--Clean", help="Set Clean Images" + inputText)
    parser.add_argument("-de", "--Debug", help="Enable debug prints default False" + inputText)
    parser.add_argument("-b", "--BorderSize",
                        help="Set the size of the 4 borders to trim to \n"
                             "Accepts 4 numbers seperated by commas, as so: 't,b,l,r'\n"
                             "defaults to 0,0,0,0")
    parser.add_argument("-e", "--Expand",
                        help="Adds empty space after each border up to a certain ratio of the card. \n"
                             "Accepts 4 numbers seperated by commas, as so: 't,b,l,r'")
    parser.add_argument("-ef", "--EdgeFiltering",
                        help="customises the filtering of lines too far away from the edge. \n"
                             "Accepts 4 numbers seperated by commas, as so: 't,b,l,r'. \n"
                             "default is Y res dependent, 27 at 800")
    parser.add_argument("-tr", "--Trim", help="decides whether or not to trim and deskew the image. default True")
    parser.add_argument("-s", "--Show", help="show images instead of saving them. default False")

    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Clean:
        clean = args.Clean
    if args.Deborder:
        deborder = args.Deborder
    if args.Debug:
        debug = args.Debug
    if args.Show:
        show = args.Show
    if args.BorderSize:
        border = processMultiArg(args.BorderSize, 4)
    if args.Trim:
        trim = bool(args.Trim)
    if args.Expand:
        expand = processMultiArg(args.Expand, 4)
    if args.EdgeFiltering:
        edge = processMultiArg(args.EdgeFiltering, 4)
    return input, clean, deborder, border, trim, expand, edge, debug, show


def resolveImage(input, clean, deborder, border, trim, expand, edge, debug, show):
    print("processing " + input)
    if show:
        image, dst, cdst, cdstV, cdstH = processImage(input, clean, border, trim, expand, edge, debug, show)
        cv.imshow("edges", dst)
        cv.imshow("Horizontal Lines - Standard Hough Line Transform", cdstH)
        cv.imshow("Vertical Lines - Standard Hough Line Transform", cdstV)
        cv.imshow("4 main lines - Probabilistic Line Transform", cdst)
        cv.imshow("debordered", image)
        cv.waitKey()
    else:
        image = processImage(input, clean, border, trim, expand, edge, debug)
        if image is not None:
            cv.imwrite(deborder, image)


def main():
    input, clean, deborder, border, trim, expand, edge, debug, show = processArgs("folder")
    if not input:
        input = os.path.join(os.getcwd(), "input")
    if not clean:
        clean = os.path.join(os.getcwd(), "temp")
    if not border:
        border = [0, 0, 0, 0]
    with os.scandir(input) as entries:
        for entry in entries:
            if entry.is_file() and entry.name != "Place Images Here":
                imgname, extension = os.path.splitext(os.path.basename(entry.name))
                cleanPath = os.path.join(clean, imgname + ".png")
                deborderPath = os.path.join(deborder, imgname + ".png")
                inputPath = os.path.join(input, entry.name)
                resolveImage(inputPath, cleanPath, deborderPath, border, trim, expand, edge, debug, show)


if __name__ == "__main__":
    main()
