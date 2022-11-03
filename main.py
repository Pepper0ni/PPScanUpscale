import math
import cv2 as cv
import numpy as np
import os
import argparse
from imutils import perspective

ANGLE_TOLERANCE = 0.02 #in radians
DEFAULT_BORDER_TOLERANCE = 0.0327272727273 #multiplied by y coord

def angleToColour():#rad):
    return (0,255,0)
    degrees = abs(rad)*180/np.pi
    angle = (degrees+45) % 90-45
    if abs(angle) < (2.55/8):
        green = 255 - (abs(angle)*800)
    else:
        green = 0
    if angle > 0:
        red = min(255, angle*1600)
        blue = 0
    elif angle == 0:
        red = 0
        blue = 0
    else:
        blue = min(255, angle*-1600)
        red = 0
    return (blue, green, red)

def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)

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
    return[(ratio * disX)+pt1[0], (ratio * disY)+pt1[1]]

def trimLine(pt1,pt2,maxX,maxY):
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
    maxLine = [None, None]
    minLine = [None, None]
    imgsize = (img.shape[1], img.shape[0])
    if not edge:
        maxEdge = round(DEFAULT_BORDER_TOLERANCE * imgsize[1])
        minEdge = round(DEFAULT_BORDER_TOLERANCE * imgsize[1])
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
            pt1 = (x0 + 10000 * (-b), y0 + 10000 * (a))
            pt2 = (x0 - 10000 * (-b), y0 - 10000 * (a))
            if debug:
                int1 = [int(pt1[0]), int(pt1[1])]
                int2 = [int(pt2[0]), int(pt2[1])]
                cv.line(drawImg, int1, int2, angleToColour(), 1, cv.LINE_AA)
            pt1, pt2 = trimLine(pt1, pt2, imgsize[0], imgsize[1])
            if debug:
                print(pt1)
                print(pt2)
                print(axis)
                print(imgsize[axis] - maxEdge)
                print(minEdge)
            if pt1[axis] > imgsize[axis] - maxEdge and pt2[axis] > imgsize[axis] - maxEdge:
                if not maxLine[0] or maxLine[0][axis] < pt1[axis]:
                    maxLine = (pt1, maxLine[1])
                if not maxLine[1] or maxLine[1][axis] < pt2[axis]:
                    maxLine = (maxLine[0], pt2)

            if pt1[axis] < minEdge and pt2[axis] < minEdge:
                if not minLine or not minLine[0] or minLine[0][axis] > pt1[axis]:
                    minLine = (pt1, minLine[1])
                if not minLine or not minLine[1] or minLine[1][axis] > pt2[axis]:
                    minLine = (minLine[0], pt2)

    if maxLine[0] == None or maxLine[1] == None:
        maxLine = None
    if minLine[0] == None or minLine[1] == None:
        minLine = None
    return drawImg, maxLine, minLine

def processImage(baseImg, cleanImg, border, trim, expand, edge, debug=False):

    src = cv.imread(cv.samples.findFile(baseImg))
    clean = cv.imread(cv.samples.findFile(cleanImg))

    if src is None:
        print('Base Image at ' + baseImg + ' Not Found, skipping')
        return
    if clean is None:
        print('Clean Image at ' + cleanImg + ' Not Found, attempting with base image')
        clean = cv.imread(cv.samples.findFile(baseImg))

    dst = cv.Canny(clean, 25, 200, True, 5)

    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    threshold = 0.733333333333*src.shape[1]
    if debug:
        print(threshold)
    while True:
        horiLines = cv.HoughLines(dst, 1, np.pi / 2880, round(threshold), None, 0,0, np.pi*(0.5-ANGLE_TOLERANCE),np.pi*(0.5+ANGLE_TOLERANCE))
        cdstH, lowerLine, upperLine = processLines(cdst, horiLines, 1, edge, debug)
        if upperLine and lowerLine:
            break
        threshold -= 20
        if threshold <= 0:
            break
    if debug:
        print("H Threshold:" + str(round(threshold)))
    threshold = 0.666666666667*src.shape[1]
    if debug:
        print(threshold)
    while True:
        vertLines =   cv.HoughLines(dst, 1, np.pi / 2880, round(threshold), None, 0, 0, np.pi * (1-ANGLE_TOLERANCE), np.pi * (1+ANGLE_TOLERANCE))
        cdstV, rightLine, leftLine = processLines(cdst, vertLines, 0, edge, debug)
        if rightLine and leftLine:
            break
        threshold -= 20
        if threshold <= 0:
            break
    if debug:
        print("V Threshold:" + str(round(threshold)))

    if lowerLine and rightLine and upperLine and leftLine:
        if debug:
            print(lowerLine)
            cv.line(cdst, [int(lowerLine[0][0]), int(lowerLine[0][1])],
                    [int(lowerLine[1][0]), int(lowerLine[1][1])], angleToColour(), 1,cv.LINE_AA)
            print(rightLine)
            cv.line(cdst, [int(rightLine[0][0]), int(rightLine[0][1])],[int(rightLine[1][0]), int(rightLine[1][1])], angleToColour(), 1, cv.LINE_AA)
            print(upperLine)
            cv.line(cdst, [int(upperLine[0][0]), int(upperLine[0][1])],
                    [int(upperLine[1][0]), int(upperLine[1][1])], angleToColour(), 1,
                    cv.LINE_AA)
            print(leftLine)
            cv.line(cdst, [int(leftLine[0][0]), int(leftLine[0][1])],
                    [int(leftLine[1][0]), int(leftLine[1][1])],angleToColour(), 1, cv.LINE_AA)

        borderToLeave = [border[0],border[1],border[2],border[3]]
        if upperLine[0][1] < border[0] or upperLine[1][1] < border[0]:
            borderToLeave[0] = math.floor(min(upperLine[0][1], upperLine[1][1]))
        if lowerLine[0][1] + borderToLeave[1] >= src.shape[0] or lowerLine[1][1] + borderToLeave[1] >= src.shape[0]:
            borderToLeave[1] = math.floor(src.shape[0] - min(lowerLine[0][1], lowerLine[1][1]))
        if leftLine[0][0] < borderToLeave[2] or leftLine[1][0] < borderToLeave[2]:
            borderToLeave[2] = math.floor(min(leftLine[0][0], leftLine[1][0]))
        if rightLine[0][0] + borderToLeave[3] >= src.shape[1] or rightLine[1][0] + borderToLeave[3] >= src.shape[1]:
            borderToLeave[3] = math.floor(src.shape[1] - min(rightLine[0][0], rightLine[1][0]))
        upperLine[0][1] = max(0, upperLine[0][1] - borderToLeave[0])
        upperLine[1][1] = max(0, upperLine[1][1] - borderToLeave[0])
        lowerLine[0][1] = min(src.shape[0]-1, lowerLine[0][1] + borderToLeave[1])
        lowerLine[1][1] = min(src.shape[0]-1, lowerLine[1][1] + borderToLeave[1])
        leftLine[0][0] = max(0, leftLine[0][0] - borderToLeave[2])
        leftLine[1][0] = max(0, leftLine[1][0] - borderToLeave[2])
        rightLine[0][0] = min(src.shape[1]-1, rightLine[0][0] + borderToLeave[3])
        rightLine[1][0] = min(src.shape[1]-1, rightLine[1][0] + borderToLeave[3])

        print(upperLine)
        print(lowerLine)

        upperLeft = intersect(upperLine[0], upperLine[1], leftLine[0],
                               leftLine[1])
        upperRight = intersect(upperLine[0], upperLine[1], rightLine[0],
                              rightLine[1])
        lowerLeft = intersect(lowerLine[0], lowerLine[1], leftLine[0],
                               leftLine[1])
        lowerRight = intersect(lowerLine[0], lowerLine[1], rightLine[0],
                              rightLine[1])

        upperLeft = (max(upperLeft[0], 0), min(upperLeft[1], src.shape[0] - 1))
        upperRight = (min(upperRight[0], src.shape[1]-1), min(upperRight[1], src.shape[0]-1))
        lowerLeft = (max(lowerLeft[0], 0), max(lowerLeft[1],0))
        lowerRight = (min(lowerRight[0], src.shape[1]-1), max(lowerRight[1],0))


        corners = np.array([upperLeft, upperRight, lowerLeft, lowerRight])
        if trim:
            warped = perspective.four_point_transform(src, corners)
        else:
            warped = np.copy(src)
        if expand:
            warped = cv.cvtColor(warped, cv.COLOR_BGR2BGRA)
            cardSize = (warped.shape[0]-(borderToLeave[0]+borderToLeave[1]), warped.shape[1]-(borderToLeave[2]+borderToLeave[3]))
            expanded = cv.copyMakeBorder(warped, round(cardSize[0]*expand[0]-borderToLeave[0]), round(cardSize[0]*expand[1]-borderToLeave[1]), round(cardSize[1]*expand[2]-borderToLeave[2]), round(cardSize[1]*expand[3]-borderToLeave[3]), cv.BORDER_CONSTANT, (0,0,0,0))
        else:
            expanded = np.copy(warped)

        if debug:
            return expanded, dst, cdst, cdstV, cdstH
        else:
            return expanded
    else:
        print("H Threshold:" + str(round(threshold)))
        print("ERROR: 4 lines not found in image " + baseImg)
        if debug:
            return src, dst, cdst, cdstV, cdstH

def processMultiArg(arg,numNeeded):
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

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input" + inputText)
    parser.add_argument("-d", "--Deborder", help="Set Deborder" + inputText)
    parser.add_argument("-c", "--Clean", help="Set Clean Images" + inputText)
    parser.add_argument("-r", "--border[3]", help="Set the size of the Right border to trim to. default 0")
    parser.add_argument("-l", "--border[2]", help="Set the size of the Left border to trim to. default 0")
    parser.add_argument("-t", "--border[0]", help="Set the size of the Top border to trim to. default 0")
    parser.add_argument("-b", "--BorderSize",
                        help="Set the size of the 4 borders to trim to \n"
                             "Accepts 4 numbers seperated by commas, as so: 't,                b,l,r'\n"
                             "defaults to 0,0,0,0")
    parser.add_argument("-e", "--Expand",
                        help="Adds empty space after each border up to a certain ratio of the card. \n"
                             "Accepts 4 numbers seperated by commas, as so: 't,b,l,r'")
    parser.add_argument("-ef", "--EdgeFiltering",
                        help="customises the filtering of lines too far away from the edge. \n"
                             "Accepts 4 numbers seperated by commas, as so: 't,b,l,r'. \n"
                             "default is Y res dependent, 27 at 800")
    parser.add_argument("-tr", "--Trim", help="decides whether or not to trim and deskew the image. default True")
    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Clean:
        clean = args.Clean
    if args.Deborder:
        deborder = args.Deborder
    if args.BorderSize:
        border = processMultiArg(args.BorderSize, 4)
    if args.Trim:
        trim = bool(args.Trim)
    if args.Expand:
        expand = processMultiArg(args.Expand,4)
    if args.EdgeFiltering:
        edge = processMultiArg(args.EdgeFiltering,4)
    return input, clean, deborder, border, trim, expand, edge

def main():
    input, clean, deborder, border, trim, expand, edge = processArgs("folder")
    if not input:
        input = os.path.join(os.getcwd(), "input")
    if not clean:
        clean = os.path.join(os.getcwd(), "temp")
    if not border:
        border = [0,0,0,0]
    with os.scandir(input) as entries:
        for entry in entries:
            if entry.is_file() and entry.name != "Place Images Here":
                print("processing " + entry.name)
                imgname, extension = os.path.splitext(os.path.basename(entry.name))
                cleanPath = os.path.join(clean, imgname + ".png")
                deborderPath = os.path.join(deborder, imgname + ".png")
                image = processImage(os.path.join(input, entry.name), cleanPath, border, trim, expand, edge)
                if image is not None:
                    cv.imwrite(deborderPath, image)

if __name__ == "__main__":
    main()