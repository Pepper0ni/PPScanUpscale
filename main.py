import math
import cv2 as cv
import numpy as np
import os
import argparse
from imutils import perspective

def angleToColour(rad):
    degrees = abs(rad)*180/np.pi
    angle = (degrees+45)%90-45
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

def processLines(img, lines, axis, debug):
    drawImg = np.copy(img)
    maxLine = None
    minLine = None
    imgsize = (img.shape[1], img.shape[0])
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
                cv.line(drawImg, int1, int2, angleToColour(theta), 1, cv.LINE_AA)
                print(imgsize)
                print(pt1)
                print(pt2)
            pt1, pt2 = trimLine(pt1, pt2, imgsize[0], imgsize[1])
            if (pt1[axis] > imgsize[axis] - 30 and pt2[axis] > imgsize[axis] - 30) and (not maxLine or max(maxLine["processed"][0][axis], maxLine["processed"][1][axis]) > max(pt1[axis], pt2[axis]) or \
                    max(maxLine["processed"][0][axis], maxLine["processed"][1][axis]) == max(pt1[axis], pt2[axis]) and \
                    maxLine["processed"][0][axis] + maxLine["processed"][1][axis] > pt1[axis] + pt2[axis]):
                maxLine = {"processed": [pt1, pt2], "line": lines[i]}
            if (pt1[axis] < 30 and pt2[axis] < 30) and (not minLine or min(minLine["processed"][0][axis], minLine["processed"][1][axis]) < min(pt1[axis], pt2[axis]) or \
                    min(minLine["processed"][0][axis], minLine["processed"][1][axis]) == min(pt1[axis], pt2[axis]) and \
                    minLine["processed"][0][axis] + minLine["processed"][1][axis] < pt1[axis] + pt2[axis]):
                minLine = {"processed": [pt1, pt2], "line": lines[i]}
    return drawImg, maxLine, minLine

def processImage(baseImg, cleanImg, debug=False):

    src = cv.imread(cv.samples.findFile(baseImg))
    clean = cv.imread(cv.samples.findFile(cleanImg))#, cv.IMREAD_GRAYSCALE)

    if src is None:
        print('Base Image at ' + baseImg + ' Not Found, skipping')
        return
    if clean is None:
        print('Clean Image at ' + cleanImg + ' Not Found, attempting with base image')
        clean = cv.imread(cv.samples.findFile(baseImg))

    dst = cv.Canny(clean, 25, 100, True, 5)

    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    threshold = 440
    while True:
        horiLines = cv.HoughLines(dst, 1, np.pi / 2880, threshold, None, 0,0, np.pi*0.47,np.pi*0.53)
        cdstH, upperLine, lowerLine = processLines(cdst, horiLines, 1, debug)
        if upperLine and lowerLine:
            break
        threshold -= 20
        if threshold <= 0:
            break
    threshold = 550
    while True:
        vertLines = cv.HoughLines(dst, 1, np.pi / 2880, threshold, None, 0, 0, np.pi * 0.97, np.pi * 1.03)
        cdstV, leftLine, rightLine = processLines(cdst, vertLines, 0, debug)
        if leftLine and rightLine:
            break
        threshold -= 20
        if threshold <= 0:
            break

    if lowerLine and leftLine and upperLine and rightLine:
        if debug:
            print(abs(lowerLine["line"][0][1])*180/np.pi)
            print(lowerLine["processed"])
            cv.line(cdst, [int(lowerLine["processed"][0][0]), int(lowerLine["processed"][0][1])],
                    [int(lowerLine["processed"][1][0]), int(lowerLine["processed"][1][1])], angleToColour(lowerLine["line"][0][1]), 1, cv.LINE_AA)
            print(abs(leftLine["line"][0][1]) * 180 / np.pi)
            print(leftLine["processed"])
            cv.line(cdst, [int(leftLine["processed"][0][0]), int(leftLine["processed"][0][1])],
                    [int(leftLine["processed"][1][0]), int(leftLine["processed"][1][1])], angleToColour(leftLine["line"][0][1]), 1,cv.LINE_AA)
            print(abs(upperLine["line"][0][1]) * 180 / np.pi)
            print(upperLine["processed"])
            cv.line(cdst, [int(upperLine["processed"][0][0]), int(upperLine["processed"][0][1])],
                    [int(upperLine["processed"][1][0]), int(upperLine["processed"][1][1])], angleToColour(upperLine["line"][0][1]), 1,cv.LINE_AA)
            print(abs(rightLine["line"][0][1]) * 180 / np.pi)
            print(rightLine["processed"])
            cv.line(cdst, [int(rightLine["processed"][0][0]), int(rightLine["processed"][0][1])],
                    [int(rightLine["processed"][1][0]), int(rightLine["processed"][1][1])],
                    angleToColour(rightLine["line"][0][1]), 1, cv.LINE_AA)
        upperRight = intersect(upperLine["processed"][0],upperLine["processed"][1], rightLine["processed"][0],rightLine["processed"][1])
        upperLeft = intersect(upperLine["processed"][0],upperLine["processed"][1], leftLine["processed"][0],leftLine["processed"][1])
        lowerRight = intersect(lowerLine["processed"][0],lowerLine["processed"][1], rightLine["processed"][0],rightLine["processed"][1])
        lowerLeft = intersect(lowerLine["processed"][0],lowerLine["processed"][1], leftLine["processed"][0],leftLine["processed"][1])

        corners = np.array([upperRight, upperLeft, lowerRight, lowerLeft])
        warped = perspective.four_point_transform(src, corners)

        if debug:
            return warped, dst, cdst, cdstV, cdstH
        else:
            return warped
    else:
        print("ERROR: 4 lines not found in image " + baseImg)


def main():
    input = os.path.join(os.getcwd(), "input")
    clean = os.path.join(os.getcwd(), "temp")
    deborder = os.path.join(os.getcwd(), "debordered")

    msg = "Adding description"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input folder")
    parser.add_argument("-d", "--Deborder", help="Set Deborder folder")
    parser.add_argument("-c", "--Clean", help="Set Clean Images folder")
    parser.add_argument("-o", "--Output", help="Set Output folder")
    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Clean:
        clean = args.Clean
    if args.Deborder:
        deborder = args.Deborder
    if args.Output:
        output = args.Output

    with os.scandir(input) as entries:
        for entry in entries:
            if entry.is_file() and entry.name != "Place Images Here":
                imgname, extension = os.path.splitext(os.path.basename(entry.name))
                cleanPath = os.path.join(clean, imgname + ".png")
                #outputPath = os.path.join(output, imgname + ".png")
                deborderPath = os.path.join(deborder, imgname + ".png")
                image = processImage(os.path.join(input, entry.name), cleanPath)
                if image is not None:
                    cv.imwrite(deborderPath, image)

if __name__ == "__main__":
    main()