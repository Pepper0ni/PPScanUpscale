import cv2 as cv
import numpy as np
from decimal import Decimal
import utilities

TRACKBAR_WINDOW_NAME = "Manual point set"
lowHue = 0
hiHue = 180
lowSat = 0
hiSat = 255
lowVal = 0
hiVal = 255
posDict = {}
declared = False
lineTog = True

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
    cv.setTrackbarPos("HiHue", TRACKBAR_WINDOW_NAME, hiHue)


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
    cv.setTrackbarPos("HiSat", TRACKBAR_WINDOW_NAME, hiSat)


def lowValTrackbar(val):
    global lowVal
    global hiVal
    lowVal = val
    lowVal = min(hiVal - 1, lowVal)
    cv.setTrackbarPos("LowVal", TRACKBAR_WINDOW_NAME, lowVal)


def hiValTrackbar(val):
    global lowVal
    global hiVal
    hiVal = val
    hiVal = max(hiVal, lowVal + 1)
    cv.setTrackbarPos("HiVal", TRACKBAR_WINDOW_NAME, hiVal)


def toggleLines(a, b):
    global lineTog
    if lineTog:
        lineTog = False
    else:
        lineTog = True

def drawLineWithMid(img, pt1, pt2, mid): #draw a 3 point line
    cv.line(img, [round(pt1[0]), round(pt1[1])],
            [round(mid[0]), round(mid[1])], (0, 255, 0), 1, cv.LINE_AA)
    cv.line(img, [round(pt2[0]), round(pt2[1])],
            [round(mid[0]), round(mid[1])], (0, 255, 0), 1, cv.LINE_AA)
    return img

def drawBox(img, upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight): #draw an 8 point box
    drawLineWithMid(img, upLeft, upRight, upMid)
    drawLineWithMid(img, lowLeft, lowRight, lowMid)
    drawLineWithMid(img, upLeft, lowLeft, leftMid)
    drawLineWithMid(img, upRight, lowRight, rightMid)
    return img

def CustomBordersUI(src, upLeft, upMid, upRight, leftMid, rightMid, lowLeft, lowMid, lowRight): #open a UI that allows you to set custom border positions on an image
    global declared
    global posDict
    cv.namedWindow(TRACKBAR_WINDOW_NAME)
    if not upLeft:
        upLeft = [0, 0]
    if not upMid:
        upMid = [src.shape[1]/2, 0]
    if not upRight:
        upRight = [src.shape[1], 0]
    if not leftMid:
        leftMid = [0, src.shape[0] / 2]
    if not rightMid:
        rightMid = [src.shape[1], src.shape[0] / 2]
    if not lowLeft:
        lowLeft = [0, src.shape[0]]
    if not lowMid:
        lowMid = [src.shape[1] / 2, src.shape[0]]
    if not lowRight:
        lowRight = [src.shape[1], src.shape[0]]
    posDict = {"TLX": round(upLeft[0]), "TLY": round(upLeft[1]), "TMX": round(upMid[0]), "TMY": round(upMid[1]),
               "TRX": round(upRight[0]), "TRY": round(upRight[1]), "MLX": round(leftMid[0]), "MLY": round(leftMid[1]),
               "MRX": round(rightMid[0]), "MRY": round(rightMid[1]), "BLX": round(lowLeft[0]), "BLY": round(lowLeft[1]),
               "BMX": round(lowMid[0]), "BMY": round(lowMid[1]), "BRX": round(lowRight[0]), "BRY": round(lowRight[1])}  # set starting position as a dict that can be looped over
    if not declared:
        for point, value in posDict.items():
            if point.endswith("X"):
                maximum = src.shape[1]
            else:
                maximum = src.shape[0]
            cv.createTrackbar(point, TRACKBAR_WINDOW_NAME, value, maximum, globals()[point + "callbackTrackbar"])#set callback dynamically
        cv.createButton("Toggle Lines", toggleLines) #a button for removing the lines, is buried in the menus
        cv.namedWindow("manual preview", cv.WINDOW_NORMAL)
        declared = True
    blank = np.zeros(shape=[3, 600], dtype=np.uint8) #used to give the taskbar window something to attach to
    while True:
        if lineTog:
            boxSrc = drawBox(np.copy(src), [posDict["TLX"], posDict["TLY"]],
                             [posDict["TMX"], posDict["TMY"]],
                             [posDict["TRX"], posDict["TRY"]],
                             [posDict["MLX"], posDict["MLY"]],
                             [posDict["MRX"], posDict["MRY"]],
                             [posDict["BLX"], posDict["BLY"]],
                             [posDict["BMX"], posDict["BMY"]],
                             [posDict["BRX"], posDict["BRY"]])
        else:
            boxSrc = src
        cv.imshow(TRACKBAR_WINDOW_NAME, blank)
        cv.imshow("manual preview", boxSrc)
        key = cv.waitKey(30)
        if key == ord('q') or key == 27: #loop until q is pressed
            return [Decimal(posDict["TLX"]), Decimal(posDict["TLY"])],\
                   [Decimal(posDict["TMX"]), Decimal(posDict["TMY"])],\
                   [Decimal(posDict["TRX"]), Decimal(posDict["TRY"])],\
                   [Decimal(posDict["MLX"]), Decimal(posDict["MLY"])],\
                   [Decimal(posDict["MRX"]), Decimal(posDict["MRY"])],\
                   [Decimal(posDict["BLX"]), Decimal(posDict["BLY"])],\
                   [Decimal(posDict["BMX"]), Decimal(posDict["BMY"])],\
                   [Decimal(posDict["BRX"]), Decimal(posDict["BRY"])]

def HSVFilterUI(clean): #load a UI for setting a custom filter
    hsvClean = cv.cvtColor(clean, cv.COLOR_BGR2HSV)
    global declared
    cv.namedWindow(TRACKBAR_WINDOW_NAME)
    if not declared:
        cv.createTrackbar("LowHue", TRACKBAR_WINDOW_NAME, lowHue, 180, lowHueTrackbar)
        cv.createTrackbar("HiHue", TRACKBAR_WINDOW_NAME, hiHue, 180, hiHueTrackbar)
        cv.createTrackbar("LoSat", TRACKBAR_WINDOW_NAME, lowSat, 255, lowSatTrackbar)
        cv.createTrackbar("HiSat", TRACKBAR_WINDOW_NAME, hiSat, 255, hiSatTrackbar)
        cv.createTrackbar("LoVal", TRACKBAR_WINDOW_NAME, lowVal, 255, lowValTrackbar)
        cv.createTrackbar("HiVal", TRACKBAR_WINDOW_NAME, hiVal, 255, hiValTrackbar)
        cv.createButton("Toggle Lines", toggleLines)
        cv.namedWindow("manual preview", cv.WINDOW_NORMAL)
        declared = True
    else:
        cv.setTrackbarPos("LowHue", TRACKBAR_WINDOW_NAME, lowHue)
        cv.setTrackbarPos("HiHue", TRACKBAR_WINDOW_NAME, hiHue)
        cv.setTrackbarPos("LoSat", TRACKBAR_WINDOW_NAME, lowSat)
        cv.setTrackbarPos("HiSat", TRACKBAR_WINDOW_NAME, hiSat)
        cv.setTrackbarPos("LoVal", TRACKBAR_WINDOW_NAME, lowVal)
        cv.setTrackbarPos("HiVal", TRACKBAR_WINDOW_NAME, hiVal)
    blank = np.zeros(shape=[3, 600], dtype=np.uint8)

    while True:
        newClean = cv.inRange(hsvClean, (lowHue, lowSat, lowVal), (hiHue, hiSat, hiVal))
        preview = np.bitwise_and(cv.medianBlur(newClean, 3)[:, :, np.newaxis], clean)
        cv.imshow(TRACKBAR_WINDOW_NAME, blank)
        cv.imshow("manual preview", preview)
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            return newClean