import json
import numpy as np
import cv2 as cv

def trimImage(img, fromTop, newBot, fromLeft, newRight): #crop the image based on the supplied values
    return img[fromTop:newBot, fromLeft:newRight]

def loadJson(filename):
    with open(filename, 'r') as file:
        JSON = json.load(file)
    return JSON

def clipToOne(img):
    img[img < 0] = 0
    img[img > 1] = 1
    return img

def customLog(array, log):
    return np.log(array) / np.log(log)

LUMA_BLUE = 0.114
LUMA_RED = 0.299
LUMA_GREEN = 0.587

def getLuminosity(img):
    B, G, R = cv.split(img)
    return np.multiply(B, 1/3) + np.multiply(G, 1/3) + np.multiply(R, 1/3)

def getLuma(img):
    B, G, R = cv.split(img)
    return np.multiply(B, LUMA_BLUE) + np.multiply(G, LUMA_GREEN) + np.multiply(R, LUMA_RED)

def BGR2HSY(img):
    B, G, R = cv.split(img)
    #B, G, R = cv.split(img.astype(np.double)/255)
    H = np.zeros_like(B)
    C = np.copy(H)
    S = np.copy(H)
    maxS = np.full_like(H, 0.5)
    Ya = np.copy(H)
    rR = np.multiply(R, LUMA_RED)
    bB = np.multiply(B, LUMA_BLUE)
    gG = np.multiply(G, LUMA_GREEN)
    Y = rR + bB + gG

    condMask = np.logical_and(np.greater_equal(R, G), np.greater_equal(G, B))
    np.subtract(R, B, C, where=condMask) #67
    condMask = np.logical_and(condMask, C)
    np.divide(np.subtract(G, B), C, H, where=condMask) #66 / 67 = 0.985074626866


    condMask = np.logical_and(np.greater(G, R), np.greater_equal(G, B))
    np.subtract(G, np.minimum(B, R), C, where=condMask)
    condMask = np.logical_and(condMask, C)
    np.divide(np.subtract(B, R), C, H, where=condMask)
    np.add(H, 2, H, where=condMask)

    condMask = np.logical_and(np.greater(B, R), np.greater(B, G))
    np.subtract(B, np.minimum(G, R), C, where=condMask)
    condMask = np.logical_and(condMask, C)
    np.divide(np.subtract(R, G), C, H, where=condMask)
    np.add(H, 4, H, where=condMask)

    condMask = np.logical_and(np.greater_equal(R, B), np.greater_equal(B, G))
    np.subtract(R, G, C, where=condMask)
    condMask = np.logical_and(condMask, C)
    np.divide(np.subtract(G, B), C, H, where=condMask)
    np.add(H, 6, H, where=condMask)

    np.add(LUMA_RED, np.multiply(LUMA_GREEN, H), maxS, where=np.greater_equal(H, 0)) #0.704525373134 + LUMA_RED = 0.917125373134
    np.subtract(LUMA_RED + LUMA_GREEN, np.multiply(LUMA_RED, H - 1), maxS, where=np.greater_equal(H, 1))
    np.add(LUMA_GREEN, np.multiply(LUMA_BLUE, H - 2), maxS, where=np.greater_equal(H, 2))
    np.subtract(LUMA_GREEN + LUMA_BLUE, np.multiply(LUMA_GREEN, H - 3), maxS, where=np.greater_equal(H, 3))
    np.add(LUMA_BLUE, np.multiply(LUMA_RED, H - 4), maxS, where=np.greater_equal(H, 4))
    np.subtract(LUMA_RED + LUMA_BLUE, np.multiply(LUMA_BLUE, H - 5), maxS, where=np.logical_and(np.greater_equal(H, 5), np.less_equal(H, 6)))

    condMask = np.less_equal(Y, maxS) #0.9782250980392156 > 0.917125373134
    np.multiply(np.divide(Y, maxS, where=np.logical_and(C, condMask)), 0.5, Ya, where=np.logical_and(C, condMask))
    np.divide(Y - maxS, (1 - maxS) * 2, Ya, where=np.logical_and(C, np.logical_not(condMask))) #0.0610997249052 / 0.165749253732 = 0.368627450981 + 0.5 = 0.868627450981
    np.add(Ya, 0.5, Ya, where=np.logical_and(C, np.logical_not(condMask)))

    np.divide(C, Ya*2, S, where=np.logical_and(C, condMask))
    np.divide(C, 2-(Ya*2), S, where=np.logical_and(C, np.logical_not(condMask))) # 0.868627450981 * 2 = 1.73725490196. 2 - 1.73725490196 = 0.26274509804

    return cv.merge([H, np.maximum(S, 0), np.maximum(Y, 0)])

def HSY2BGR(img):
    H, S, Y = cv.split(img)
    maxS = np.zeros_like(H)
    B = np.copy(maxS)
    G = np.copy(maxS)
    R = np.copy(maxS)
    Ya = np.copy(maxS)
    C = np.copy(maxS)
    X = np.copy(maxS)
    M = np.copy(maxS)

    condMask1 = np.logical_or(np.less(H, 1), np.greater_equal(H, 6))
    np.add(LUMA_RED, LUMA_GREEN*H, maxS, where=condMask1)

    condMask2 = np.logical_and(np.greater_equal(H, 1), np.less(H, 2))
    np.subtract(LUMA_RED+LUMA_GREEN, LUMA_RED * (H - 1), maxS, where=condMask2)

    condMask3 = np.logical_and(np.greater_equal(H, 2), np.less(H, 3))
    np.add(LUMA_GREEN, LUMA_BLUE * (H - 2), maxS, where=condMask3)

    condMask4 = np.logical_and(np.greater_equal(H, 3), np.less(H, 4))
    np.subtract(LUMA_GREEN+LUMA_BLUE, LUMA_GREEN * (H - 3), maxS, where=condMask4)
    condMask5 = np.logical_and(np.greater_equal(H, 4), np.less(H, 5))
    np.add(LUMA_BLUE, LUMA_RED * (H - 4), maxS, where=condMask5)

    condMask6 = np.logical_or(np.greater_equal(H, 5), np.less(H, 0))
    np.subtract(LUMA_BLUE + LUMA_RED, LUMA_BLUE * (H - 5), maxS, where=condMask6)

    YmaxSCond = np.less(Y, maxS)
    np.multiply(np.divide(Y, maxS, where=YmaxSCond), 0.5, Ya, where=YmaxSCond)
    np.multiply(S * 2, Ya, C, where=YmaxSCond)
    np.multiply(np.divide(Y, maxS, where=np.logical_and(condMask1, np.equal(Y, maxS))), 0.5, Ya, where=np.logical_and(condMask1, np.equal(Y, maxS)))

    np.multiply(np.divide(Y - maxS, 1 - maxS, where=np.logical_not(YmaxSCond)), 0.5, Ya, where=np.logical_not(YmaxSCond))
    np.add(Ya, 0.5, Ya, where=np.logical_not(YmaxSCond))
    np.multiply(S, 2 - (Ya * 2), C, where=np.logical_not(YmaxSCond))
    np.multiply(1 - np.abs((H % 2) - 1), C, X)

    np.putmask(R, np.logical_or(condMask1, condMask6), C)
    np.putmask(G, np.logical_or(condMask2, condMask3), C)
    np.putmask(B, np.logical_or(condMask4, condMask5), C)

    np.putmask(R, np.logical_or(condMask2, condMask5), X)
    np.putmask(G, np.logical_or(condMask1, condMask4), X)
    np.putmask(B, np.logical_or(condMask3, condMask6), X)

    np.subtract(Y, R * LUMA_RED + G * LUMA_GREEN + B * LUMA_BLUE, M)
    np.add(M, R, R)
    np.add(M, G, G)
    np.add(M, B, B)

    #return cv.normalize(cv.merge([np.maximum(B, 0), np.maximum(G, 0), np.maximum(R, 0)])*255, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return cv.merge([np.maximum(B, 0), np.maximum(G, 0), np.maximum(R, 0)])