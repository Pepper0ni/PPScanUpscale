import cv2 as cv
import numpy as np
import os
import argparse
from contextlib import suppress
from decimal import Decimal
import math

def trimImage(img, fromTop, newBot, fromLeft, newRight): #crop the image based on the supplied values
    return img[fromTop:newBot, fromLeft:newRight]

def pasteImage(base, sprite, posX, posY): #paste 1 image into another in the specified position
    base[round(posY):round(sprite.shape[0] + posY), round(posX):round(sprite.shape[1] + posX), :] = sprite
    return base

def argmedian(x):
    print(x)
    return np.argpartition(x, 1, 2)

def BGR2HSY(img):
    print(img[287,287])
    B, G, R = cv.split(img.astype(float))
    H = np.zeros_like(B)
    S = np.copy(H)

    condMask = np.logical_and(np.greater_equal(R, G), np.greater_equal(G, B))
    np.subtract(R, B, S, where=condMask)
    condMask = np.logical_and(condMask, S)
    np.divide((G-B)*60, S, H, where=condMask)

    condMask = np.logical_and(np.greater(G, R), np.greater_equal(R, B))
    np.subtract(G, B, S, where=condMask)
    np.divide((G - R)*60, S, H, where=condMask)
    np.add(H, 60, H, where=condMask)

    condMask = np.logical_and(np.greater_equal(G, B), np.greater(B, R))
    np.subtract(G, R, S, where=condMask)
    np.divide((B-R)*60, S, H, where=condMask)
    np.add(H, 120, H, where=condMask)

    condMask = np.logical_and(np.greater(B, G), np.greater(G, R))
    np.subtract(B, R, S, where=condMask)
    np.divide((B-G)*60, S, H, where=condMask)
    np.add(H, 180, H, where=condMask)

    condMask = np.logical_and(np.greater(B, R), np.greater_equal(R, G))
    np.subtract(B, G, S, where=condMask)
    np.divide((R-G)*60, S, H, where=condMask)
    np.add(H, 240, H, where=condMask)

    condMask = np.logical_and(np.greater_equal(R, B), np.greater(B, G))
    np.subtract(R, G, S, where=condMask)
    np.divide((R-B)*60, S, H, where=condMask)
    np.add(H, 300, H, where=condMask)

    Y = (B * 0.11) + (G * 0.59) + (R * 0.3)
    H %= 360
    return cv.merge([H, S, Y])

def HSY2BGR(img):
    print(img)
    print(img[287,287])
    H, S, Y = cv.split(img)
    B = np.zeros_like(H)
    G = np.zeros_like(H)
    R = np.zeros_like(H)
    K = np.multiply(S, (H % 60) / 60)
    b = 0.11
    g = 0.59
    r = 0.3

    condMask = np.less(H, 60)
    np.subtract(Y, r * S + g * K, B, where=condMask)
    np.add(B, S, R, where=condMask)
    np.add(B, K, G, where=condMask)
    print(cv.merge([B, G, R])[287,287])
    condMask = np.logical_and(np.greater_equal(H, 60), np.less(H, 120))
    np.add(Y, b * S + r * K, G, where=condMask)
    np.subtract(G, S, B, where=condMask)
    np.subtract(G, K, R, where=condMask)
    print(cv.merge([B, G, R])[287,287])
    condMask = np.logical_and(np.greater_equal(H, 120), np.less(H, 180))
    np.subtract(Y, g * S + b * K, R, where=condMask)
    np.add(R, S, G, where=condMask)
    np.add(R, K, B, where=condMask)
    print(cv.merge([B, G, R])[287,287])
    condMask = np.logical_and(np.greater_equal(H, 180), np.less(H, 240))
    np.add(Y, r * S + g * K, B, where=condMask)
    np.subtract(B, S, R, where=condMask)
    np.subtract(B, K, G, where=condMask)
    print(cv.merge([B, G, R])[287,287])
    condMask = np.logical_and(np.greater_equal(H, 240), np.less(H, 300))
    np.subtract(Y, b * S + r * K, G, where=condMask)
    np.add(G, S, B, where=condMask)
    np.add(G, K, R, where=condMask)
    print(cv.merge([B, G, R])[287,287])
    condMask = np.greater_equal(H, 300)
    np.add(Y, g * S + b * K, R, where=condMask)
    np.subtract(R, S, G, where=condMask)
    np.subtract(R, K, B, where=condMask)
    RGB = cv.merge([B, G, R])

    print((cv.normalize(RGB, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)[287,287]))

    return cv.normalize(RGB, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def varianceColorTransfer(img, doner):
    doner = cv.resize(doner, (img.shape[1], img.shape[0]))
    donerRe = np.reshape(doner, (doner.shape[1]*doner.shape[0], doner.shape[2])).astype(float)
    imgRe = np.reshape(img, (img.shape[1] * img.shape[0], img.shape[2])).astype(float)
    donerVar = np.var(donerRe, 0)
    imgVar = np.var(imgRe, 0)
    # print(donerVar)
    # print(imgVar)
    powers = donerVar / imgVar
    #np.log(np.subtract(donerRangeH, donerRangeL)) / np.log(np.subtract(imgRangeH, imgRangeL)) #/
    #print(powers)
    scaledImg = np.multiply(img, powers[np.newaxis: np.newaxis:])
    simgRe = np.reshape(scaledImg, (scaledImg.shape[1] * scaledImg.shape[0], scaledImg.shape[2]))
    #simgVar = np.var(simgRe, 0)
    #print(simgRe)
    #print(simgVar)
    meanDiff = np.mean(donerRe - simgRe, 0)# + ((127.5 * powers) - 127.5)
    print(meanDiff)
    # print(np.mean(np.reshape(doner, (doner.shape[1]*doner.shape[0], doner.shape[2])),0))
    # print(np.mean(np.reshape(img, (img.shape[1]*img.shape[0], img.shape[2])), 0))
    #newImg = np.power())
    scaledImg = scaledImg + meanDiff[np.newaxis: np.newaxis:]
    scaledImg[scaledImg > 255] = 255
    scaledImg[scaledImg < 0] = 0
    return cv.normalize(scaledImg, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def BGR2BGRSV(img):
    img = np.array(img, dtype=float)
    V = np.max(img, 2)[:, :, np.newaxis]
    min = np.min(img, 2)[:, :, np.newaxis]
    diff = V - min
    S = np.full(shape=(V.shape[0], V.shape[1], 1), fill_value=0, dtype=float)
    S = np.divide(diff, V, S, where=V != 0)*255
    newBGR = img + 127.5 - ((diff/2) + min)
    return np.dstack((newBGR, S, V))

def BGR2Nine(img):
    img = BGR2BGRSV(img)
    iB, iG, iR, iS, iV = cv.split(img)
    print(img)
    C = np.minimum(iG, iB)
    D = np.minimum(iR, C) * 2
    B = np.maximum(0, iB - np.maximum(iG, iR))
    G = np.maximum(0, iG - np.maximum(iB, iR))
    R = np.maximum(0, iR - np.maximum(iG, iB))
    C = np.maximum(0, C - iR)
    M = np.maximum(0, np.minimum(iB, iR) - iG)
    Y = np.maximum(0, np.minimum(iG, iR) - iB)
    return np.dstack((D, B, G, R, C, Y, M, iS, iV))

def BGRSV2BGR(img):
    #print(img)
    B, G, R, S, V = cv.split(img)
    BGR = np.dstack((B, G, R))
    # cv.imshow("outputed", BGR/255)
    # cv.waitKey()
    #np.minimum(S, V, S, where=S > V)
    min = V - ((S/255)*V)
    #print(np.min(min))
    BGRmed = np.median(BGR, 2)
    #print(BGRmed)
    BGRMin = np.min(BGR, 2)
    #print(BGRMin)
    mid = np.copy(min)
    temp = np.divide(np.max(BGR, 2)-BGRMin, BGRmed-BGRMin, mid, where=BGRmed-BGRMin != 0)
    np.divide(V - min, temp, mid, where=temp != 0)
    mid = np.add(mid, min)
    #print(mid)
    newImg = np.zeros_like(BGR)
    midVar = argmedian(BGR)
    unsorted = np.dstack((min, mid, V))
    #print(midVar)
    np.put_along_axis(newImg, midVar, unsorted, 2)
    #print(newImg)
    return newImg.astype(np.uint8)

def Nine2BGR(img):
    iD, iB, iG, iR, iC, iY, iM, iS, iV = cv.split(img)
    iD = iD / 2
    B = iD + iB + iC + iM
    G = iD + iG + iC + iY
    R = iD + iR + iY + iM
    print(np.dstack((B, G, R, iS, iV)))
    return BGRSV2BGR(np.dstack((B, G, R, iS, iV)))

def BGR2CYMK(img):
    # img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # img = Image.fromarray(img)
    # img = img.convert("CMYK")
    # return np.array(img)
    #img = np.array(img, dtype=float)/255
    #with np.errstate(divide="ignore", invalid="ignore"):
    J = np.max(img, axis=2)
    print(J)
    K = 255 - J
    J = np.array(J, dtype=float)+0.001
    img = np.array(img, dtype=float)
    C = 255 * (J - img[..., 2]) / J
    M = 255 * (J - img[..., 1]) / J
    Y = 255 * (J - img[..., 0]) / J
    print(np.dstack((C, M, Y, K)).astype(np.uint8))
    return np.dstack((C, M, Y, K)).astype(np.uint8)

def CYMK2BGR(img):
    # img = Image.fromarray(img, mode="CMYK")
    # print(img.mode)
    # img = img.convert("LAB")
    # return cv.cvtColor(np.array(img), cv.COLOR_BGR2LAB
    img = np.array(img, dtype=float)
    R = 255 * (1.0 - img[..., 0] / float(1)) * (1.0 - img[..., 3] / float(1))
    G = 255 * (1.0 - img[..., 1] / float(1)) * (1.0 - img[..., 3] / float(1))
    B = 255 * (1.0 - img[..., 2] / float(1)) * (1.0 - img[..., 3] / float(1))
    return (np.dstack((B, G, R))).astype(np.uint8)

def findNearestTileSize(axisSize, tolerance, idealSize, minSize):
    sizeMod = axisSize % idealSize
    print(sizeMod)
    if sizeMod <= tolerance:
        return idealSize, sizeMod
    curOffset = 1
    while (True):
        sizeMod = axisSize % (idealSize + curOffset)
        if sizeMod <= tolerance:
            return idealSize + curOffset, sizeMod

        if idealSize - curOffset >= minSize:
            sizeMod = axisSize % (idealSize - curOffset)
            if sizeMod <= tolerance:
                return idealSize - curOffset, sizeMod

        curOffset += 1

def getTileSize(img, idealSize, tolerance):
    y, yExcess = findNearestTileSize(img.shape[0], tolerance, idealSize, round(idealSize/2))
    x, xExcess = findNearestTileSize(img.shape[1], tolerance, idealSize, round(idealSize / 2))
    return x, xExcess, y, yExcess

def tiler(img, doner, smooth):
    xTile, xExcess, yTile, yExcess = getTileSize(img, 70, 4)
    print(img.shape)
    reDoner = cv.resize(doner, (img.shape[1], img.shape[0]))
    print(reDoner.shape)
    fillNum = (smooth+1) ** 2
    tileMask = np.full(shape=(int(yTile+smooth/2), int(xTile+smooth/2)), fill_value=fillNum, dtype=float)
    finalPCAImg = np.full(shape=img.shape, fill_value=0, dtype=np.uint8)
    finalCholImg = np.full(shape=img.shape, fill_value=0, dtype=np.uint8)
    finalSymImg = np.full(shape=img.shape, fill_value=0, dtype=np.uint8)
    for count in range(smooth):
        tileMask[:, count] *= 1 / (smooth + 1) * (count + 1)
        tileMask[:, -(count+1)] *= 1 / (smooth + 1) * (count + 1)

        tileMask[count, :] *= 1 / (smooth + 1) * (count + 1)
        tileMask[-(count+1), :] *= 1 / (smooth + 1) * (count + 1)

    yIter = math.floor(img.shape[0] / yTile)
    xIter = math.floor(img.shape[1] / xTile)
    xExcess1 = math.ceil(xExcess/2)
    xExcess2 = xExcess - xExcess1
    yExcess1 = math.ceil(yExcess/2)
    yExcess2 = yExcess - yExcess1
    print(xIter)
    print(yIter)
    print(xTile)
    print(yTile)
    print(xExcess1)
    print(xExcess2)
    print(yExcess1)
    print(yExcess2)
    for xCount in range(xIter):
        for yCount in range(yIter):
            tileTop = yExcess1 + yCount * yTile
            tileBot = yExcess1 + (yCount + 1) * yTile
            tileLeft = xExcess1 + xCount * xTile
            tileRight = xExcess1 + (xCount +1) * xTile
            print(tileTop)
            if yCount == 0:
                tileTop -= yExcess1
            if yCount == yIter-1:
                tileBot += yExcess2
            if xCount == 0:
                tileLeft -= xExcess1
            if xCount == xIter-1:
                tileBot += xExcess2

            imgTile = trimImage(img, tileTop, tileBot, tileLeft, tileRight)
            print(imgTile.shape)
            donTile = trimImage(reDoner, tileTop, tileBot, tileLeft, tileRight)
            print(donTile.shape)
            transTilePCA, transTileChol, transTileSym = color_transfer(imgTile, donTile)

            finalPCAImg = pasteImage(finalPCAImg, transTilePCA, tileLeft, tileTop)
            finalCholImg = pasteImage(finalCholImg, transTileChol, tileLeft, tileTop)
            finalSymImg = pasteImage(finalSymImg, transTileSym, tileLeft, tileTop)

    return finalPCAImg.astype(float), finalCholImg.astype(float), finalSymImg.astype(float)

def color_transfer(img, doner):
    # Compute the mean and standard deviation of the L*a*b* color channels
    # for the source and target images
    source_mean = img.mean(0).mean(0)
    target_mean = doner.mean(0).mean(0)

    _, target_std = cv.meanStdDev(doner)

    srcUnrav = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    tarUnrav = np.reshape(doner, (doner.shape[0] * doner.shape[1], img.shape[2]))

    # Subtract the means and divide by the standard deviation of the source image
    # to standardize the source image
    srcUnrav = (srcUnrav - source_mean)
    tarUnrav = (tarUnrav - target_mean)

    # Compute the covariance matrix of the source image in L*a*b* color space
    source_cov = np.cov(srcUnrav.T)
    tarCov = np.cov(tarUnrav.T)
    #print(source_cov)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    srcEvalues, srcEvectors = np.linalg.eigh(source_cov)
    tarEvalues, tarEvectors = np.linalg.eigh(tarCov)
    #print("values" + str(srcEvalues))
    #print(values)
    #print("vectors" + str(srcEvectors))
    #print(np.fliplr(vectors.T))

    # multiply the vectors by the square root of the values, and ???
    QsUnk = (srcEvectors * np.sqrt(np.abs(srcEvalues))).dot(srcEvectors.T) + 1e-5 * np.eye(tarUnrav.shape[1])
    QtUnk = (tarEvectors * np.sqrt(np.abs(tarEvalues))).dot(tarEvectors.T) + 1e-5 * np.eye(tarUnrav.shape[1])
    # print(srcUnrav.T)
    # print(QsUnk)
    transformed = QtUnk.dot(np.linalg.inv(QsUnk)).dot(srcUnrav.T)
    #print(transformed)
    matched_img = transformed.reshape(*img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += target_mean
    matched_img[matched_img > 255] = 255
    matched_img[matched_img < 0] = 0

    # Convert the transformed image back to the BGR color space
    PCAImage = np.copy(matched_img)

    chol_t = np.linalg.cholesky(tarCov + 1e-5 * np.eye(srcUnrav.shape[1]))
    chol_s = np.linalg.cholesky(source_cov + 1e-5 * np.eye(tarUnrav.shape[1]))
    #print(chol_s)
    transformed = chol_t.dot(np.linalg.inv(chol_s)).dot(srcUnrav.T)
    #print(transformed)
    matched_img = transformed.reshape(*img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += target_mean
    matched_img[matched_img > 255] = 255
    matched_img[matched_img < 0] = 0
    CholImage = np.copy(matched_img)

    #print(tarCov)
    #print(QsUnk)
    Qt_Cs_Qt = QsUnk.dot(tarCov).dot(QsUnk)
    #print(Qt_Cs_Qt)
    eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
    QtCsQt = eve_QtCsQt.dot(np.sqrt(abs(np.diag(eva_QtCsQt)))).dot(eve_QtCsQt.T)
    transformed = np.linalg.inv(QsUnk).dot(QtCsQt).dot(np.linalg.inv(QsUnk)).dot(srcUnrav.T)
    matched_img = transformed.reshape(*img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += target_mean
    matched_img[matched_img > 255] = 255
    matched_img[matched_img < 0] = 0
    SymImage = np.copy(matched_img)

    return PCAImage, CholImage, SymImage

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
    output = os.path.join(os.getcwd(), "output")
    doner = None

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input" + inputText)
    parser.add_argument("-o", "--Output", help="Set Output" + inputText)
    parser.add_argument("-d", "--Doner", help="Set Doner image" + inputText)
    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Output:
        output = args.Output
    if args.Doner:
        doner = args.Doner
    return input, doner, output


def resolveImage(input, doner, output):
    print("processing " + input)
    inputImg = cv.imread(cv.samples.findFile(input))
    donerImg = cv.imread(cv.samples.findFile(doner))
    result = varianceColorTransfer(inputImg, donerImg)
    reDoner = cv.resize(donerImg, (result.shape[1], result.shape[0]))
    dH, dS, dY = cv.split(BGR2HSY(reDoner))
    rH, rS, rY = cv.split(BGR2HSY(result))
    result = HSY2BGR(cv.merge([rH, dS, rY]))
    result = varianceColorTransfer(result, donerImg)

    if result is not None:
        cv.imwrite(output, result)
    # print(inputImg)
    # cv.imshow("outputed", HSY2BGR(BGR2HSY(inputImg)))
    # cv.waitKey()
    # cv.imshow("outputed", Nine2BGR(BGR2Nine(inputImg)))
    # cv.waitKey()
    # inputImg = cv.cvtColor(inputImg, cv.COLOR_BGR2LAB)
    # LabDonerImg = cv.cvtColor(donerImg, cv.COLOR_BGR2LAB)
    #inputImg = BGR2BGRSV(inputImg)
    #donerImg = BGR2BGRSV(donerImg)
    # B, G, R, S, V = cv.split(donerImg)
    # BGR = np.dstack((B, G, R))
    # cv.imshow("outputed", BGR/255)
    # cv.waitKey()
    #PCAImage, CholImage, SymImage = tiler(inputImg, donerImg, 4)
    #PCAImage, CholImage, SymImage = color_transfer(inputImg, LabDonerImg)
    #PCAImage, CholImage, SymImage = color_transfer(inputImg, donerImg)
    #print(PCAImage)
    # PCAImage = BGR2Nine(cv.cvtColor(cv.normalize(PCAImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), cv.COLOR_LAB2BGR))
    # CholImage = BGR2Nine(cv.cvtColor(cv.normalize(CholImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), cv.COLOR_LAB2BGR))
    # SymImage = BGR2Nine(cv.cvtColor(cv.normalize(SymImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), cv.COLOR_LAB2BGR))
    # donerImg = BGR2Nine(donerImg)
    # #img = BGR2Nine(inputImg)
    # #PCAImage = BGR2Nine(PCAImage)
    # newPCAImage, newCholImage, newSymImage = tiler(PCAImage, donerImg, 4)
    # #PCAImage, CholImage, SymImage = color_transfer(img, donerImg)
    # D, B, G, R, C, Y, M, S, V = cv.split(PCAImage)
    # nD, nB, nG, nR, nC, nY, nM, nS, nV = cv.split(newPCAImage)
    # print(G)
    # print(nG)
    # PCAImage = cv.merge([D, B, G, R, C, Y, M, nS, nV])
    # D, B, G, R, C, Y, M, S, V = cv.split(CholImage)
    # nD, nB, nG, nR, nC, nY, nM, nS, nV = cv.split(newCholImage)
    # CholImage = cv.merge([D, B, G, R, C, Y, M, nS, nV])
    # D, B, G, R, C, Y, M, S, V = cv.split(SymImage)
    # nD, nB, nG, nR, nC, nY, nM, nS, nV = cv.split(newSymImage)
    # SymImage = cv.merge([D, B, G, R, C, Y, M, nS, nV])

    # if PCAImage is not None:
    #     #cv.imwrite(output + "NCA.png", Nine2BGR(PCAImage))
    #     #cv.imwrite(output + "PCA.png", CYMK2BGR(PCAImage))
    #     cv.imwrite(output + "PCA.png", cv.cvtColor(cv.normalize(PCAImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), cv.COLOR_LAB2BGR))
    #     #cv.imwrite(output + "PCA.png", BGRSV2BGR(PCAImage))
    # if CholImage is not None:
    #     #cv.imwrite(output + "chol.png", Nine2BGR(CholImage))
    #     #cv.imwrite(output + "chol.png", CYMK2BGR(CholImage))
    #     cv.imwrite(output + "chol.png", cv.cvtColor(cv.normalize(CholImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), cv.COLOR_LAB2BGR))
    # if SymImage is not None:
    #     #cv.imwrite(output + "sym.png", Nine2BGR(SymImage))
    #     #cv.imwrite(output + "sym.png", CYMK2BGR(SymImage))
    #     cv.imwrite(output + "sym.png", cv.cvtColor(cv.normalize(SymImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), cv.COLOR_LAB2BGR))


def processFolder(input, doner, output):
    with suppress(FileExistsError):
        os.mkdir(output)
    with os.scandir(input) as entries:
        for entry in entries:
            inputPath = os.path.join(input, entry.name)
            outputPath = os.path.join(output, entry.name)
            if os.path.isfile(inputPath) and entry.name != "Place Images Here":
                resolveImage(inputPath, doner, outputPath)
            elif os.path.isdir(inputPath):
                processFolder(inputPath, doner, outputPath)


def main():
    input, doner, output = processArgs("folder")
    if not doner or not os.path.isfile(doner):
        print("Doner file not found.")
        return
    if os.path.isfile(input):
        resolveImage(input, doner, output)
    elif os.path.isdir(input):
        processFolder(input, doner, output)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()

