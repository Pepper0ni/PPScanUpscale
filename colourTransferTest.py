import cv2 as cv
import numpy as np
import os
import argparse
from contextlib import suppress
from decimal import Decimal

def color_transfer(source, target):
    # Convert images from BGR to L*a*b* color space
    source_lab = cv.cvtColor(source, cv.COLOR_BGR2LAB)
    target_lab = cv.cvtColor(target, cv.COLOR_BGR2LAB)

    # Compute the mean and standard deviation of the L*a*b* color channels
    # for the source and target images
    source_mean, source_std = cv.meanStdDev(source_lab)
    target_mean, target_std = cv.meanStdDev(target_lab)

    # Subtract the means and divide by the standard deviation of the source image
    # to standardize the source image
    source_lab = (source_lab - source_mean) / source_std

    # Compute the covariance matrix of the source image in L*a*b* color space
    source_cov = np.cov(source_lab.T)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(source_cov)

    # Perform PCA by projecting the source image onto the eigenspace defined
    # by the eigenvectors of the covariance matrix
    projection = source_lab.dot(eigenvectors)

    # Transform the projected source image by scaling and shifting the colors
    # to match the mean and standard deviation of the target image
    transformed = projection * target_std + target_mean

    # Convert the transformed image back to the BGR color space
    transformed = cv.cvtColor(transformed, cv.COLOR_LAB2BGR)

    return transformed

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
    
    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input" + inputText)
    parser.add_argument("-o", "--Output", help="Set Output" + inputText)
    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Output:
        output = args.Output
    return input, output


def resolveImage(input, output):
    print("processing " + input)
    image = 
    if image is not None:
        cv.imwrite(output, image)


def processFolder(input, output):
    with suppress(FileExistsError):
        os.mkdir(output)
    with os.scandir(input) as entries:
        for entry in entries:
            inputPath = os.path.join(input, entry.name)
            outputPath = os.path.join(output, entry.name)
            if os.path.isfile(inputPath) and entry.name != "Place Images Here":
                resolveImage(inputPath, outputPath)
            elif os.path.isdir(inputPath):
                processFolder(inputPath, outputPath)


def main():
    input, output = processArgs("folder")
    if os.path.isfile(input):
        resolveImage(input, output)
    elif os.path.isdir(input):
        processFolder(input, output)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()

