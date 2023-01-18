import os
import argparse
import cv2 as cv

def processMask(src, mask):
    #print(src.shape)
    maskImg = cv.imread(cv.samples.findFile(mask), cv.IMREAD_UNCHANGED)
    #print(maskImg.shape)
    src = cv.cvtColor(src, cv.COLOR_BGR2BGRA)
    maskImg = cv.cvtColor(maskImg, cv.COLOR_BGR2BGRA)
    result = cv.bitwise_and(src, maskImg)
    return result


def main():
    input = os.path.join(os.getcwd(), "resized")
    output = os.path.join(os.getcwd(), "output")
    mask = os.path.join(os.getcwd(), "cardmask.png")
    single = False


    msg = "Masks pokemon card scan corners out"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input folder")
    parser.add_argument("-o", "--Output", help="Set Output folder")
    parser.add_argument("-m", "--Mask", help="Set mask path")
    parser.add_argument("-s", "--Single", help="process only a single card instead of a folder, default: False")

    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Output:
        output = args.Output
    if args.Mask:
        mask = args.Mask
    if args.Single:
        single = args.Single

    if single:
        processMask(input, mask, output)
    else:
        with os.scandir(input) as entries:
            for entry in entries:
                if entry.is_file() and entry.name != "Place Images Here":
                    imgname, extension = os.path.splitext(os.path.basename(entry.name))
                    result = processMask(cv.imread(cv.samples.findFile(os.path.join(input, entry.name)), cv.IMREAD_UNCHANGED), mask)
                    if result is not None:
                        cv.imwrite(os.path.join(output, imgname + ".png"), result)
if __name__ == "__main__":
    main()





