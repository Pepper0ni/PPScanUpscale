import os
import argparse
import cv2 as cv
import utilities

count = 0

def cutBorders(input, output, size):
    global count
    top = utilities.trimImage(input, 0, size, 0, input.shape[1])
    bot = utilities.trimImage(input, input.shape[0] - size, input.shape[0], 0, input.shape[1])
    left = utilities.trimImage(input, 0, input.shape[0], 0, size)
    right = utilities.trimImage(input, 0, input.shape[0], input.shape[1] - size, input.shape[1])
    bot = cv.cvtColor(bot, cv.COLOR_BGR2BGRA)
    #cv.imshow("outputed " + str(count), bot)
    print(os.path.join(output, "top", str(count) + ".png"))
    cv.imwrite(os.path.join(output, "top", str(count) + ".png"), top)
    cv.imwrite(os.path.join(output, "bottom", str(count) + ".png"), bot)
    cv.imwrite(os.path.join(output, "left", str(count) + ".png"), left)
    cv.imwrite(os.path.join(output, "right", str(count) + ".png"), right)
    count += 1


def main():
    input = os.path.join(os.getcwd(), "input")
    output = os.path.join(os.getcwd(), "exBorders")
    size = 27
    msg = "collects borders for use in other images"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input folder")
    parser.add_argument("-o", "--Output", help="Set Output folder")
    parser.add_argument("-s", "--Size", help="Set border size")
    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Output:
        output = args.Output
    if args.Size:
        size = args.Size

    if os.path.isfile(input):
        cutBorders(input, output, size)
    elif os.path.isdir(input):
        with os.scandir(input) as entries:
            for entry in entries:
                if entry.is_file() and entry.name != "Place Images Here":
                    print(output)
                    cutBorders(cv.imread(cv.samples.findFile(os.path.join(input, entry.name)), cv.IMREAD_UNCHANGED), output, size)
if __name__ == "__main__":
    main()





