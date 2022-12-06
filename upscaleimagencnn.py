import os
import cv2 as cv
import argparse
from ncnn_vulkan import ncnn
import numpy as np
from contextlib import suppress
import re

noiseNet = ncnn.Net()
noiseNet.opt.use_vulkan_compute = True

ditherNet = ncnn.Net()
ditherNet.opt.use_vulkan_compute = True

OVERFLOW = 16

def trimImage(img, fromTop, newBot, fromLeft, newRight): #crop the image based on the supplied values
    return np.copy(img[fromTop:newBot, fromLeft:newRight])

def processImage(input, output):
    img = cv.imread(input)
    xSize = round(img.shape[1] / 2)
    ySize = round(img.shape[0] / 2)

    odds = [[0, 0],
            [img.shape[0] % 2, 0],
            [0, img.shape[1] % 2],
            [img.shape[0] % 2, img.shape[1] % 2],
            ]
    
    tiles = [
        trimImage(img, 0, ySize + OVERFLOW + odds[0][0], 0, xSize + OVERFLOW + odds[0][1]),
        trimImage(img, ySize - (OVERFLOW + odds[1][0]), img.shape[0], 0, xSize + OVERFLOW + odds[1][1]),
        trimImage(img, 0, ySize + OVERFLOW + odds[2][0], xSize - (OVERFLOW + odds[2][1]), img.shape[1]),
        trimImage(img, ySize - (OVERFLOW + odds[3][0]), img.shape[0], xSize - (OVERFLOW + odds[3][1]), img.shape[1]),
    ]
    outputs = []
    for tile in tiles:
        noiseEx = noiseNet.create_extractor()
        ditherEx = ditherNet.create_extractor()
        # Convert image to ncnn Mat
        mat_in = ncnn.Mat.from_pixels(
            tile,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            tile.shape[1],
            tile.shape[0]
        )
        # Normalize image (required)
        # Note that passing in a normalized numpy array will not work.
        mean_vals = []
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        try:
            # Make sure the input and output names match the param file
            noiseEx.input("input", mat_in)
            ret, denoise = noiseEx.extract("output")
            ditherEx.input("input", denoise)
            ret, dedither = ditherEx.extract("output")

            # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
            outputs.append(np.array(dedither).transpose(1, 2, 0) * 255)
            #print(len(outputs)-1)
            #print(outputs[len(outputs)-1].shape)
        except:
            ncnn.destroy_gpu_instance()
    #newOverflow = int(OVERFLOW * 2)
    trimOutputs = [
        trimImage(outputs[0], 0, outputs[0].shape[0] - (OVERFLOW + (odds[0][0])), 0,
                  outputs[0].shape[1] - (OVERFLOW + (odds[0][1]))),
        trimImage(outputs[1], OVERFLOW + (odds[1][0]), outputs[1].shape[0], 0,
                  outputs[0].shape[1] - (OVERFLOW + (odds[1][1]))),
        trimImage(outputs[2], 0, outputs[0].shape[0] - (OVERFLOW + (odds[2][0])), OVERFLOW + (odds[2][1]),
                  outputs[2].shape[1]),
        trimImage(outputs[3], OVERFLOW + (odds[3][0]), outputs[3].shape[0], OVERFLOW + (odds[3][1]),
                  outputs[3].shape[0]),
    ]
    top = cv.hconcat([trimOutputs[0], trimOutputs[2]])
    bot = cv.hconcat([trimOutputs[1], trimOutputs[3]])
    cv.imwrite(output, cv.cvtColor(cv.vconcat([top, bot]), cv.COLOR_RGB2BGR))
    
def processArgs():
    input = os.path.join(os.getcwd(), "input")
    output = os.path.join(os.getcwd(), "output")
    model = None

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input")
    parser.add_argument("-o", "--Output", help="Set Output")
    parser.add_argument("-m", "--Model", help="Set model folder")

    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Output:
        output = args.Output
    if args.Model:
        model = args.Model

    return input, model, output

def resolveImage(input, output):
    print("processing " + input)
    output = re.sub("jpg$", "png", output)
    image = processImage(input, output)
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
    input, model, output,  = processArgs()
    noiseNet.load_param(os.path.join(model, "1x_ISO_denoise_v1.param"))
    noiseNet.load_model(os.path.join(model, "1x_ISO_denoise_v1.bin"))
    ditherNet.load_param(os.path.join(model, "1x_artifacts_dithering_alsa.param"))
    ditherNet.load_model(os.path.join(model, "1x_artifacts_dithering_alsa.bin"))
    if os.path.isfile(input):
        resolveImage(input, output)
    elif os.path.isdir(input):
        processFolder(input, output)
    else:
        print("Input file not found.")


if __name__ == "__main__":
    main()