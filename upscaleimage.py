import sys
import os
import cv2 as cv
import argparse

def processFolder(upsampler, path, output):
    try:
        os.mkdir(output)
    except FileExistsError:
        pass
    with os.scandir(path) as entries:
        for entry in entries:
            curPath = os.path.join(path, entry.name)
            outputPath = os.path.join(output, entry.name)
            if os.path.isfile(curPath):
                processFile(upsampler, curPath, outputPath)
            elif os.path.isdir(curPath):
                processFolder(upsampler, curPath, outputPath)


def processFile(upsampler, path, outputPath):
    img = cv.imread(path, cv.IMREAD_UNCHANGED)

    try:
        print("Upscaling: " + path)
        output, _ = upsampler.enhance(img, outscale=2)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        print("saving:" + outputPath)
        cv.imwrite(outputPath, output)

def exportImages(esrgan, model, input, outputPath):
    sys.path.insert(0, esrgan)
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    if not model or not os.path.isfile(model):
        print("model not found:" + str(model))
        return

    upsampler = RealESRGANer(
        scale=2,
        model_path=model,
        dni_weight=None,
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=32, scale=2),#todo investigate
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None) #todo add the ability to not use CPU mode

    if os.path.isfile(input):
        processFile(upsampler, input, outputPath)
    elif os.path.isdir(input):
        processFolder(upsampler, input, outputPath)
    else:
        print("image not found:" + str(input))
        return


def processArgs():
    input = None
    output = os.path.join(os.getcwd(), "output")
    esrgan = None
    model = None

    msg = "Improves old pokemon card scans"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-i", "--Input", help="Set Input")
    parser.add_argument("-o", "--Output", help="Set Output")
    parser.add_argument("-e", "--ESRGAN", help="Set ESRGAN")
    parser.add_argument("-m", "--Model", help="Set model")

    args = parser.parse_args()

    if args.Input:
        input = args.Input
    if args.Output:
        output = args.Output
    if args.ESRGAN:
        esrgan = args.ESRGAN
    if args.Model:
        model = args.Model

    return esrgan, model, input, output,

def main():
    esrgan, model, input, outputPath = processArgs()
    exportImages(esrgan, model, input, outputPath)


if __name__ == '__main__':
    main()