import main
import os


input, clean, output, border, trim, edge, res, mask, debug, show = main.processArgs("folder")

if not input:
    input = os.path.join(os.getcwd(), "input/test.jpg")
if not clean:
    clean = os.path.join(os.getcwd(), "temp/test.png")

main.resolveImage(input, clean, output, border, trim, edge, res, mask, debug, show)
