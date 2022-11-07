import main
import os


input, clean, deborder, border, trim, expand, edge, debug, show = main.processArgs("folder")

if not input:
    input = os.path.join(os.getcwd(), "input/test.jpg")
if not clean:
    clean = os.path.join(os.getcwd(), "temp/test.png")
if not border:
    border = [0,0,0,0]

main.resolveImage(input, clean, deborder, border, trim, expand, edge, debug, show)
