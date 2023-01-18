import json
import numpy as np

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