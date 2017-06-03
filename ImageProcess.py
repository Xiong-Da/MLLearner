import cv2

SIZEX=300
SIZEY=300

def readImage(path):
    image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    return cv2.resize(image,(SIZEX,SIZEY))

def binaryImage(image,threshhold):
    return cv2.threshold(image,threshhold,255,cv2.THRESH_BINARY)[1]