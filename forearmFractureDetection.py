"Appendix A: Project Code"
#!/usr/bin/env python

import cv2 as cv
import numpy as np
import math

# Constant Values for Gaussian blur and Canny Edge Detection
SIGMA=2*math.sqrt(2)/2
LOW_THRESHOLD=50
HIGH_THRESHOLD=100

# Custom function that will allow contours to be sorted by arc length
def customArcLength(args):
    return cv.arcLength(args,True)

# Fracture detection algorithm
def is_fracture(contour):
    x1=contour[0][0][0]
    y1=contour[0][0][1]
    yLength=0
    xLength=0
    lowX = contour[0][0][0]
    highX = contour[0][0][0]
    # Loops through contours to obtain information contour lengths in the y and x direction
    for i in range(contour.shape[0]-1):
        x2=contour[i][0][0]
        y2=contour[i][0][1]
        if(abs(y1-y2)>yLength):
            yLength=abs(y1-y2)
            yIndex=i
            highX=contour[yIndex][0][0]
        if (abs(x1 - x2) > xLength):
            xLength = abs(x1 - x2)

    if(highX-lowX==0):
        fractureRatio=xLength
    else:
        x1Length=float(abs(highX-lowX))
        x2Length=float(xLength)
        # Higher x2 length means that the contour width is inflated past its expected x1 length due to a potential fracture
        fractureRatio=x2Length/x1Length
    if(fractureRatio>2):
        return True
    else:
        return False

# Core algorithm of the project
def show_contours(img):
    # Performs morphological gradient
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((10,10), np.uint8))

    # Applies Gaussian blurring
    filteredImg = cv.sepFilter2D(opening, -1, cv.getGaussianKernel(51, SIGMA), cv.getGaussianKernel(51, SIGMA))
    imgray = cv.cvtColor(filteredImg, cv.COLOR_BGR2GRAY)

    # Applies Canny edge detection
    edges = cv.Canny(imgray, LOW_THRESHOLD, HIGH_THRESHOLD)

    # Applies contour detection
    contours, hier = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Sorts contours by arc length and runs fracture detection algorithm on 3 longest contours
    c1=(sorted(contours,key=customArcLength))
    c1 = c1[::-1]
    finalImg=cv.drawContours(opening, c1, 0, (0, 255, 0), 2)
    #cv.imshow("Display window", cv.drawContours(opening, c1, 0, (0, 255, 0), 2))
    #cv.waitKey(0)
    boneFractured=False
    boneFractureContour=-1
    i=0
    while(i<3 and not boneFractured):
        if(is_fracture(c1[i])):
            boneFractured=True
            boneFractureContour=i
        i=i+1

    # Will display the bone fracture contour if the bone is fractured
    # otherwise it will print that the bone is not fractured
    if(boneFractured):
        print("Bone is fractured")
        cv.imshow("Display window", cv.drawContours(img, c1, boneFractureContour, (0, 255, 0), 2))
        cv.waitKey(0)
    else:
        print("Bone is not fractured")

if __name__ == '__main__':
    img = cv.imread('broken1.png')
    show_contours(img)
