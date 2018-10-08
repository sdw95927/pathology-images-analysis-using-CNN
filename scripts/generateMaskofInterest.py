import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import cv2
# import display as dp
import Image
from scipy import misc
import re
import time

windowIndex = 0

# Equivalent of MATLAB's imfill(BW, 'holes')
def fillHoles(bwMask):
    rWidth,cWidth  = bwMask.shape
    # Needs to be 2 pixels larger than image sent to floodFill per API (not sure why)
    mask = np.zeros((rWidth+4, cWidth+4), np.uint8)
    # Add one pixel of padding all around so that objects touching border aren't filled against border
    bwMaskCopy = np.zeros((rWidth+2, cWidth+2), np.uint8)
    bwMaskCopy[1:(rWidth+1), 1:(cWidth+1)] = bwMask
    cv2.floodFill(bwMaskCopy, mask, (0, 0), 255)
    bwMask = bwMask | (255-bwMaskCopy[1:(rWidth+1), 1:(cWidth+1)])
    return bwMask

# Equivalent of bwareaopen(BW, P)
def deleteSmallObjects(bwMask, minPixelCount):
    maskToDelete = np.ones(bwMask.shape, np.uint8)*255
    im, contours, h = cv2.findContours(bwMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minPixelCount:
            # first -1 indicates draw all contours, 255 is value to draw, -1 means draw interiors
            cv2.drawContours(maskToDelete, [contour], -1, 0, -1)
    bwMask = bwMask & maskToDelete
    return bwMask

# Used for computing circularity - see deleteNonCircular
def circularity(area, perim):
    return (perim*perim)/(4*math.pi*area)

# Remove objects from bwMask with circularity lower than circThreshold 
# Circularity calculated using circularity function above
def deleteNonCircular(bwMask, circThreshold):
    maskToDelete = np.ones(bwMask.shape, np.uint8)*255
    im, contours, h = cv2.findContours(bwMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if  area == 0 or circularity(area, cv2.arcLength(contour,True)) > circThreshold:
            print(circularity(area, cv2.arcLength(contour,True)))
            # first -1 indicates draw all contours, 255 is value to draw, -1 means draw interiors
            cv2.drawContours(maskToDelete, [contour], -1, 0, -1)
    bwMask = bwMask & maskToDelete
    return bwMask

def getMaskForSlideImage(filePath, displayProgress=False):
    slide = open_slide(filePath)
    
    # Want to capture whole image, so take first level with size less than MAX_NUM_PIXELS
    MAX_NUM_PIXELS = 5000*5000
    # levelToAnalyze = -1
    # dimsOfSelected = (-1, -1)
    levelDims = slide.level_dimensions
    # for levelIndex in range(0, len(levelDims)):
    #     if (levelDims[levelIndex][0] * levelDims[levelIndex][1]) < (MAX_NUM_PIXELS):
    #         levelToAnalyze = levelIndex
    #         dimsOfSelected = levelDims[levelIndex]
    #         break
    # if levelToAnalyze == -1:
    #     raise ValueError('No level less than ' + str(MAX_NUM_PIXELS) + ' pixels was found')
    levelToAnalyze = len(levelDims)-1
    dimsOfSelected = levelDims[-1]

    if displayProgress:
        print('Selected image of size (' + str(levelDims[levelToAnalyze][0]) + ', ' + str(levelDims[levelToAnalyze][1]) + ')')
    slideImage = slide.read_region((0, 0), levelToAnalyze, levelDims[levelToAnalyze])
    slideImageCV = np.array(slideImage)
    # Imported image is RGB, flip to get BGR, this way imshow will understand correct ordering
    slideImageCV = cv2.cvtColor(slideImageCV, cv2.COLOR_RGB2BGR)
    if displayProgress:
        #dp.displayBGRImage(slideImageCV, title="Original Image")
        plt.figure()
        plt.imshow(slideImageCV)
        
    # Perform Otsu thresholding
    threshB, maskB = cv2.threshold(slideImageCV[:,:,0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshG, maskG = cv2.threshold(slideImageCV[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshR, maskR = cv2.threshold(slideImageCV[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if displayProgress:
        #dp.displayGrayImage(maskR, title="Channel 1")
        #dp.displayGrayImage(maskG, title="Channel 2")
        #dp.displayGrayImage(maskB, title="Channel 3")
        plt.figure()
        plt.imshow(maskR)
        plt.figure()
        plt.imshow(maskG)
        plt.figure()
        plt.imshow(maskB)

    # Add the channels together
    bwMask = ((255-maskR) | (255-maskG) | (255-maskB))
    if displayProgress:
        #dp.displayGrayImage(bwMask, title="aggregate")
        plt.figure()
        plt.imshow(bwMask)

    # ADDED FOR LUNG CANCER (not in google)---------------------
    # Dilate the image
    kernel = np.ones((3,3), np.uint8)
    bwMask = cv2.dilate(bwMask, kernel, iterations=3)
    #-----------------------------------------------------------
    if displayProgress:
        #dp.displayGrayImage(bwMask, title="after removing")
        plt.figure()
        plt.imshow(bwMask)
    
    # Delete small objects
    numPixelsInImage = dimsOfSelected[0] * dimsOfSelected[1]
    minPixelCount = 0.0005 * numPixelsInImage
    bwMask = deleteSmallObjects(bwMask, minPixelCount)
    if displayProgress:
        #dp.displayGrayImage(bwMask, title="With Small Deleted")
        plt.figure()
        plt.imshow(bwMask)
        
    # Dilate the image
    kernel = np.ones((3,3), np.uint8)
    bwMask = cv2.dilate(bwMask, kernel, iterations=5)
    bwMask = cv2.erode(bwMask, kernel, iterations=3)
    bwMask = cv2.dilate(bwMask, kernel, iterations=2)
    # Fill holes
    bwMask = fillHoles(bwMask)
    if displayProgress:
        #dp.displayGrayImage(bwMask, 'After eroding operations')
        plt.figure()
        plt.imshow(bwMask)
        
    #cv2.imwrite("E:\\mask.png", bwMask)
    
    #---------------------------------------------------------
    # BEGIN OF Delete square-like objects
    # Add up the second derivative around perimieter to get curvature and delete
    # objects with very low curvature (linear objects)
    
    rWidth, cWidth = bwMask.shape
    # Add 1 pixel padding
    maskPad = np.zeros((rWidth+2, cWidth+2), np.uint8)
    maskPad[1:(rWidth+1), 1:(cWidth+1)] = bwMask
    kernel = np.ones((3,3), np.uint8)
    
    maskReduced = cv2.erode(maskPad, kernel, iterations=1)
    maskPerim = maskReduced - maskPad
    # Remove the one pixel padding
    maskPerim = maskPerim[1:(rWidth+1), 1:(cWidth+1)]
    im, contours, h = cv2.findContours(maskPerim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im, contoursFull, h = cv2.findContours(bwMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maskToDelete = np.ones(bwMask.shape, np.uint8)*255
    for index in range(0,len(contours)):
        contour = contours[index]
        xCoords = []
        yCoords = []
        for point in contour:
            xCoords.append(point[0][0])
            yCoords. append(point[0][1])
        total = np.sum(np.abs(np.diff(np.diff(xCoords)))) + np.sum(np.abs(np.diff(np.diff(yCoords))))	
        total = total/(len(xCoords))
        if total < 0.20:
            print("Deleting contour with lin value of " + str(total))
            cv2.drawContours(maskToDelete, [contoursFull[index]], -1, 0, -1)
        else:
            pass
            #print("Keeping value of " + str(total))

    #bwMask = bwMask & maskToDelete
    # END OF Delete square-like objects
    #---------------------------------------------------------
    
    # Delete artifacts such as slide labels by circularity
    if displayProgress:
        #dp.displayGrayImage(bwMask, 'After Deleting NonCircular')
        plt.figure()
        plt.imshow(bwMask)
        # overlayImage = dp.getImageWithOverlay(slideImageCV, bwMask, alpha=0.4)
        plt.figure()
        # plt.imshow(overlayImage)
        #dp.displayBGRImage(overlayImage, title="Image Overaly")
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        plt.show()
    return bwMask, slideImageCV
