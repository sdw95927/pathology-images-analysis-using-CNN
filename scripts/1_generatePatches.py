## Python 2.7.12
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from skimage.draw import polygon
from openslide import open_slide, ImageSlide
import matplotlib.pyplot as plt
import random 

def parseXML(xmlFile):
    """
    Parse XML File and returns an object containing all the vertices 
    Verticies: (dict)
         'ROI': (list) of dicts, each with 'X' and 'Y' key 
                [{ 'X': [1,2,3], 
                   'Y': [1,2,3]  }]
         'Normal': (list) of dicts, each with 'X' and 'Y' key 
                [{ 'X': [4,5,6], 
                   'Y': [4,5,6]  }]
    """
    
    tree = ET.parse(xmlFile) # Convert XML file into tree representation
    root = tree.getroot()

    regions = root.iter('Region') # Extract all Regions
    vertices = {'ROI': [], 'Normal': []} # Store all vertices in a dictionary

    for region in regions: 
        label = region.get('Text') # label either as 'ROI' or 'normal'
        if label not in {'ROI', 'Normal'}: print('Unidentified labelled region: ', label)
        vertices[label].append({'X':[], 'Y':[]})
        
        for vertex in region.iter('Vertex'): 
            X = float(vertex.get('X'))
            Y = float(vertex.get('Y'))
                
            vertices[label][-1]['X'].append(X)
            vertices[label][-1]['Y'].append(Y)

    return vertices

def calculateRatio(levelDims):
    """ Calculates the ratio between the highest resolution image and lowest resolution image.
    Returns the ratio as a tuple (Xratio, Yratio). 
    """
    highestReso = np.asarray(levelDims[0])
    lowestReso = np.asarray(levelDims[-1])
    Xratio, Yratio = highestReso/lowestReso
    return (Xratio, Yratio)

def createMask(levelDims, vertices):
    """
    Input: levelDims (nested list): dimensions of each layer of the slide.
           vertices (dict object as describe above)
    Output: (tuple) ROI mask, Normal mask (tuple) 
            two numpy nd arrays of 0/1, where 1 indicates inside the region
            and 0 is outside the region
    """
    # Down scale the XML region to create a low reso image mask, and then 
    # rescale the image to retain reso of image mask to save memory and time 
    Xratio, Yratio = calculateRatio(levelDims)

    nRows, nCols = levelDims[-1]
    maskROI = np.zeros((nRows, nCols), dtype=np.uint8)
    maskNormal = np.zeros((nRows, nCols), dtype=np.uint8)

    for region in ['ROI', 'Normal']:
        for i in range(len(vertices[region])):
            lowX = np.array(vertices[region][i]['X'])/Xratio
            lowY = np.array(vertices[region][i]['Y'])/Yratio
            rr, cc = polygon(lowX, lowY, (nRows, nCols))
            if region == 'ROI':
                maskROI[rr, cc] = 1
            elif region == 'Normal':
                maskNormal[rr, cc] = 1    

    return maskROI, maskNormal

def getMask(xmlFile, svsFile):
    """ Parses XML File to get mask vertices and returns matrix masks 
    where 1 indicates the pixel is inside the mask, and 0 indicates outside the mask.

    @param: {string} xmlFile: name of xml file that contains annotation vertices outlining the mask. 
                    (Annotations must be 'ROI' or 'Normal')
    @param: {string} svsFile: name of svs file that contains the slide image.
    Returns: slide - openslide slide Object 
             maskROI - matrix mask of ROI 
             maskNormal - matrix mask of Normal region 
    """
    vertices = parseXML(xmlFile) # Parse XML to get vertices of mask 

    slide = open_slide(svsFile)
    levelDims = slide.level_dimensions
    maskROI, maskNormal = createMask(levelDims, vertices)

    return slide, maskROI, maskNormal

def plotMask(mask):
    fig, ax1 = plt.subplots(nrows=1, figsize=(6,10))
    ax1.imshow(mask)
    plt.show()

def chooseRandPixel(mask):
    """ Returns [x,y] numpy array of random pixel.
    @param {numpy matrix} mask from which to choose random pixel.
    """
    array = np.transpose(np.nonzero(mask)) # Get the indices of nonzero elements of mask.
    index = random.randint(0,len(array)-1) # Select a random index
    return array[index]

def plotImage(image):
    plt.imshow(image)
    plt.show()
    
def checkWhiteSlide(image):
    im = np.array(image.convert(mode='RGB'))
    pixels = np.ravel(im)
    mean = np.mean(pixels)
    return mean >= 220

def getPatches(slide, mask, numPatches=0, dims=(0,0), dirPath='', slideNum='', plot=False, plotMask=False):
    """ Generates and saves 'numPatches' patches with dimension 'dims' from image 'slide' contained within 'mask'.
    @param {Openslide Slide obj} slide: image object
    @param {numpy matrix} mask: where 0 is outside region of interest and 1 indicates within 
    @param {int} numPatches
    @param {tuple} dims: (w,h) dimensions of patches
    @param {string} dirPath: directory in which to save patches
    @param {string} slideNum: slide number 
    Saves patches in directory specified by dirPath as [slideNum]_[patchNum]_[Xpixel]x[Ypixel].png
    """ 
    w,h = dims 
    levelDims = slide.level_dimensions
    Xratio, Yratio = calculateRatio(levelDims)

    i = 0
    while i < numPatches:
        firstLoop = True # Boolean to ensure while loop runs at least once. 

        while firstLoop or not mask[rr,cc].all(): # True if it is the first loop or if all pixels are in the mask 
            firstLoop = False
            x, y = chooseRandPixel(mask) # Get random top left pixel of patch. 
            xVertices = np.array([x, x+(w/Xratio), x+(w/Xratio), x, x])
            yVertices = np.array([y, y, y-(h/Yratio), y-(h/Yratio), y])
            rr, cc = polygon(xVertices, yVertices)

        image = slide.read_region((x*Xratio, y*Yratio), 0, (w,h))
        
        isWhite = checkWhiteSlide(image)
        newPath = 'patchesWhite' if isWhite else dirPath
        if not isWhite: i += 1

        slideName = '_'.join([slideNum, 'x'.join([str(x*Xratio),str(y*Yratio)])])
        image.save(os.path.join(newPath, slideName+".png"))

        if plot: 
            plotImage(image)
        if plotMask: mask[rr,cc] = 0

    if plotMask:
        plotImage(mask)

def main():
    dirName = 'lung_slides_NLST' 
    completedSlides = 'log.txt'
    numPatches = 50

    f = open(completedSlides, 'r+')
    slides = [line.strip() for line in f]

    for slideNum in slides:
        try: 
            xmlFile = slideNum+'.xml'
            svsFile = slideNum+'.svs'

            xmlFile = os.path.join(dirName, xmlFile)
            svsFile = os.path.join(dirName, svsFile) 

            slide, maskROI, maskNormal = getMask(xmlFile, svsFile)
            
            getPatches(slide, maskROI, numPatches=numPatches, dims=(300,300), dirPath="patchesROI/", slideNum=slideNum) # Get ROI Patches
            if np.any(maskNormal): # If slide has normal mask
                getPatches(slide, maskNormal, numPatches=numPatches, dims=(300,300), dirPath="patchesNormal/", slideNum=slideNum) # Get Normal Patches
            
            print('Done with ' + slideNum)
            
        except:
            print('Error with ' + slideNum)

if __name__ == "__main__":
    main()

