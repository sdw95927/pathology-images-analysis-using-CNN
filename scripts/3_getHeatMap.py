import os
import cv2
import time
import numpy             as np
import cPickle           as pickle
import tensorflow        as tf
import matplotlib.pyplot as plt

from openslide    import open_slide, ImageSlide
from skimage.draw import polygon
from keras.models import load_model

from generateMaskofInterest import getMaskForSlideImage
from plotHeatMap            import format_heatmap, save_heatmap
from get40xSlides           import get_40x_slides

STEP_SIZE = 300
WINDOW_SIZE =(300,300)
BATCH_THRESHOLD = 300

class Slide:
    """ Object to store properties of each slide. 
    Attributes: 
        whitePatches: set of indices of white patches. Populated by get_image_arrays()
        set_regionDict: dictionary of region boundaries. Populated by get_boundaries_dict()
    """
    def __init__(self, slideFile):
        image, dims, ratio = self.load_slide(slideFile)
        self.id = slideFile.split("/")[-1].split(".")[0]
        self.file = slideFile
        self.image = image
        self.dims = dims
        self.ratio = ratio
        self.whitePatches = set()
        self.regionDict = self.store_region_boundaries()
        self.startTime = time.time()
        self.estTime = self.get_predicted_time()

    def add_white_patch(self, index):
        self.whitePatches.add(index)

    def store_region_boundaries(self):
        """ Return a dictionary specifying the min x and max x boundaries that contain cells for each row.
        """ 
        # Generate mask that distinguishes cells from white space. 
        bwMask, _ = getMaskForSlideImage(self.file)
        
        # Get the indices for mask border. 
        _, contours, _ = cv2.findContours(bwMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Get region boundaries for each row. 
        regionDict = self.get_boundaries_dict(contours)
        return regionDict

    def load_slide(self, fileName):
        """ Returns an openslide image, an array of dimensions for each resolution, 
        and ratio between the highest and lowest resolution.
        """
        image = open_slide(fileName)
        dims = image.level_dimensions
        ratio = np.array(image.level_dimensions[0])/np.array(image.level_dimensions[-1])

        return image, dims, ratio

    def get_boundaries_dict(self, contours):
        """ Stores the (min x, max x) value for each region in each horizontal slice. 
        Returns a dictionary of the following structure: 
            key: index of row
            value: list of (min x, max x) tuples for each region
        _____________
        | xx x      |       For the above example, get_boundaries_dict() will return
        |  xxx    xx|       0: [(1,4)]
        |___________|       1: [(2,4), (9,10)]

        """
        dims = self.dims[-1]
        ratio = self.ratio
        regionDict = {}
        for region in contours:
            pts = region[:,0,:]

            for i, yMin in enumerate(np.arange(0, dims[1], STEP_SIZE/ratio[1])): # For each row
                yMax = yMin+(STEP_SIZE/ratio[1])
                allVals = pts[(yMin <= pts[:,1]) & (pts[:,1] < yMax)] # Isolate the points in each row.
                if len(allVals) == 0: continue # If there are no points in the row, continue to the next row. 
                xBounds = (min(allVals[:,0]), max(allVals[:,0])) # Else, then isolate the max and min points of this row.

                # Add the (min,max) tuple to the dictionary.
                if i in regionDict:
                    regionDict[i].append(xBounds)
                else:
                    regionDict[i] = [xBounds]

        return regionDict

    def get_predicted_time(self):
        """ Calculates an estimated time to process the slide. """
        estPatches = 0
        for row in self.regionDict:
            for minX, maxX in self.regionDict[row]:
                estPatches += (maxX-minX)/(STEP_SIZE/self.ratio[0])

        estTime = (estPatches/300.)*15 # Assume 10 sec per 300 patches (3 sec to generate, 7 sec to predict)
        return estTime/60. # Return minutes

    def print_status(self):
        """ Prints the elapsed time in minutes and estimated time to finish. """
        currentTime = time.time()
        elapsedTime = (currentTime - self.startTime)/60.
        print('Elapsed time/Estimated time: {:.2f}/{:.2f} mins'.format(elapsedTime, self.estTime))

def save_original_image(slide, heatmapPath):
    """ Save original image to a lower resolution png file
    """
    dims = slide.dims
    img = slide.image.read_region((0,0), len(dims)-1, dims[-1])
    img.save(os.path.join(heatmapPath, slide.id) + ".png")

def get_patch_predictions(slide, model):
    """ Yields patch predictions and mutates the set slide.whitePatches.

    For each batch of patches (specified by BATCH_THRESHOLD), a check is done to determine 
    whether the patch contains cells or just white space. 

    If it contains cells, the patches
    and their image matrices will be generated, and then the specified model
    will predict the probabilities for each classes. This numpy array of probabilities 
    is yielded after completion and later appended to each other to form a full array 
    of probabilities. 

    If the patch is just white space, a set, slide.whitePatches is mutated with the indices of
    all the white patches to store their location for future regeneration of a heatmap.

    """
    dims = slide.dims
    ratio = slide.ratio
    imgArr = np.zeros((BATCH_THRESHOLD, WINDOW_SIZE[0], WINDOW_SIZE[1], 3))
    totalIndex = -1 # Total count of all patches
    imgIndex = 0 # Count of only image patches
    
    for iy, y in enumerate(np.arange(0, dims[-1][1], STEP_SIZE/ratio[1])):
        for ix, x in enumerate(np.arange(0, dims[-1][0], STEP_SIZE/ratio[0])):
            totalIndex += 1

            # If there are no cells in this row, add the index to the set of white patches. 
            if iy not in slide.regionDict:
                slide.add_white_patch(totalIndex)
                continue

            # Else, check if the x values fall within the region boundaries.
            regions = slide.regionDict[iy]
            cont = np.any([(region[0] <= x < region[1]) for region in regions])
            if not cont: # If x doesn't fall within the boundaries, add index to white patches.
                slide.add_white_patch(totalIndex)
                continue
            
            # Extract the image patch, convert it to a numpy matrix, and normalize.
            image = slide.image.read_region((x*ratio[0] ,y*ratio[1]), 0, WINDOW_SIZE )
            imgMat = np.array(image.convert(mode="RGB")) / 255. 
            imgArr[imgIndex] = imgMat
            imgIndex += 1   

            # For every batch (specified by BATCH_THRESHOLD),
            if imgIndex%BATCH_THRESHOLD == 0:
                # Predict the classes for each image patch
                classes = model.predict(imgArr, batch_size=32, verbose=1)
                yield classes

                imgIndex = 0
                imgArr = np.zeros((BATCH_THRESHOLD, WINDOW_SIZE[0], WINDOW_SIZE[1], 3))

            if totalIndex%2500==0: slide.print_status()

    # Trim zeros and predict final batch if any.
    imgArr_trimmed = imgArr[:imgIndex] 
    if len(imgArr_trimmed):
        classes = model.predict(imgArr_trimmed, batch_size=32, verbose=1)
        yield classes

def main():
    dirPath = "lung_slides_NLST"
    pklPath = "pickleFiles"
    heatmapPath = "heatmaps"
    modelFile = "inception_07062017.h5"

    slides = [os.path.join(dirPath, fileName) for fileName in os.listdir(dirPath) if fileName.endswith('.svs')]
    # completedSlides = {fileName.split(".")[0] for fileName in os.listdir(pklPath) if fileName.endswith('.pkl')}
    completedSlides = {fileName.split(".")[0] for fileName in os.listdir(heatmapPath) if fileName.endswith('.pdf')}
    validSlides = get_40x_slides(dirPath, '40xSlides.pkl') # Set of all 40x slides
    numRemaining = len(validSlides - completedSlides)
    model = load_model(modelFile)
    print(numRemaining) 
    for slideNum, slideFile in enumerate(slides):
        print(slideFile)

        # Avoid reprocessing slides. 
        if slideFile.split("/")[-1].split(".")[0] in completedSlides: continue

        # Only process slides with 40x magnification.
        if slideFile.split("/")[-1].split(".")[0] not in validSlides: continue

        # slideFile = os.path.join(dirPath, 'NLSI0000267.svs')
        slide = Slide(slideFile)

        print('Starting slide {}, file {}, dims {}, est.time {:.2f}'.format(slideNum+1, slide.id, slide.dims[0], slide.estTime))
        start = time.time()

        # Append the returns from all the batches together.
        def append_all(): return lambda x, y: np.append(x,y,axis=0)
        patchPreds = reduce(append_all(), get_patch_predictions(slide, model))
        print('Final image predictions shape: {}'.format(patchPreds.shape))

        # Save classes probabilities in pickle. 
        filename = os.path.join(pklPath, slide.id) + ".pkl"
        pickle.dump((patchPreds, slide.whitePatches, slide.dims), open(filename, "wb"))

        # Save original image and heatmap.
        save_original_image(slide, heatmapPath)

        predLabel, tumorProbs = format_heatmap(patchPreds, slide.whitePatches, slide.dims)
        outputName = os.path.join(heatmapPath, slide.id) + ".pdf"
        save_heatmap(predLabel, tumorProbs, outputName)

        print('Saved heatmap {}'.format(outputName))
        print('Total time for slide {}: {:.2f} mins.'.format(slide.id, (time.time()-start)/60.))
        numRemaining -= 1
        print('Number of slides remaining: {}'.format(numRemaining))
        
if __name__ == "__main__":
    main()

