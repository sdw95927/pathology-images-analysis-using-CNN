import numpy             as np
import cPickle           as pickle
import matplotlib.pyplot as plt
import os

def combine(cellPatchesProbs, whitePatches):
    """ Inserts probability for all white patches at indices specified in the set whitePatches. 
    @param cellPatchesProbs {Numpy array}: Contains probability of each class for each cell patch. 
    @param whitePatches {set}: indices of white patches

    Returns the combined array. 
    """
    numPatches = len(cellPatchesProbs) + len(whitePatches)

    allProbs = []
    index = 0
    for i in range(numPatches):
        if i in whitePatches:
            allProbs.append([0,0,1]) # 100% probability that patch is white.
        else:
            allProbs.append(cellPatchesProbs[index])
            index+=1

    return allProbs

def format_heatmap(cellPatchesProbs, whitePatches, dims):
    """ Takes cell patch probabilities and the indicies of white patches to reconstruct a 2D matrix. 
    @param cellPatchesProbs {Numpy array}: Contains probability of each class for each cell patch. 
    @param whitePatches {set}: indices of white patches.
    @param dims {array}: dimensions of each resolution of a slide.

    Returns a 2D np array with predicted classes of each patch, and a 2D np array with tumor probability of each patch.
    """
    dim = dims[-1]
    ratio = np.array(dims[0])/np.array(dims[-1])
    stepSize = (300/ratio[0], 300/ratio[1])
    newDim = map(int, np.ceil((float(dim[1])/stepSize[0], float(dim[0])/stepSize[1])))

    allClasses = combine(cellPatchesProbs, whitePatches)
    
    #in case newDim doesn't match len(allClasses)
    if newDim[0] * newDim[1] != len(allClasses):
        newDim = map(int, np.round((float(dim[1])/300*(dims[0][0]/dims[-1][0]), float(dim[0])/300*(dims[0][1]/dims[-1][1]))))
    
    predLabel = np.argmax(allClasses, axis=1)
    predLabel_matrix = np.reshape(predLabel, newDim)
    
    tumorProbs = np.array(allClasses)[:,1]
    tumorProbs_matrix = np.reshape(tumorProbs, newDim)

    return predLabel_matrix, tumorProbs_matrix

def save_heatmap(predLabel, tumorProbs, outputName):
    """ Transforms 2D numpy arrays into heatmaps and saves in a files
    specified in outputName. 
    @param predLabel {Numpy array}: output of format_heatmap(); predicted class of each patch.
    @param tumorProbs {Numpy array}: 2nd output of format_heatmap(); tumor probability of each patch.
    @param outputName {string}: File name to save heatmap as.
    """
    # Transform 2D arrays into heat maps
    fig, (axis1, axis2) = plt.subplots(nrows=2)
    im = axis1.imshow(predLabel)
    axis1.xaxis.set_visible(False)
    axis1.yaxis.set_visible(False)
    axis1.set_title("Labelled regions of highest probability", fontsize=12)
    cbar = fig.colorbar(im, ax=axis1, ticks = [0, 1, 2])
    cbar.ax.set_yticklabels(["Normal", "Tumor", "White"], fontsize=8)

    im2 = axis2.imshow(tumorProbs, cmap=plt.cm.rainbow, interpolation='nearest', vmin=0, vmax=1)
    axis2.xaxis.set_visible(False)
    axis2.yaxis.set_visible(False)
    axis2.set_title("Heatmap of Tumor Probability", fontsize=12)
    cbar = fig.colorbar(im2, ax=axis2)
    cbar.ax.tick_params(labelsize=8)
    
    # Save heatmaps
    fig.set_size_inches(20,10)
    fig.savefig(outputName, transparent=True, bbox_inches='tight', pad_inches=2)
    plt.close()

def main():
    dirPath = "NLST_classesprobs2"
    heatmapPath = "NLST_classesprobs2"

    files = [fileName for fileName in os.listdir(dirPath) if fileName.endswith(".pkl")]

    for pkl in files:
        if pkl != 'NLSI0000267_2.pkl': continue
        slideId = pkl.split(".")[0]

        # Loads predicted class probabilities.
        cellPatchesProbs, whitePatches, dims = pickle.load(open(os.path.join(dirPath, pkl), 'rb'))

        # Formats predictions into a 2D matrix.
        predLabel, tumorProbs = format_heatmap(cellPatchesProbs, whitePatches, dims)

        # Plots and saves heatmap.
        outputName = os.path.join(heatmapPath, slideId) + "_2.pdf"
        heatmap = save_heatmap(predLabel, tumorProbs, outputName)        

if __name__ == '__main__':
    main()