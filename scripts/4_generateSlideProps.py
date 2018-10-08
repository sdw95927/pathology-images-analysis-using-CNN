import sys
import os
import time
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops, label
from plotHeatMap import format_heatmap

# 8 connectivity, extent, eccentricity, solidity, euler number

PROPERTIES = ['slide_id',
              'time',
              'status',
              'age',
              'gender', # 1 is male, 2 is female
              'tobacco', # 1 if patient has a history of tobacco usage, 0 if not.
              'stage',
              'grade',
                
              'tumor_percent', # Number of predicted tumor pixels / all predicted tumor and normal pixels.
              'tumor_probability', # Average tumor probability of pixels predicted as tumor or normal.

            ##############################################################
            ### Properties from image that underwent image processing. ###
            ##############################################################

              'num_regions', # Number of regions in the slide.
              'total_area', # Area sum of all the regions.
              'total_perimeter', # Perimeter sum of all the regions.
              'total_convex_area', # Sum of convex area for all the regions.
              'total_filled_area', # Sum of filled area for all the regions.
              'total_euler_num', # Sum of euler number of all regions.
              'total_mj_axis', # Sum of major axis length of all regions.
              'total_mi_axis', # Sum of minor axis length of all regions.
              'total_pa_ratio', # Perimeter/Area ratio of all the regions.

              'main_area', # Area of largest region.
              'main_convex_area', # Convex area of largest region.
              'main_eccentricity', # Ratio of the focal distance over the major axis length. When it is 0, the ellipse becomes a circle.
              'main_equiv_diameter', # Equivalent diameter of largest region. The diameter of a circle with the same area as the region.
              'main_euler_num', # Euler number of largest region. Computed as number of objects (= 1) subtracted by number of holes.
              'main_extent', # Extent of largest region. Ratio of pixels in the region to pixels in the total bounding box.
              'main_filled_area', # Filled area for largest region.
              'main_mj_axis', # Major axis length for largest region.
              'main_mi_axis', # Minor axis length for largest region.
              'main_orientation', #  Angle between the X-axis and the major axis of the ellipse that has the same second-moments as the region.
              'main_perimeter', # Perimeter of largest region.
              'main_solidity', # Solidity of largest region. Ratio of pixels in the region to pixels of the convex hull image.
              'main_percentage', # Area of largest region / area of all regions. 
              'main_tumor_percent', # Percentage of pixels in the largest region predicted as tumor / all pixels in the region.
              'main_normal_percent', # Percentage of pixels in the largest region predicted as normal / all pixels in the region. 
              'main_tumor_ratio', # Ratio of tumor pixels / normal pixels in the largest region. 
              'main_tumor_probability', # Average tumor probability of the largest region.
              'main_pa_ratio', # Perimeter/Area ratio for main region.

              'distance_avg', # Average distance between other regions' centroid and the largest region's centroid.
              'distance_max', # Max distance between a region's centroid and largest region's centroid.
              'distance_min', # Min distance between a region's centroid and largest region's centroid.
              'distance_std', # Standard deviation of distances between other regions' centroids and the largest region's centroid.

            ##############################################################
            ### Properties with image that did not undergo processing. ###
            ##############################################################

              'num_regions_2', # Number of regions in the slide.
              'total_area_2', # Area sum of all the regions.
              'total_perimeter_2', # Perimeter sum of all the regions.
              'total_convex_area_2', # Sum of convex area for all the regions.
              'total_filled_area_2', # Sum of filled area for all the regions.
              'total_euler_num_2', # Sum of euler number of all regions.
              'total_mj_axis_2', # Sum of major axis length of all regions.
              'total_mi_axis_2', # Sum of minor axis length of all regions.
              'total_pa_ratio_2', # Perimeter/Area ratio of all the regions.

              'main_area_2', # Area of largest region.
              'main_convex_area_2', # Convex area of largest region.
              'main_eccentricity_2', # Ratio of the focal distance over the major axis length. When it is _20, the ellipse becomes a circle.
              'main_equiv_diameter_2', # Equivalent diameter of largest region. The diameter of a circle with the same area as the region.
              'main_euler_num_2', # Euler number of largest region. Computed as number of objects (= 1) subtracted by number of holes.
              'main_extent_2', # Extent of largest region. Ratio of pixels in the region to pixels in the total bounding box.
              'main_filled_area_2', # Filled area for largest region.
              'main_mj_axis_2', # Major axis length for largest region.
              'main_mi_axis_2', # Minor axis length for largest region.
              'main_orientation_2', #  Angle between the X-axis and the major axis of the ellipse that has the same second-moments as the region.
              'main_perimeter_2', # Perimeter of largest region.
              'main_solidity_2', # Solidity of largest region. Ratio of pixels in the region to pixels of the convex hull image.
              'main_percentage_2', # Area of largest region / area of all regions. 
              'main_tumor_percent_2', # Percentage of pixels in the largest region predicted as tumor / all pixels in the region.
              'main_normal_percent_2', # Percentage of pixels in the largest region predicted as normal / all pixels in the region. 
              'main_tumor_ratio_2', # Ratio of tumor pixels / normal pixels in the largest region. 
              'main_tumor_probability_2', # Average tumor probability of the largest region.
              'main_pa_ratio_2', # Perimeter/Area ratio for main region.

              'distance_avg_2', # Average distance between other regions' centroid and the largest region's centroid.
              'distance_max_2', # Max distance between a region's centroid and largest region's centroid.
              'distance_min_2', # Min distance between a region's centroid and largest region's centroid.
              'distance_std_2', # Standard deviation of distances between other regions' centroids and the largest region's centroid.

              ]

class Slide():
    def __init__(self, pklFile):
        # Get predicted classes and tumor probability of each pixel.
        cellPatchesProbs, whitePatches, dims = pickle.load(open(pklFile, 'rb'))
        classes, tumorProbs = format_heatmap(cellPatchesProbs, whitePatches, dims)

        # Process the image, label each tumor region, and get region properties.
        
        regionProps_morph = self.region_props_morph(classes)
        regionProps_connect = self.region_props_connect(classes)

        self.slide_id = pklFile.split("/")[-1].split(".")[0]
        self.time, self.status = self.get_survival()
        self.age, self.gender, self.tobacco, self.stage, self.grade = self.extract_clinical_features()
        self.predicted_classes = classes
        self.tumor_probs = tumorProbs

        self.tumor_percent = self.get_tumor_percent()
        self.tumor_probability, self.tumor_std = self.get_tumor_probability()

        ##################################################################
        ### Get properties from image that underwent image processing. ###
        ##################################################################

        properties_all = regionProps_morph
        properties_main = self.get_main_properties(regionProps_morph)
        main_classes = self.get_main_classes(properties_main)
        main_probs = self.get_main_probs(properties_main)

        self.num_regions = len(regionProps_morph)
        self.total_area = self.get_total_prop(properties_all, 'area')
        self.total_perimeter = self.get_total_prop(properties_all, 'perimeter')
        self.total_convex_area = self.get_total_prop(properties_all, 'convex_area')
        self.total_filled_area = self.get_total_prop(properties_all, 'filled_area')
        self.total_euler_num = self.get_total_prop(properties_all, 'euler_number')
        self.total_mj_axis = self.get_total_prop(properties_all, 'major_axis_length')
        self.total_mi_axis = self.get_total_prop(properties_all, 'minor_axis_length')
        self.total_pa_ratio = self.get_pa_ratio(self.total_perimeter, self.total_area)

        self.main_area = self.get_main_prop(properties_main, 'area')
        self.main_convex_area = self.get_main_prop(properties_main, 'convex_area')
        self.main_eccentricity = self.get_main_prop(properties_main, 'eccentricity')
        self.main_equiv_diameter = self.get_main_prop(properties_main, 'equivalent_diameter')
        self.main_extent = self.get_main_prop(properties_main, 'extent')
        self.main_filled_area = self.get_main_prop(properties_main, 'filled_area')
        self.main_euler_num = self.get_main_prop(properties_main, 'euler_number')
        self.main_mj_axis = self.get_main_prop(properties_main, 'major_axis_length')
        self.main_mi_axis = self.get_main_prop(properties_main, 'minor_axis_length')
        self.main_orientation = self.get_main_prop(properties_main, 'orientation')
        self.main_perimeter = self.get_main_prop(properties_main, 'perimeter')
        self.main_solidity = self.get_main_prop(properties_main, 'solidity')
        self.main_percentage = self.get_main_percentage(self.main_area, self.total_area)
        self.main_tumor_percent = self.get_main_tumor_percent(main_classes)
        self.main_normal_percent = self.get_main_normal_percent(main_classes)
        self.main_tumor_ratio = self.get_main_tumor_ratio(self.main_tumor_percent, self.main_normal_percent)
        self.main_tumor_probability, self.main_tumor_std = self.get_main_tumor_probability(main_probs, main_classes)
        self.main_pa_ratio = self.get_pa_ratio(self.main_perimeter, self.main_area)

        main_centroid = self.get_main_prop(properties_main, 'centroid')
        self.distances = self.get_distances(properties_all, main_centroid)
        self.distance_avg = np.mean(self.distances)
        self.distance_std = np.std(self.distances)
        self.distance_max = max(self.distances)
        self.distance_min = min(self.distances)

        #####################################################################
        ### Repeat properties with image that did not undergo processing. ###
        #####################################################################

        properties_all_2 = regionProps_connect
        properties_main_2 = self.get_main_properties(regionProps_connect)
        main_classes_2 = self.get_main_classes(properties_main_2)
        main_probs_2 = self.get_main_probs(properties_main_2)

        self.num_regions_2 = len(regionProps_connect)
        self.total_area_2 = self.get_total_prop(properties_all_2, 'area')
        self.total_perimeter_2 = self.get_total_prop(properties_all_2, 'perimeter')
        self.total_convex_area_2 = self.get_total_prop(properties_all_2, 'convex_area')
        self.total_filled_area_2 = self.get_total_prop(properties_all_2, 'filled_area')
        self.total_euler_num_2 = self.get_total_prop(properties_all_2, 'euler_number')
        self.total_mj_axis_2 = self.get_total_prop(properties_all_2, 'major_axis_length')
        self.total_mi_axis_2 = self.get_total_prop(properties_all_2, 'minor_axis_length')
        self.total_pa_ratio_2 = self.get_pa_ratio(self.total_perimeter_2, self.total_area_2)

        self.main_area_2 = self.get_main_prop(properties_main_2, 'area')
        self.main_convex_area_2 = self.get_main_prop(properties_main_2, 'convex_area')
        self.main_eccentricity_2 = self.get_main_prop(properties_main_2, 'eccentricity')
        self.main_equiv_diameter_2 = self.get_main_prop(properties_main_2, 'equivalent_diameter')
        self.main_extent_2 = self.get_main_prop(properties_main_2, 'extent')
        self.main_filled_area_2 = self.get_main_prop(properties_main_2, 'filled_area')
        self.main_euler_num_2 = self.get_main_prop(properties_main_2, 'euler_number')
        self.main_mj_axis_2 = self.get_main_prop(properties_main_2, 'major_axis_length')
        self.main_mi_axis_2 = self.get_main_prop(properties_main_2, 'minor_axis_length')
        self.main_orientation_2 = self.get_main_prop(properties_main_2, 'orientation')
        self.main_perimeter_2 = self.get_main_prop(properties_main_2, 'perimeter')
        self.main_solidity_2 = self.get_main_prop(properties_main_2, 'solidity')
        self.main_percentage_2 = self.get_main_percentage(self.main_area_2, self.total_area_2)
        self.main_tumor_percent_2 = self.get_main_tumor_percent(main_classes_2)
        self.main_normal_percent_2 = self.get_main_normal_percent(main_classes_2)
        self.main_tumor_ratio_2 = self.get_main_tumor_ratio(self.main_tumor_percent_2, self.main_normal_percent_2)
        self.main_tumor_probability_2, self.main_tumor_std_2 = self.get_main_tumor_probability(main_probs_2, main_classes_2)
        self.main_pa_ratio_2 = self.get_pa_ratio(self.main_perimeter_2, self.main_area_2)

        main_centroid_2 = self.get_main_prop(properties_main_2, 'centroid')
        self.distances_2 = self.get_distances(properties_all_2, main_centroid_2)
        self.distance_avg_2 = np.mean(self.distances_2)
        self.distance_std_2 = np.std(self.distances_2)
        self.distance_max_2 = max(self.distances_2)
        self.distance_min_2 = min(self.distances_2)

    def region_props_morph(self, classes):  
        """ Process image with binary opening and closing, label regions, and return region properties """ 
        # Convert from Normal/Tumor/White label to Other/Tumor label. 
        binary_classes = map(lambda x: x==1, classes) 

        closing = binary_closing(binary_classes, structure=np.ones((3,3)))
        opening = binary_opening(closing, structure=np.ones((2,2)))
        second_closing = binary_closing(opening, structure=np.ones((3,3)))
        second_opening = binary_opening(second_closing, structure=np.ones((3,3)))
        third_closing = binary_closing(second_opening, structure=np.ones((4,4)))
        final = binary_opening(third_closing, structure=np.ones((4,4)))
        
        # Label regions.
        label_morph = label(final, background=0)
        # Get properties for each region.
        return regionprops(label_morph)

    def region_props_connect(self, classes):
        """ 
        This method doesn't process images, but simply labels regions with looser criteria. 
        Pixels are connected if they are touching or orthogonal. Thus, this method results in more regions. 
        Returns region properties for each region. 
        """
        # Convert from Normal/Tumor/White label to Other/Tumor label. 
        binary_classes = np.reshape(map(lambda x: x==1, classes), classes.shape)

        # Label image; Pixels are connected if they are touching or orthogonal.
        label_connect = label(binary_classes, background=0) 

        # Discard all regions with less than 4 pixels.
        region_props = [region for region in regionprops(label_connect) if region['area'] >= 4.] 
        return region_props

    def get_main_properties(self, regionProps):
        main_index = np.argmax([region['area'] for region in regionProps])
        return regionProps[main_index]

    def get_main_classes(self, properties_main):
        minRow, minCol, maxRow, maxCol = properties_main['bbox']
        return self.predicted_classes[minRow:maxRow, minCol:maxCol]

    def get_main_probs(self, properties_main):
        minRow, minCol, maxRow, maxCol = properties_main['bbox']
        return self.tumor_probs[minRow:maxRow, minCol:maxCol]

    def get_distances(self, properties_all, main_centroid):
        centroids = [np.array(region['centroid']) for region in properties_all]
        main_centroid = np.array(main_centroid)
        distances = [np.linalg.norm(centroid - main_centroid) for centroid in centroids \
                    if (centroid[0]!=main_centroid[0]) & (centroid[1]!=main_centroid[1])]
        return distances

    def get_tumor_percent(self):
        allPredictions = self.predicted_classes.flatten()
        tumorCount = [pixel for pixel in allPredictions if pixel != 2]
        return sum(tumorCount)/float(len(tumorCount))

    def get_tumor_probability(self):
        allProbs = [prob for index, prob in enumerate(self.tumor_probs.flatten()) \
                        if self.predicted_classes.flatten()[index] != 2]
        return np.mean(allProbs), np.std(allProbs)

    def get_total_prop(self, properties_all, prop):
        return sum([region[prop] for region in properties_all])

    def get_pa_ratio(self, perimeter, area):
        return perimeter/float(area)

    def get_main_prop(self, properties_main, prop):
        return properties_main[prop]

    def get_main_percentage(self, main_area, total_area):
        return main_area/float(total_area)

    def get_main_tumor_percent(self, main_classes):
        tumorCount = len([1 for pixel in main_classes.flatten() if pixel==1])
        allPixels = len(main_classes.flatten())
        return tumorCount / float(allPixels)

    def get_main_normal_percent(self, main_classes):
        normalCount = len([1 for pixel in main_classes.flatten() if pixel==0])
        allPixels = len(main_classes.flatten())
        return normalCount / float(allPixels)

    def get_main_tumor_ratio(self, tumor_percent, normal_percent):
        return tumor_percent / normal_percent

    def get_main_tumor_probability(self, main_probs, main_classes):
        allProbs = [prob for i, prob in enumerate(main_probs.flatten()) \
                    if main_classes.flatten()[i] != 2]
        return np.mean(allProbs), np.std(allProbs)

    def get_survival(self):
        # mapping_dict and survival are defined globably in main()
        patientId = mapping_dict[str(self.slide_id)]['aaCode']
        surv = survival[survival['ID']==patientId]
        surv.reset_index(drop=True, inplace=True)
        return surv.get_value(0,'Overall_Survival'), surv.get_value(0,'vital.status')

    def extract_clinical_features(self):
        # mapping dict and clinical are defined globably in main()
        patientId = mapping_dict[str(self.slide_id)]['aaCode']
        clin = clinical[clinical['ID']==patientId]
        clin.reset_index(drop=True, inplace=True)
        return clin.get_value(0,'Age'), clin.get_value(0,'Gender'), clin.get_value(0,'Tobacco.history'), clin.get_value(0,'Stage'), clin.get_value(0,'Grade')



def main():
    dirPath = 'pickleFiles'
    propertiesFilename = "region_properties_slides_all.csv"
    validSlides = pickle.load(open('40xSlides.pkl', 'rb')) # Set of all 40x slides
    pklFiles = [os.path.join(dirPath, pkl) for pkl in os.listdir(dirPath) if pkl.split("/")[-1].split(".")[0] in validSlides]
    
    # Get survival and clinical data. 
    dirPath = "~/matlab/image_patch_extraction/NLST"
    survivalFile = os.path.join(dirPath, "NLST_Survival.csv")
    clinicalFile = os.path.join(dirPath, "nlst_clinical.csv")
    mappingFile = os.path.join(dirPath, "AACode-Filename-Mapping-curated.csv")
    global survival
    survival = pd.read_csv(survivalFile) 
    global clinical
    clinical = pd.read_csv(clinicalFile)
    mapping = pd.read_csv(mappingFile)
    mapping['aaCode'] = mapping['aaCode'].str[:-5]
    mapping['filename'] = mapping['filename'].str[:-4]
    new_mapping = mapping.set_index('filename')
    global mapping_dict
    mapping_dict = new_mapping.to_dict("index")

    # Create main dataframe.
    maindf = pd.DataFrame(columns=PROPERTIES)

    for i, pklFile in enumerate(pklFiles):
        try:
            start = time.time()
            slide = Slide(pklFile)
        except:
            print('{} can\'t be processed'.format(pklFile))
            continue

        values = [slide.__dict__[prop] for prop in PROPERTIES]
        maindf.loc[i] = values
        print('{} Num regions: {}, Total time: {}'.format(pklFile, slide.num_regions, time.time()-start))

    maindf.to_csv(propertiesFilename, index=False)


if __name__ == '__main__':
    main()
