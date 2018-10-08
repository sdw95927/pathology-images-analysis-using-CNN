import os
import cPickle as pickle
from openslide import open_slide

def get_40x_slides(dirPath, pklFile):
	slides = [os.path.join(dirPath, filename) for filename in os.listdir(dirPath) if filename.endswith('.svs')]

	validSlides = set()

	for fileName in slides:
		slide = open_slide(fileName)
		magnification = slide.properties['openslide.objective-power']
		if int(magnification) == 40: validSlides.add(fileName.split("/")[-1].split(".")[0])

	print(validSlides)
	pickle.dump(validSlides, open(pklFile, 'wb'))

	return validSlides

if __name__ == '__main__':
	get_40x_slides()