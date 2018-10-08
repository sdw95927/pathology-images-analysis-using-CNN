import os
import numpy as np
# import cPickle as pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
import pdb

def loadData():
	directoryROI, directoryNormal, directoryWhite = ("patchesROI", "patchesNormal", "patchesWhite")
	imgType = ".png"

	patchesROI = [img for img in os.listdir(directoryROI) if img.endswith(imgType)]
	patchesNormal = [img for img in os.listdir(directoryNormal) if img.endswith(imgType)]
	patchesWhite = [img for img in os.listdir(directoryWhite) if img.endswith(imgType)]

	def loadImgFromFile(dirPath, patch):
		""" Reads an image from file and returns a numpy matrix with shape (1, 3, width, height)
		@param {string} dirPath: directory that patch file is located
		@param {string} patch: file of image patch
		"""
		img = load_img(os.path.join(dirPath, patch)) # this is a PIL image
		x = img_to_array(img) # this is a Numpy array with shape (3, 300, 300)
		return x 

	# Load image matrices for each file
	X_ROI = np.array([ loadImgFromFile(directoryROI, patch) for patch in patchesROI]) # 1
	X_Normal = np.array([ loadImgFromFile(directoryNormal, patch) for patch in patchesNormal]) # 0
	X_White = np.array([ loadImgFromFile(directoryWhite, patch) for patch in patchesWhite]) # 2

	X = np.vstack((X_ROI, X_Normal, X_White))
	y = [1]*len(X_ROI) + [0]*len(X_Normal) + [2]*len(X_White)
	X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.8, random_state=42, stratify=y) # Split into train and test
	X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, train_size=0.9, random_state=42, stratify=y_train) # Split into train and validation
	print('Train shape: ', X_train.shape, np.array(y_train).shape)
	print('Val shape: ', X_val.shape, np.array(y_val).shape)
	print('Test shape: ', X_test.shape, np.array(y_test).shape)

	return X_train, X_val, X_test, y_train, y_val, y_test

# pklFilename = 'patchesData.pkl'
# X_train, X_val, X_test, y_train, y_val, y_test = loadData()
# pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), open(pklFilename, 'wb'))
# print("Pickle " + pklFilename + " saved.")

