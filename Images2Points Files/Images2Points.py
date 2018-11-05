# Written by Steve S. Lee
# 10/14/2018
# stevesl@umich.edu

# Spec:
# Takes two images as an input and creates csv file that has landmark/regisreted points between the points.
# Presumably it will have 4 columns with first two being coordinates of a point in image 1 and next two the
# coordinates of the corresponding point in second image.

import cv2
import numpy
import csv
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
#from skimage.transform import AffineTransform
class Images2Points(object):

	# Initializer.
	def __init__(self):
		print("Images2Points is created.")

	# Get matching points.
	###
	# Use: Input two images and the function returns numpy array of points for image 1 and 2 that match.
	# If outputcsvFileName is specified, the function also creates a csv file of the points for image 1 and 2.
	def getPointsFromImages(self, firstImage, secondImage, outputcsvFileName=None, detectorType=None, crossCheck=None, normType=None, createImageWithPtsAndLines=None, numOfPtsAndLinesToShow=None):
		
		# Normalizing Images. Leave this commented for now.
		# normalizedImg = numpy.zeros((len(firstImage), len(firstImage[0])))
		# firstImage = cv2.normalize(firstImage,  normalizedImg, 0, 100, cv2.NORM_MINMAX)
		# normalizedImg = numpy.zeros((len(secondImage), len(secondImage[0])))
		# secondImage = cv2.normalize(secondImage,  normalizedImg, 0, 100, cv2.NORM_MINMAX)

		# By default use ORB.
		detector = cv2.ORB_create()
		if (detectorType is not None):
			if (detectorType == "SIFT"):
				detector = cv2.xfeatures2d.SIFT_create()
			elif (detectorType == "SURF"):
				detector = cv2.xfeatures2d.SURF_create()

		# Finding keypoints using SURF.
		points1, firstImageFeatures = detector.detectAndCompute(firstImage, None)
		points2, secondImageFeatures = detector.detectAndCompute(secondImage, None)

		# Creating BFMatcher object for feature matching.
		# Use NORM_HAMMING by default.
		bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
		if (normType is not None):
			if (normType == "NORM_L1"):
				if (crossCheck):
					bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
				else:
					bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
			elif (normType == "NORM_L2"):
				if (crossCheck):
					bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
				else:
					bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)

		# Getting the index pairs that match.
		tempindexPairs = bf.match(secondImageFeatures, firstImageFeatures)
		tempindexPairs = sorted(tempindexPairs, key = lambda x:x.distance)
		indexPairs = tempindexPairs

		# indexPairs[i].queryIdx gives index of points that were matched.
		matchedPointsOnImage2 = numpy.asarray([points2[indexPairs[i].queryIdx] for i in range(len(indexPairs))])
		numpyArrayMatchedPointsOnImage2 = numpy.asarray([matchedPointsOnImage2[i].pt for i in range(len(matchedPointsOnImage2))])
		# matchedPoints have the KeyPoint objects, which an be accesesd by index and .pt.
		# print(matchedPoints[0].pt)
		matchedPointsOnImage1 = numpy.asarray([points1[indexPairs[i].trainIdx] for i in range(len(indexPairs))])
		numpyArrayMatchedPointsOnImage1 = numpy.asarray([matchedPointsOnImage1[i].pt for i in range(len(matchedPointsOnImage1))])

		# If csv file name is specified, export the points as csv file format.
		if (outputcsvFileName is not None):
			self.exportPointsAsCSV(csvFileName=outputcsvFileName, pointsFromImage1=numpyArrayMatchedPointsOnImage1, pointsFromImage2=numpyArrayMatchedPointsOnImage2)

		# Creating and exporting an image with feature points and matching points with lines between the matching ones.
		if (createImageWithPtsAndLines is not None):
			showNPtsAndLines = len(indexPairs) - 1
			if (numOfPtsAndLinesToShow is not None):
				if (numOfPtsAndLinesToShow <= showNPtsAndLines):
					showNPtsAndLines = numOfPtsAndLinesToShow

			self.exportImagePointsAndMatchesGraphically(firstImage, points1, secondImage, points2, indexPairs, createImageWithPtsAndLines, showNPtsAndLines)

		# Returning the points.
		return numpyArrayMatchedPointsOnImage1, numpyArrayMatchedPointsOnImage2

	# Finding robust matches using ransac from points.
	###
	# Use: Input the raw matching points for the image 1 and 2 and the function returns numpy arrays of robust matches for image 1 and 2.
	# If inputcsvFileName is specified, the function looks for the specified csv file name and uses the points that are specified in the csv
	# file for finding robust matches.
	# If outputcsvFileName is specified, the function also creates a csv file of the points for image 1 and 2.
	def find_robust_matches_ranscac(self, inputcsvFileName=None, inputPointsFromImage1=None, inputPointsFromImage2=None, outputcsvFileName=None):
		# Variables to hold the unfiltered points from image1 and image2.
		pointsFromImage1 = []
		pointsFromImage2 = []

		# If output csv file name is specified and points are passed in, use the csv file name as the values by t.
		if (inputcsvFileName is not None):
			pointsFromImage1, pointsFromImage2 = self.importPointsFromCSV(inputcsvFileName)
		else:
			pointsFromImage1 = inputPointsFromImage1
			pointsFromImage2 = inputPointsFromImage2

		# Using ransac to get the robust matches.
		model_robust, inliers = ransac((pointsFromImage1, pointsFromImage2), ProjectiveTransform, min_samples=3, residual_threshold=2, max_trials=2000)

		# Total number of unfiltered points.
		numOfUnfilteredPoints = len(pointsFromImage1)
		# Filtering out the points that are not inliers.
		robustMatchesFromImage1 = []
		robustMatchesFromImage2 = []
		for i in range(numOfUnfilteredPoints):
			# If the point at the index i, it is an inlier.
			if (inliers[i]):
				robustMatchesFromImage1.append(pointsFromImage1[i])
				robustMatchesFromImage2.append(pointsFromImage2[i])

		# Converting the list to numpy array type.
		robustMatchesFromImage1 = numpy.array(robustMatchesFromImage1)
		robustMatchesFromImage2 = numpy.array(robustMatchesFromImage2)

		if (outputcsvFileName is not None):
			self.exportPointsAsCSV(csvFileName=outputcsvFileName, pointsFromImage1=robustMatchesFromImage1, pointsFromImage2=robustMatchesFromImage2)

		# Returning the robust matches.
		return robustMatchesFromImage1, robustMatchesFromImage2

	# Export Points.
	###
	# Use: Takes points from image 1 and 2 and creates csv file with the csv file name that is specified.
	def exportPointsAsCSV(self, csvFileName, pointsFromImage1, pointsFromImage2):
		# Creating a csv file writer.
		with open(csvFileName, "w") as csvfile:
			filewriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
			totalNumberOfPoints = len(pointsFromImage1)
			# Writing in each row with the x and y coordinate points for each image 1 and image 2.
			filewriter.writerow(["Image1-x", "Image1-y", "Image2-x", "Image2-y"])
			for i in range(totalNumberOfPoints):
				filewriter.writerow([pointsFromImage1[i][0], pointsFromImage1[i][1], pointsFromImage2[i][0], pointsFromImage2[i][1]])

	# Import/parse points from specified csv file.
	###
	# Use: Takes csv file name and parses the points that are in the csv file and returns them as
	# numpy arrays.
	def importPointsFromCSV(self, csvFileName):
		# Variables to hold the unfiltered points from image1 and image2.
		pointsFromImage1 = []
		pointsFromImage2 = []

		# Parse the specified csv file.
		with open(csvFileName) as csv_file:
			fileReader = csv.reader(csv_file, delimiter=",")
			rowNamesGoneThrough = False
			for row in fileReader:
				if (rowNamesGoneThrough):
					pointsFromImage1.append([float(row[0]), float(row[1])])
					pointsFromImage2.append([float(row[2]), float(row[3])])
				else:
					rowNamesGoneThrough = True

		pointsFromImage1 = numpy.array(pointsFromImage1)
		pointsFromImage2 = numpy.array(pointsFromImage2)

		return pointsFromImage1, pointsFromImage2

	
	def exportImagePointsAndMatchesGraphically(self, firstImage, firstImageKeyPts, secondImage, secondImageKeyPts, matches, exportImageName, numOfMatchesToShow=None):
		# Shows all of the matches by default.
		numOfPointsToUse = len(matches) - 1
		if (numOfMatchesToShow is not None):
			if (numOfMatchesToShow <= numOfPointsToUse):
				numOfPointsToUse = numOfMatchesToShow

		imageWithDrawings = cv2.drawMatches(firstImage, firstImageKeyPts, secondImage, secondImageKeyPts, matches[:numOfPointsToUse], None, flags=4)
		cv2.imwrite(exportImageName, imageWithDrawings)
