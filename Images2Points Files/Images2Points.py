# Spec:
# Takes two images as an input and creates csv file that has landmark/regisreted points between the points.
# Presumably it will have 4 columns with first two being coordinates of a point in image 1 and next two the
# coordinates of the corresponding point in second image.

# - - does this need number of points one wants to register as input? if so, let that be input.

# ind_robust_matches_ransac
# find_robust_matches_ranscac might likely use ransac and it loads the txt file and then trims to number.

# https://www.mathworks.com/discovery/ransac.html

# load the text file csv and export the tform.

# Steps:
# Import images.
# Get the points.
# Create CSV file.
# Export it.

import cv2
import numpy
from skimage.measure import ransac
from skimage.transform import AffineTransform

class Images2Points(object):

	# Initializer.
	def __init__(self):
		print("Images2Points is created.")

	# Get Points.
	# image1 and image2 are image matrix data.
	def getPointsFromImages(self, firstImage, secondImage, numOfPointsToCalculate):

		# Creating the SURF detector.
		surf = cv2.xfeatures2d.SURF_create()

		# Finding keypoints using SURF.
		points1, firstImageFeatures = surf.detectAndCompute(firstImage, None)
		points2, secondImageFeatures = surf.detectAndCompute(secondImage, None)

		# Creating BFMatcher object for feature matching.
		bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)

		# Getting the index pairs that match.
		indexPairs = bf.match(secondImageFeatures, firstImageFeatures)

		# indexPairs[i].queryIdx gives index of points that were matched.
		matchedPointsOnImage2 = numpy.asarray([points2[indexPairs[i].queryIdx] for i in range(len(indexPairs))])
		numpyArrayMatchedPointsOnImage2 = numpy.asarray([matchedPointsOnImage2[i].pt for i in range(len(matchedPointsOnImage2))])
		# matchedPoints have the KeyPoint objects, which an be accesesd by index and .pt.
		# print(matchedPoints[0].pt)
		matchedPointsOnImage1 = numpy.asarray([points1[indexPairs[i].trainIdx] for i in range(len(indexPairs))])
		numpyArrayMatchedPointsOnImage1 = numpy.asarray([matchedPointsOnImage1[i].pt for i in range(len(matchedPointsOnImage1))])
		# Estimating geometric transform.
		# This uses Ransac.
		estimatePair, status = cv2.findHomography(numpyArrayMatchedPointsOnImage2, numpyArrayMatchedPointsOnImage1, cv2.RANSAC, 5.0)

		# # Transformation Matrix.
		tforms = estimatePair


		model_robust, inliers = ransac((numpyArrayMatchedPointsOnImage1, numpyArrayMatchedPointsOnImage2), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100)

		print(model_robust.scale, model_robust.translation, model_robust.rotation)

		print(tforms)

		cv2.waitKey(0)



	# Export Points.




# Ransac option steps (http://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html):
# Take the src and dst points and ransac them.
# get the inliers and model_robust.
# export them as csv.
# Or create a function that returns inliers and model_robust.