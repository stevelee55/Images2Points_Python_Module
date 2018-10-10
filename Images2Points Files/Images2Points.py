# Spec:
# Takes two images as an input and creates csv file that has landmark/regisreted points between the points.
# Presumably it will have 4 columns with first two being coordinates of a point in image 1 and next two the
# coordinates of the corresponding point in second image.

# - - does this need number of points one wants to register as input? if so, let that be input.

# ind_robust_matches_ransac
# find_robust_matches_ranscac might likely use ransac and it loads the txt file and then trims to number.

# https://www.mathworks.com/discovery/ransac.html

# Steps:
# Import images.
# Get the points.
# Create CSV file.
# Export it.

import cv2

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
		points1, firstFeatures = surf.detectAndCompute(firstImage, None)
		points2, secondFeatures = surf.detectAndCompute(secondImage, None)

		# Creating BFMatcher object for feature matching.
		bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)

		# Getting the index pairs that match.
		indexPairs = bf.match(secondFeatures, firstFeatures)

		print(indexPairs)
		cv2.waitKey(0)



	# Export Points.




# Ransac option steps (http://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html):
# Take the src and dst points and ransac them.
# get the inliers and model_robust.
# export them as csv.
# Or create a function that returns inliers and model_robust.