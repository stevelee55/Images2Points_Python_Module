# Written by Steve S. Lee
# 10/14/2018
# stevesl@umich.edu

from Images2Points import Images2Points

# For reading in the images.
import cv2

moduler = Images2Points()

image1Name = "hatcher1.jpg"
image2Name = "hatcher2.jpg"

# Reading in the images using the image names.
img1 = cv2.imread(image1Name, 0)
img2 = cv2.imread(image2Name, 0)

# Getting the matching points from the images.
# Example 1: File name specified, exports the points in csv file.
pointsFromImage1, pointsFromImage2 = moduler.getPointsFromImages(firstImage=img1, secondImage=img2, outputcsvFileName="matchedPointsRaw.csv")

# Getting the matching points from the images.
# Example 2: File name not specified.
pointsFromImage1, pointsFromImage2 = moduler.getPointsFromImages(firstImage=img1, secondImage=img2)


pointsFromImage1, pointsFromImage2 = moduler.getPointsFromImages(firstImage=img1, secondImage=img2, outputcsvFileName="matchedPointsRaw.csv", detectorType="SIFT", crossCheck=True, normType="NORM_L2", createImageWithPtsAndLines="imageWithPtsAndLines.jpg", numOfPtsAndLinesToShow=50)

###

# find_robust_matches_ranscac using the points.
# Option 1: File name specified. Exports the points in csv file.
robustMatchesFromImage1, robustMatchesFromImage2 = moduler.find_robust_matches_ranscac(inputPointsFromImage1=pointsFromImage1, inputPointsFromImage2=pointsFromImage2, outputcsvFileName="robustMatches.csv")

# find_robust_matches_ranscac using the points.
# Option 2: File name not specified.
robustMatchesFromImage1, robustMatchesFromImage2 = moduler.find_robust_matches_ranscac(inputPointsFromImage1=pointsFromImage1, inputPointsFromImage2=pointsFromImage2)


# find_robust_matches_ranscac using the csv file.
# Option 1: Import file name specified, imports the points in matched points that are raw in csv file.
robustMatchesFromImage1, robustMatchesFromImage2 = moduler.find_robust_matches_ranscac(inputcsvFileName="matchedPointsRaw.csv")

# find_robust_matches_ranscac using the csv file.
# Option 2: Import file name not specified, uses the points that were previously matched.
robustMatchesFromImage1, robustMatchesFromImage2 = moduler.find_robust_matches_ranscac(inputPointsFromImage1=pointsFromImage1, inputPointsFromImage2=pointsFromImage2)

# find_robust_matches_ranscac using the csv file.
# Option 3: Export file name specified, imports the points in matched points that are raw in csv file.
robustMatchesFromImage1, robustMatchesFromImage2 = moduler.find_robust_matches_ranscac(inputcsvFileName="matchedPointsRaw.csv", outputcsvFileName="robustMatches.csv")

###
# Example of using the points.
###

# Using the points to find transformation matrix and applying it to the image to warp it.
tform, status = cv2.findHomography(robustMatchesFromImage2, robustMatchesFromImage1, cv2.RANSAC, 5.0)
transimg2 = cv2.warpPerspective(img2, tform, (len(img2[0]), len(img2)))
# Original position.
cv2.imwrite("firstImage.jpg", img1)
# Warped second image but not "shifted" relative to the first image.
cv2.imwrite("secondImage.jpg", transimg2)