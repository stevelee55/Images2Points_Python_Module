from Images2Points import Images2Points

import cv2

moduler = Images2Points()

image1Name = "00000.jpg"
image2Name = "00001.jpg"

# Reading in the images using the image names.
img1 = cv2.imread(image1Name, 0)
img2 = cv2.imread(image2Name, 0)

# Number of points to compute doesn't really do much for now.
pointsFromImage1, pointsFromImage2 = moduler.getPointsFromImages(firstImage=img1, secondImage=img2)

# Export Points as CSV.
moduler.exportPointsAsCSV(pointsFromImage1, pointsFromImage2)

###

# find_robust_matches_ranscac using the points.
robustMatchesFromImage1, robustMatchesFromImage2 = moduler.find_robust_matches_ranscac(inputPointsFromImage1=pointsFromImage1, inputPointsFromImage2=pointsFromImage2)

# Export robust matches as CSV.
moduler.exportPointsAsCSV(robustMatchesFromImage1, robustMatchesFromImage2)

###

# find_robust_matches_ranscac using the csv file.
fileName = "pointsFile.csv" # This file should be in the same directory.
robustMatchesFromImage1FromFile, robustMatchesFromImage2FromFile = moduler.find_robust_matches_ranscac(inputcsvFileName=fileName)

# Export robust matches as CSV.
moduler.exportPointsAsCSV(robustMatchesFromImage1FromFile, robustMatchesFromImage2FromFile)
