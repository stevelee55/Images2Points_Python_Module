from Images2Points import Images2Points

import cv2

moduler = Images2Points()

image1Name = "00000.jpg"
image2Name = "00001.jpg"

# Reading in the images using the image names.
img1 = cv2.imread(image1Name)
img2 = cv2.imread(image2Name)

moduler.getPointsFromImages(img1, img2, 3)