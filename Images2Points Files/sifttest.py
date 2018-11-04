import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img1 = cv.imread("law1.jpg",0) # trainImage
img2 = cv.imread("law2.jpg",0) # queryImage

normalizedImg = np.zeros((len(img1), len(img1[0])))
img1 = cv.normalize(img1,  normalizedImg, 0, 100, cv.NORM_MINMAX)

normalizedImg = np.zeros((len(img2), len(img2[0])))
img2 = cv.normalize(img2,  normalizedImg, 0, 100, cv.NORM_MINMAX)

# Noralmize


# Initiate ORB detector
#use orb with norm hamming.
orb = cv.ORB_create()

#// l2 for surf,sift
#// for ORB,BRIEF,etc.
#orb = cv.xfeatures2d.SIFT_create()

# Allow the user to change betwee sift, surf, and orb and write down the benifits and whannot.

# Normalize images.

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
# Cross check is alternative for ratio test.
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# ratio = 0.8

# goodMatches = []

# for i in range(len(matches) - 1):
# 	if (matches[i].distance < ratio * matches[i].distance):
# 		goodMatches.append(matches[i])


print(len(matches))

# Draw first 10 matches.

# min 4 points.

img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:50], None, flags=2)
plt.imshow(img3),plt.show()