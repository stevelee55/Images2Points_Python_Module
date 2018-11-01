import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img1 = cv.imread("empty1.jpg",1)          # queryImage
img2 = cv.imread("empty2.jpg",1) # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# ratio = 0.8

# goodMatches = []

# for i in range(len(matches) - 1):
# 	if (matches[i].distance < ratio * matches[i].distance):
# 		goodMatches.append(matches[i])


print(len(matches))

# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)
plt.imshow(img3),plt.show()