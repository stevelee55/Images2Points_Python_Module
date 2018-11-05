Written by Steve S. Lee
stevesl@umich.edu
		
# Images2Points

Python script that can:
1) Detect feature points in two images.
2) Export detected feature points in csv format.
3) Find matching feature points between two images.
4) Export matching feature points in csv format.
5) Export an image that graphically shows detected feature points and matched feature points.

## Getting Started

This module uses SIFT, SURF, and ORB, some of which are patented and cannot be used for commmerical purposes without license. Academic use only.

### Prerequisites

This module requires cv2, numpy, csv, skimage, and matplotlib. Install these before using the module.


## Included Functions

### getPointsFromImages(...)

Required Parameters:
1) firstImage(BGR.)
2) secondImage(BGR.)

Optional Parameters:
1) outputcsvFileName(String. No file exported by default.)
* Pass in name of the csv file that will be exported. Include file extension.
2) detectorType(String. Uses "ORB" by default.)
* Options: "ORB", "SIFT", and "SURF".
3) crossCheck(Bool. Set "True" by default".)
4) normType(String. Uses "NORM_HAMMING" by default.)
* Options: "NORM_HAMMING", "NORM_L1", and "NORM_L2".
5) createImageWithPtsAndLines(String. No image exported by default.)
* Pass in name of the image file that will be exported. Include file extension.
6) numOfPtsAndLinesToShow(Int. Number of total matches and detected feature points by default.)
* Pass in number of points and matches to display in the image with pts and lines.

Example 1:
```
img1 = cv2.imread("image1.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

pointsFromImage1, pointsFromImage2 = moduler.getPointsFromImages(firstImage=img1, secondImage=img2)
```

Example 2:
```
img1 = cv2.imread("image1.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

pointsFromImage1, pointsFromImage2 = moduler.getPointsFromImages(firstImage=img1, secondImage=img2, outputcsvFileName="matchedPointsRaw.csv", detectorType="SIFT", crossCheck=True, normType="NORM_L2", createImageWithPtsAndLines="imageWithPtsAndLines.jpg", numOfPtsAndLinesToShow=10)
```

Recommendations:
1) If ORB detector is used, use "NORM_HAMMING".
2) If SIFT or SURF detector is used, use either "NORM_L1" or "NORM_L2".

### find_robust_matches_ranscac(...)

"Required" Parameters:
1) inputcsvFileName(String.)
* CSV file of points from the function getPointsFromImages(...).
2) inputPointsFromImage1(Array.)
* Array of points from the function getPointsFromImages(...).
3) inputPointsFromImage2(Array.)
* Array of points from the function getPointsFromImages(...).

** It is possible to not pass in "inputcsvFileName", "inputPointsFromImage1", and "inputPointsFromImage2", but to get the desired output, either "inputcsvFileName" or "inputPointsFromImage1", and "inputPointsFromImage2" should be passed in.

## Acknowledgments

* Sponsored by Raj Rao Nadakuditi.
